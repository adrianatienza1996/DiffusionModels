import torch
import torch.nn as nn

import math
from einops.layers.torch import Rearrange

from einops import repeat, rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


class MultiHead_Attention(nn.Module):

    def __init__(self, model_dim, number_heads, do_prob):
        super(MultiHead_Attention, self).__init__()
        self.number_heads = number_heads

        self.scale_factor = 1 / ((model_dim / number_heads) ** 0.5)
        self.att_drop_out = nn.Dropout(do_prob)
        self.output_drop_out = nn.Dropout(do_prob)

        self.block_output = nn.Linear(model_dim, model_dim)

        self.split_head = Rearrange('b l (h d) -> b h l d', h = self.number_heads)
        self.split_head_t = Rearrange('b l (h d) -> b h d l', h = self.number_heads)
        self.concat = Rearrange('b h l d -> b l (h d)') 

        self.x_to_q = nn.Linear(model_dim, model_dim)
        self.x_to_k = nn.Linear(model_dim, model_dim)
        self.x_to_v = nn.Linear(model_dim, model_dim)


    def forward(self, q, k, v):
        # q, k and v with shape (batch_size, seq_len, embedding_dimension)
        q = self.split_head(self.x_to_q(q))
        k_transpose = self.split_head_t(self.x_to_k(k))
        v = self.split_head(self.x_to_v(v))

        attention = torch.matmul(q, k_transpose)
        attention = attention * self.scale_factor
        
        attention = attention.softmax(-1)
        output = self.att_drop_out(attention)
        output = torch.matmul(output, v)
        output = self.concat(output)
        output = self.block_output(output)
        return attention, self.output_drop_out(output)


class FeedForwardNet(nn.Module):
    def __init__(self, model_dim, do_prob, wide_factor=1):
        super(FeedForwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim * wide_factor),
            nn.GELU(),
            nn.Dropout(do_prob),
            nn.Linear(model_dim * wide_factor, model_dim),
            nn.Dropout(do_prob)
        )

    def forward(self, x):
        return self.net(x)


class Add_and_Norm(nn.Module):
    
    def __init__(self, model_dim):
        super(Add_and_Norm, self).__init__()
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, res):
        return self.norm(x + res)


class TransformerLayer(nn.Module):
    def __init__(self, number_heads, model_dim, do_prob):
        super(TransformerLayer, self).__init__()
        self.mh_atten_block = MultiHead_Attention(number_heads=number_heads, 
                                                  model_dim=model_dim,
                                                  do_prob=do_prob)
        
        self.add_norm_mh = Add_and_Norm(model_dim=model_dim)
        
        self.ffn = FeedForwardNet(model_dim=model_dim, 
                                  do_prob=do_prob)

        self.add_norm_ffn = Add_and_Norm(model_dim=model_dim)

    def forward(self, x):
        res = x
        attention, h = self.mh_atten_block(x, x, x)
        h = self.add_norm_mh(h, res)
        
        res = h
        h = self.ffn(h)

        return self.add_norm_ffn(h, res)


class ResidualBlock(nn.Module):

    def __init__(self, number_heads, model_dim, emb_dim, side_dim, do_prob):
        super(ResidualBlock, self).__init__()
        self.temporal_reshape = Rearrange("b c k l -> (b l) k c")
        self.temporal_transformer = TransformerLayer(number_heads=number_heads, model_dim=model_dim, do_prob=do_prob)
        
        self.feature_reshape = Rearrange("b c k l -> (b k) l c")
        self.feature_transformer = TransformerLayer(number_heads=number_heads, model_dim=model_dim, do_prob=do_prob)
        
        self.middle_conv = nn.Conv2d(in_channels=model_dim, out_channels=model_dim * 2, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Conv2d(in_channels=model_dim, out_channels=model_dim * 2, kernel_size=1, stride=1, padding=0)

        self.diff_emb_conv = nn.Conv2d(in_channels=emb_dim, out_channels=model_dim, kernel_size=1, stride=1, padding=0)
        self.side_conv = nn.Conv2d(in_channels=side_dim, out_channels=2 * model_dim, kernel_size=1, stride=1, padding=0)


    def forward(self, x, diff_emb, side_emb):
        b, c, k, l = x.shape

        diff_emb = self.diff_emb_conv(diff_emb).squeeze()
        diff_emb = repeat(diff_emb, "b c -> b c k l", k = k, l = l)
        
        h = x + diff_emb
        h = self.temporal_reshape(h)
        h = self.temporal_transformer(h)
        h = rearrange(h, "(b l) k c -> b c k l", b = b, l = l)

        h = self.feature_reshape(h)
        h = self.feature_transformer(h)
        h = rearrange(h, "(b k) l c -> b c k l", b = b, k = k)
        
        h = self.middle_conv(h)

        side_emb = self.side_conv(side_emb)
        h = h + side_emb

        gate, filter = torch.chunk(h, 2, dim=1)     
        h = torch.sigmoid(gate) * torch.tanh(filter)
        
        h = self.out_conv(h)
        res, out = torch.chunk(h, chunks=2, dim=1)
        res = res + x / math.sqrt(2.0)

        return res, out    


class CSDI(nn.Module):
    def __init__(self, num_res_blocks, number_heads, model_dim, emb_dim, side_dim, do_prob):
        super(CSDI, self).__init__()
        
        self.num_res_blocks = num_res_blocks

        self.x_conv = nn.Conv2d(in_channels=2, out_channels=model_dim, kernel_size=1, stride=1, padding=0)
        self.diff_linear = nn.Sequential(
                                nn.Linear(in_features=emb_dim, out_features=emb_dim),
                                nn.SiLU())

        net = nn.ModuleList()
        for _ in range(num_res_blocks):
            net.append(ResidualBlock(number_heads=number_heads,
                                     model_dim=model_dim,
                                     emb_dim=emb_dim,
                                     side_dim=side_dim,
                                     do_prob=do_prob))

        self.net = net
        self.out_net = nn.Sequential(
                            nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=1, stride=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=model_dim, out_channels=1, kernel_size=1, stride=1, padding=0))

        
    def forward(self, xco, xta, diff_emb, time_emb, feature_emb, mask):
        b, _, k, l = xco.shape

        x = torch.cat([xco, xta], dim=1)
        x = self.x_conv(x)

        diff_emb = self.diff_linear(diff_emb)
        diff_emb = rearrange(diff_emb, "b k l c -> b c k l")

        time_emb = repeat(time_emb, "b l c -> b k l c", k = k)
        feature_emb = repeat(feature_emb, "b k c -> b k l c", l = l)
        side_emb = torch.cat([time_emb, feature_emb, mask], dim = -1)
        side_emb = rearrange(side_emb, "b k l c -> b c k l")

        cum_out = 0
        
        for layer in self.net:
            res, out = layer(x, diff_emb, side_emb)
            x = res
            cum_out = out + cum_out

        cum_out = cum_out / math.sqrt(self.num_res_blocks)
        output = self.out_net(cum_out)
        
        output = rearrange(output, "b c k l -> b k l c")
        output = output * mask

        return output
        