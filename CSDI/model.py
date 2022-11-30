import torch
import torch.nn as nn

import math
from einops.layers.torch import Rearrange

from einops import repeat, rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        return self.output_drop_out(output)


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
        h = self.mh_atten_block(x, x, x)
        h = self.add_norm_mh(h, res)
        
        res = h
        h = self.ffn(h)

        return self.add_norm_ffn(h, res)


class ResidualBlock(nn.Module):

    def __init__(self, strips_lenght, number_heads, model_dim, emb_dim, side_dim, do_prob):
        super(ResidualBlock, self).__init__()

        self.strips_lenght = strips_lenght

        self.temporal_reshape = Rearrange("b c k (l s) -> (b k s) l c", s = strips_lenght)
        self.temporal_transformer = TransformerLayer(number_heads=number_heads, model_dim=model_dim, do_prob=do_prob)
        
        self.feature_reshape = Rearrange("b c k (l s) -> (b l) (k s) c", s = strips_lenght)
        self.feature_transformer = TransformerLayer(number_heads=number_heads, model_dim=model_dim, do_prob=do_prob)
        
        self.middle_conv = nn.Conv2d(in_channels=model_dim, out_channels=model_dim * 2, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Conv2d(in_channels=model_dim, out_channels=model_dim * 2, kernel_size=1, stride=1, padding=0)

        self.diff_emb_conv = nn.Conv2d(in_channels=emb_dim, out_channels=model_dim, kernel_size=1, stride=1, padding=0)
        self.side_conv = nn.Conv2d(in_channels=side_dim, out_channels=2 * model_dim, kernel_size=1, stride=1, padding=0)


    def forward(self, x, diff_emb, side_emb):
        b, c, k, l = x.shape

        diff_emb = repeat(diff_emb, "b c -> b c d1 d2", d1 = 1, d2 = 1) 
        diff_emb = self.diff_emb_conv(diff_emb)
        diff_emb = repeat(diff_emb, "b c d1 d2-> b c (d1 k) (d2 l)", k = k, l = l)
        
        h = x + diff_emb
        h = self.temporal_reshape(h)
        h = self.temporal_transformer(h)
        h = rearrange(h, "(b k s) l c -> b c k (l s)", b = b, l = l // self.strips_lenght, s = self.strips_lenght)

        h = self.feature_reshape(h)
        h = self.feature_transformer(h)
        h = rearrange(h, "(b l) (k s) c -> b c k (l s)", b = b, k = k, s = self.strips_lenght)

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
    def __init__(self, l, fs, strips_lenght, num_features, num_res_blocks, number_heads, model_dim, emb_dim, time_dim, feat_dim, do_prob):
        super(CSDI, self).__init__()
        
        self.num_res_blocks = num_res_blocks
        
        self.l = l
        self.fs = fs

        self.time_dim = time_dim
        self.num_features = num_features

        self.x_conv = nn.Conv2d(in_channels=2, out_channels=model_dim, kernel_size=1, stride=1, padding=0)
        self.diff_linear = nn.Sequential(
                                nn.Linear(in_features=emb_dim, out_features=emb_dim),
                                nn.SiLU())
        net = nn.ModuleList()
        for _ in range(num_res_blocks):
            net.append(ResidualBlock(strips_lenght=strips_lenght,
                                     number_heads=number_heads,
                                     model_dim=model_dim,
                                     emb_dim=emb_dim,
                                     side_dim=time_dim + feat_dim + 1,
                                     do_prob=do_prob))

        self.net = net
        self.out_net = nn.Sequential(
                            nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=1, stride=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=model_dim, out_channels=1, kernel_size=1, stride=1, padding=0))

        self.feat_emb = nn.Embedding(num_embeddings=num_features, embedding_dim=feat_dim)
        
        time_embeddings = self.get_time_embeddings()
        self.register_buffer('time_embeddings', time_embeddings)

        
    def forward(self, xco, xta, diff_emb, mask):
        b, _, k, l = xco.shape

        x = torch.cat([xco, xta], dim=1)
        x = self.x_conv(x)

        diff_emb = self.diff_linear(diff_emb)
        side_emb = self.get_side_embeddings(x)
        
        side_emb = torch.cat([side_emb, mask], dim = -1)
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
        

    def get_time_embeddings(self):
        position_id = torch.arange(0, self.fs * self.l).unsqueeze(1)

        freq = repeat(torch.arange(0, self.time_dim // 2, dtype=torch.float), "l -> (k l)", k = 2) / (self.time_dim / 2)
        freq = torch.pow(10000, -freq)

        positional_encodings_table = freq * position_id

        positional_encodings_table[:, :self.time_dim//2] = torch.sin(positional_encodings_table[:, :self.time_dim//2]) 
        positional_encodings_table[:, self.time_dim//2:] = torch.cos(positional_encodings_table[:, self.time_dim//2:]) 
        return positional_encodings_table

    def get_side_embeddings(self, x):
        b, c, k, l = x.shape
        time_embeddings = self.time_embeddings.to(device)
        time_embeddings = repeat(time_embeddings, "l c -> k l c", k = k)
        
        feature_embeddings = self.feat_emb(torch.arange(self.num_features).to(device))
        feature_embeddings = repeat(feature_embeddings, "k c -> k l c", l = l)

        side_embeddings = torch.cat([time_embeddings, feature_embeddings], dim=-1)
        side_embeddings = repeat(side_embeddings, "k l c -> b k l c", b = b)
        return side_embeddings