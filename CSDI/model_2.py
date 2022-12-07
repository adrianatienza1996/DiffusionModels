import torch
import torch.nn as nn

import math
from einops.layers.torch import Rearrange

from einops import repeat, rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_torch_trans(number_heads,  model_dim, layers=1):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=model_dim, nhead=number_heads, dim_feedforward=model_dim, activation="gelu")

    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class Conv1d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(Conv1d, super).__init__()
        self.conv_layer = nn.Conv1d(in_channels=channels_in,
                                    out_channels=channels_out,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0)   
        self.bn = nn.BatchNorm1d(num_features=channels_out)
        nn.init.kaiming_normal_(self.conv_layer.weight)

    def forward(self, x):
        b, c, k, l = x.shape
        h = rearrange(x, "b c k l -> b c (k l)")
        h = self.conv_layer(h)
        h = self.bn(h)
        h = nn.ReLU(h)
        h = rearrange(h, "b c (k l) -> b c k l", k = k, l = l)
        return h
    

class ResidualBlock(nn.Module):

    def __init__(self, temp_strips_blocks, feat_strips_lenght, number_heads, model_dim, emb_dim, side_dim, do_prob):
        super(ResidualBlock, self).__init__()

        self.temp_strips_blocks = temp_strips_blocks
        self.feat_strips_lenght = feat_strips_lenght

        self.temporal_reshape = Rearrange("b c k (s l) -> (b s k) l c", s = temp_strips_blocks)
        self.temporal_transformer = get_torch_trans(number_heads=number_heads, model_dim=model_dim)
        
        self.feature_reshape = Rearrange("b c k (l s) -> (b l) (k s) c", s = feat_strips_lenght)
        self.feature_transformer = get_torch_trans(number_heads=number_heads, model_dim=model_dim)
        
        self.middle_conv = Conv1d(channels_in=model_dim, channels_out=model_dim * 2, kernel_size=1)
        self.out_conv = Conv1d(channels_in=model_dim, channels_out=model_dim * 2, kernel_size=1)

        self.diff_emb_conv = Conv1d(channels_in=emb_dim, channels_out=model_dim, kernel_size=1)
        self.side_conv = Conv1d(channels_in=side_dim, channels_out=2 * model_dim, kernel_size=1)


    def forward(self, x, diff_emb, side_emb):
        b, c, k, l = x.shape

        diff_emb = repeat(diff_emb, "b c -> b c d1 d2", d1 = 1, d2 = 1) 
        diff_emb = self.diff_emb_conv(diff_emb)
        diff_emb = repeat(diff_emb, "b c d1 d2-> b c (d1 k) (d2 l)", k = k, l = l)
        
        h = x + diff_emb
        h = self.temporal_reshape(h)
        h = self.temporal_transformer(h)
        h = rearrange(h, "(b s k) l c -> b c k (s l)", b = b, l = l // self.temp_strips_blocks, s = self.temp_strips_blocks)

        h = self.feature_reshape(h)
        h = self.feature_transformer(h)
        h = rearrange(h, "(b l) (k s) c -> b c k (l s)", b = b, k = k, s = self.feat_strips_lenght)

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
    def __init__(self, l, fs, temp_strips_blocks, feat_strips_lenght, num_features, num_res_blocks, number_heads, model_dim, emb_dim, time_dim, feat_dim, do_prob):
        super(CSDI, self).__init__()
        
        self.num_res_blocks = num_res_blocks
        
        self.l = l
        self.fs = fs

        self.time_dim = time_dim
        self.num_features = num_features

        self.x_conv = Conv1d(channels_in=2, channels_out=model_dim, kernel_size=1)
        self.diff_linear = nn.Sequential(
                                nn.Linear(in_features=emb_dim, out_features=emb_dim),
                                nn.SiLU())
        net = nn.ModuleList()
        for _ in range(num_res_blocks):
            net.append(ResidualBlock(temp_strips_blocks=temp_strips_blocks,
                                     feat_strips_lenght=feat_strips_lenght,
                                     number_heads=number_heads,
                                     model_dim=model_dim,
                                     emb_dim=emb_dim,
                                     side_dim=time_dim + feat_dim + 1,
                                     do_prob=do_prob))

        self.net = net
        self.out_net = nn.Sequential(
                            Conv1d(channels_in=model_dim, channels_out=model_dim, kernel_size=1),
                            nn.ReLU(),
                            Conv1d(channels_in=model_dim, channels_out=1, kernel_size=1))

        self.feat_emb = nn.Embedding(num_embeddings=num_features, embedding_dim=feat_dim)
        
        time_embeddings = self.get_time_embeddings()
        self.register_buffer('time_embeddings', time_embeddings)

        
    def forward(self, xco, xta, diff_emb, mask_co):
        b, _, k, l = xco.shape

        x = torch.cat([xco, xta], dim=1)
        x = self.x_conv(x)

        diff_emb = self.diff_linear(diff_emb)   
        side_emb = self.get_side_embeddings(x)
        
        side_emb = torch.cat([side_emb, mask_co], dim = -1)
        side_emb = rearrange(side_emb, "b k l c -> b c k l")

        cum_out = 0
        
        for layer in self.net:
            res, out = layer(x, diff_emb, side_emb)
            x = res
            cum_out = out + cum_out

        cum_out = cum_out / math.sqrt(self.num_res_blocks)
        output = self.out_net(cum_out)
        
        output = rearrange(output, "b c k l -> b k l c")

        return output * (1 - mask_co)
        

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