import torch
import torch.nn as nn

import math
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from einops import repeat, rearrange

import numpy as np

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


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128):
        super().__init__()

        embedding = self._build_embedding(num_steps, embedding_dim / 2)
        
        self.register_buffer("embedding", embedding)

        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies 
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Conv1d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, use_act=True):
        super(Conv1d, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=channels_in,
                                    out_channels=channels_out,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0)   
        self.bn = nn.BatchNorm1d(num_features=channels_out)
        nn.init.kaiming_normal_(self.conv_layer.weight)

        self.use_act = use_act

    def forward(self, x):
        b, c, k, l = x.shape
        h = rearrange(x, "b c k l -> b c (k l)")
        h = self.conv_layer(h)
        h = self.bn(h)
        
        if self.use_act:
            h = F.relu(h)
        
        h = rearrange(h, "b c (k l) -> b c k l", k = k, l = l)
        return h
    

class ResidualBlock(nn.Module):

    def __init__(self, temp_strips_blocks, feat_strips_lenght, number_heads, model_dim, emb_dim, side_dim, do_prob):
        super(ResidualBlock, self).__init__()

        self.temp_strips_blocks = temp_strips_blocks
        self.feat_strips_lenght = feat_strips_lenght

        self.temporal_reshape = Rearrange("b c k (s l) -> (b s k) l c", s = temp_strips_blocks)
        self.temporal_transformer = TransformerLayer(number_heads=number_heads, model_dim=model_dim, do_prob=do_prob)
        
        self.feature_reshape = Rearrange("b c k (s l) -> (b l) (k s) c", s = feat_strips_lenght)
        self.feature_transformer = TransformerLayer(number_heads=number_heads, model_dim=model_dim, do_prob=do_prob)
        
        self.middle_conv = Conv1d(channels_in=model_dim, channels_out=model_dim * 2, kernel_size=1)
        self.out_conv = Conv1d(channels_in=model_dim, channels_out=model_dim * 2, kernel_size=1)

        self.diff_emb_projection = nn.Linear(in_features=emb_dim, out_features=model_dim)
        self.side_conv = Conv1d(channels_in=side_dim, channels_out=2 * model_dim, kernel_size=1)


    def forward(self, x, diff_emb, side_emb):
        b, c, k, l = x.shape
        
        diff_emb = self.diff_emb_projection(diff_emb)
        diff_emb = diff_emb.view(b, -1, 1, 1)
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
    def __init__(self, noise_steps, l, fs, beta_start, beta_end, temp_strips_blocks, feat_strips_lenght, num_features, num_res_blocks, number_heads, model_dim, emb_dim, time_dim, feat_dim, do_prob):
        super(CSDI, self).__init__()
        
        self.num_res_blocks = num_res_blocks
        
        self.l = l
        self.fs = fs

        self.time_dim = time_dim
        self.num_features = num_features

        self.noise_steps = noise_steps
        beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, noise_steps) ** 2
        alpha = 1 - beta
        alpha_hat = np.cumprod(alpha)

        self.beta = torch.tensor(beta).float().to(device).view(-1, 1, 1)
        self.alpha_torch = torch.tensor(alpha).float().to(device).view(-1, 1, 1)
        self.alpha_hat_torch = torch.tensor(alpha_hat).float().to(device).view(-1, 1, 1)
        

        self.x_conv = Conv1d(channels_in=2, channels_out=model_dim, kernel_size=1)
        self.diff_embedding = DiffusionEmbedding(num_steps=noise_steps, embedding_dim=emb_dim)

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
                            Conv1d(channels_in=model_dim, channels_out=1, kernel_size=1, use_act=False))

        self.feat_emb = nn.Embedding(num_embeddings=num_features, embedding_dim=feat_dim)
        
        time_embeddings = self.get_time_embeddings()
        self.register_buffer('time_embeddings', time_embeddings)

        
    def forward(self, x, t, mask_co):
        b, _, k, l = x.shape
        
        x = self.x_conv(x)

        diff_emb = self.diff_embedding(t)   
        side_emb = self.get_side_embeddings(x)
        
        side_emb = torch.cat([side_emb, mask_co], dim = -1)
        side_emb = rearrange(side_emb, "b k l c -> b c k l")

        cum_out = []
        
        for layer in self.net:
            res, out = layer(x, diff_emb, side_emb)
            x = res
            cum_out.append(out)

        cum_out =  torch.sum(torch.stack(cum_out), dim=0) / math.sqrt(self.num_res_blocks)
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


    def impute(self, obs, mask):
        self.eval()

        K, L = obs.shape
        x_t = np.random.randn(K, L) * mask
        
        mask_co = (1 - mask)
        x_co = obs * mask_co

        x_co = torch.tensor(x_co).float().view(1, 1, K, L).to(device)
        x_t = torch.tensor(x_t).float().view(1, 1, K, L).to(device)
        
        mask_co = torch.tensor(mask_co).float().view(1, K, L, 1).to(device)
        mask = torch.tensor(mask).float().view(1, 1, K, L).to(device)


        with torch.no_grad():
        
            for t in reversed(range(self.noise_steps)):

                x = torch.cat([x_co, x_t], dim=1).to(device)
                inp_t = torch.tensor(t).long().unsqueeze(0).to(device)

                pred_noise = self(x, inp_t, mask_co).view(1, 1, K, L)

                first_coeff = 1 / torch.sqrt(self.alpha_torch[t])
                second_coeff = (1 - self.alpha_torch[t]) / torch.sqrt(1 - self.alpha_hat_torch[t])

                x_t =  first_coeff * (x_t - second_coeff * pred_noise)
                
                if t > 0:
                    noise = torch.randn_like(x_t) * mask
                    sigma = ((1.0 - self.alpha_hat_torch[t - 1]) / (1.0 - self.alpha_hat_torch[t]) * self.beta[t]) ** 0.5
                    x_t += sigma * noise
               
                x_t = x_t * mask

        return x_t.detach().cpu(), x_co.detach().cpu()
