import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from legacy_code.s4 import S4Layer

import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from legacy_code.s4 import S4Layer

import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim_in, embedding_dim_out):
        super().__init__()

        embedding = self._build_embedding(num_steps, embedding_dim_in / 2)
        
        self.register_buffer("embedding", embedding)

        self.projection1 = nn.Linear(embedding_dim_in, embedding_dim_out)
        self.projection2 = nn.Linear(embedding_dim_out, embedding_dim_out)

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
    def __init__(self, channels_in, channels_out, kernel_size=3, dilation=1, use_act=False):
        super(Conv1d, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv_layer = nn.Conv1d(in_channels=channels_in,
                                    out_channels=channels_out,
                                    kernel_size=kernel_size,
                                    padding=self.padding,   
                                    dilation=dilation)
                                    
        self.bn = nn.BatchNorm1d(num_features=channels_out)
        nn.init.kaiming_normal_(self.conv_layer.weight)

        self.use_act = use_act

    def forward(self, x):
        h = self.conv_layer(x)
        h = self.bn(h)
        
        if self.use_act:
            h = F.relu(h)
        return h



class ResBlock(nn.Module):
    def __init__(self, cond_channels, embed_dim, channels_dim, skip_channels_dim, s4_params):
        super(ResBlock, self).__init__()
        
        self.channels_dim = channels_dim
        self.skip_channels_dim = skip_channels_dim

        self.emb_projection = nn.Linear(in_features=embed_dim, out_features=channels_dim)
        self.cond_projection = Conv1d(channels_in=cond_channels, channels_out = 2 * channels_dim, kernel_size=1)

        self.middle_net = nn.Sequential(
                                Conv1d(channels_in=channels_dim, channels_out=channels_dim * 2),
                                Rearrange("b c l -> b l c"),
                                S4Layer(features = 2 * channels_dim, 
                                        lmax = s4_params["s4_lmax"], 
                                        N = s4_params["s4_d_state"], 
                                        dropout = s4_params["s4_dropout"], 
                                        bidirectional = s4_params["s4_bidirectional"],
                                        layer_norm = s4_params["s4_layernorm"]))

        self.last_s4 = nn.Sequential(
                                Rearrange("b c l -> b l c"),
                                S4Layer(features = 2 * channels_dim, 
                                        lmax= s4_params["s4_lmax"], 
                                        N = s4_params["s4_d_state"], 
                                        dropout = s4_params["s4_dropout"], 
                                        bidirectional = s4_params["s4_bidirectional"],
                                        layer_norm = s4_params["s4_layernorm"]))

        self.res_conv = Conv1d(channels_in=channels_dim, channels_out=channels_dim, kernel_size=1)
        self.skip_conv = Conv1d(channels_in=channels_dim, channels_out=skip_channels_dim, kernel_size=1)


    def forward(self, x_t, x_con, diff_embeddings):
        B, C, L = x_t.shape
        
        diff_embeddings = self.emb_projection(diff_embeddings)

        diff_embeddings = repeat(diff_embeddings, "b c -> b c l", l = L)
        h = x_t + diff_embeddings

        h = self.middle_net(h)
        h = rearrange(h, "b l c-> b c l", l = L)
        x_con = self.cond_projection(x_con)
        h = h + x_con

        h = self.last_s4(h)

        gate, filter = torch.chunk(h, 2, dim=2)     
        h = torch.sigmoid(gate) * torch.tanh(filter)
        h = rearrange(h, "b l c-> b c l")
        
        res = self.res_conv(h)
        res = (x_t + res) * math.sqrt(0.5)
        
        skip = self.skip_conv(h)

        return res, skip


class SSSD(nn.Module):
    def __init__(self, num_noise_steps, beta_start, beta_end, num_blocks, features, embed_dim_in, embed_dim_out, channels_dim, skip_channels_dim, s4_params, device="cuda"):
        super(SSSD, self).__init__()

        self.noise_steps = num_noise_steps
        beta = np.linspace(beta_start, beta_end, num_noise_steps)
        alpha = 1 - beta
        alpha_hat = np.cumprod(alpha)

        self.beta = torch.tensor(beta).float().to(device).view(-1, 1, 1)
        self.alpha_torch = torch.tensor(alpha).float().to(device).view(-1, 1, 1)
        self.alpha_hat_torch = torch.tensor(alpha_hat).float().to(device).view(-1, 1, 1)

        self.num_blocks = num_blocks

        self.xt_conv = Conv1d(channels_in=features, channels_out=channels_dim, use_act=True)
        self.diff_emb = DiffusionEmbedding(num_noise_steps, embedding_dim_in=embed_dim_in, embedding_dim_out=embed_dim_out)

        net = nn.ModuleList()
        for _ in range(num_blocks):
            net.append(ResBlock(cond_channels = features * 2, 
                                channels_dim=channels_dim, 
                                skip_channels_dim=skip_channels_dim, 
                                embed_dim=embed_dim_out, 
                                s4_params=s4_params))

        self.net = net

        self.output_net = nn.Sequential(
                                Conv1d(channels_in=skip_channels_dim, channels_out=skip_channels_dim, kernel_size=1, use_act=True),
                                Conv1d(channels_in=skip_channels_dim, channels_out=features, kernel_size=1))
    def forward(self, x_t, x_co, mask, diff_step):

        x_co = torch.cat([x_co, mask], dim = 1)
        diff_embeddings = self.diff_emb(diff_step)
        x_t = self.xt_conv(x_t)

        cum_out = []
        
        for layer in self.net:
            res, out = layer(x_t, x_co, diff_embeddings)
            x = res
            cum_out.append(out)

        cum_out =  torch.sum(torch.stack(cum_out), dim=0) / math.sqrt(self.num_blocks)
        output = self.output_net(cum_out)
        return output.squeeze()


    def impute(self, obs, mask):
        self.eval()

        K, L = obs.shape
        x_t = np.random.randn(K, L) * mask
        
        mask_co = (1 - mask)
        x_co = obs * mask_co

        x_co = torch.tensor(x_co).float().view(1, K, L).to(device)
        x_t = torch.tensor(x_t).float().view(1, K, L).to(device)
        
        mask_co = torch.tensor(mask_co).float().view(1, K, L).to(device)
        mask = torch.tensor(mask).float().view(1, K, L).to(device)


        with torch.no_grad():
        
            for t in reversed(range(self.noise_steps)):


                inp_t = torch.tensor(t).unsqueeze(0).long().to(device)

                pred_noise = self(x_t, x_co, mask_co, inp_t).view(1, K, L)

                first_coeff = 1 / torch.sqrt(self.alpha_torch[t])
                second_coeff = (1 - self.alpha_torch[t]) / torch.sqrt(1 - self.alpha_hat_torch[t])

                x_t =  first_coeff * (x_t - second_coeff * pred_noise)
                
                if t > 0:
                    noise = torch.randn_like(x_t) * mask
                    sigma = ((1.0 - self.alpha_hat_torch[t - 1]) / (1.0 - self.alpha_hat_torch[t]) * self.beta[t]) ** 0.5
                    x_t += sigma * noise
               
                x_t = x_t * mask

        return x_t.detach().cpu(), x_co.detach().cpu()

