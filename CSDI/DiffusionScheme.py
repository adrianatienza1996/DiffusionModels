import torch
import math
import numpy as np

class Diffusion:
    
    def __init__(self, noise_steps=50, beta_start=0.0001, beta_end=0.5, l=10, fs=250, enc_dim=128):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.enc_dim = enc_dim

        self.l = l
        self.fs = fs

        self.beta_schedule = self.prepare_noise_schedule()
        self.time_embeddings = self.create_time_embeddings()

        self.alpha = 1. - self.beta_schedule
        self.alpha_hat = np.cumprod(self.alpha, axis=0)

    def prepare_noise_schedule(self):

        def cuadratic_scheme(self, t):
            first_term = (self.noise_steps - t) * math.sqrt(self.beta_start) / (self.noise_steps - 1)
            second_term = (t - 1) * math.sqrt(self.beta_end) / (self.noise_steps - 1)
            return math.pow(first_term + second_term, 2)

        beta_schedule = [cuadratic_scheme(self, t) for t in range(self.noise_steps)]
        return np.array(beta_schedule)

    def create_time_embeddings(self):
        sin_emb = np.zeros((self.noise_steps, self.enc_dim//2))
        cos_emb = np.zeros((self.noise_steps, self.enc_dim//2))

        for i in range(sin_emb.shape[0]):
            for j in range(sin_emb.shape[1]):
                sin_emb[i, j] = np.sin(math.pow(10, (j * 4 / 63)) * i)
                cos_emb[i, j] = np.cos(math.pow(10, (j * 4 / 63)) * i)

        time_embeddings = np.concatenate([sin_emb, cos_emb], axis = 1)
        return time_embeddings

    def noise_signal(self, x):
        # x is (4 x fs * l) dim . (ECG, BP, EEG, Resp)
         
        def sample_t_and_mask(self):
            t = np.random.randint(1, self.noise_steps)

            # Hidden percentaje between (10%, 90%) of the ECG Signal
            first_index = np.random.randint(0, int(np.round(0.40 * self.fs * self.l)))
            second_index = np.random.randint(first_index + int(np.round(0.1 * self.fs * self.l)), int(self.fs * self.l))

            mask = np.zeros_like(x)
            mask[0, first_index:second_index] = 1

            return t, mask

        t, mask = sample_t_and_mask(self)

        sqrt_alpha_hat = np.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = np.sqrt(1 - self.alpha_hat[t])
       
        Ɛ = np.random.randn(x.shape[0], x.shape[1]) * mask
        tmp_time_embeddings = self.time_embeddings[t, :]

        x_co = x * (1 - mask) 
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ
        x_t = x_t * mask
        
        return x_t, x_co, Ɛ, mask, tmp_time_embeddings

