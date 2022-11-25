import torch
import math
import numpy as np

class Diffusion:
    
    def __init__(self, noise_steps=50, beta_start=0.0001, beta_end=0.5, l=10, fs=250):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.l = l
        self.fs = fs

        self.beta_schedule = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta_schedule
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):

        def cuadratic_scheme(self, t):
            first_term = (self.noise_steps - t) * math.sqrt(self.beta_start) / (self.noise_steps - 1)
            second_term = (t - 1) * math.sqrt(self.beta_end) / (self.noise_steps - 1)
            return math.pow(first_term + second_term, 2)

        beta_schedule = [cuadratic_scheme(self, t) for t in range(self.noise_steps)]
        return torch.tensor(beta_schedule)

    def noise_signal(self, x):
        # x is (4 x fs * l) dim . (ECG, BP, EEG, Resp)
         
        def sample_t_and_mask(self):
            t = np.random.randint(1, self.noise_steps)

            # Hidden percentaje between (10%, 90%) of the ECG Signal
            first_index = np.random.randint(0, int(np.round(0.9 * self.fs * self.l)))
            second_index = np.random.randint(first_index + int(np.round(0.1 * self.fs * self.l)), int(self.fs * self.l))

            mask = np.zeros_like(x)
            mask[0, first_index:second_index] = 1

            return t, mask

        t, mask = sample_t_and_mask(self)

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        
        Ɛ = torch.randn_like(x) * mask
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ, mask

