import torch
import math

class Diffusion:
    
    def __init__(self, noise_steps=50, beta_start=0.0001, beta_end=0.5, l=10, fs=250, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.l = l
        self.fs = fs

        self.beta_schedule = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.device = device

    def prepare_noise_schedule(self):

        def cuadratic_scheme(self, t):
            first_term = (self.noise_steps - t) * math.sqrt(self.beta_start) / (self.noise_steps - 1)
            second_term = (t - 1) * math.sqrt(self.beta_end) / (self.noise_steps - 1)
            return math.pow(first_term + second_term, 2)

        beta_schedule = [cuadratic_scheme(self, t) for t in range(self.noise_steps)]
        return torch.tensor(beta_schedule)

    def noise_signal(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self):
        return torch.randint(low=1, high=self.noise_steps)
