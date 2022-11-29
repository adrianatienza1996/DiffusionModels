import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

from DiffusionScheme import Diffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

class MIT_PSG_Diffusion_Dataset(Dataset):
    
    def __init__(self, data_path, ecg_file, bp_file, eeg_file, resp_file, fs=75, l=10, noise_steps=50, b_low=0.0001, b_high=0.5, diff_dim=128, device="cuda"):

        ecg_signals_path = os.path.join(data_path, ecg_file)
        bp_signals_path = os.path.join(data_path, bp_file)
        eeg_signals_path = os.path.join(data_path, eeg_file)
        resp_signals_path = os.path.join(data_path, resp_file)

        print("Loading CSV files")

        self.ecg_csv = np.array(pd.read_csv(ecg_signals_path), dtype=np.float32)
        self.bp_csv = np.array(pd.read_csv(bp_signals_path), dtype=np.float32)
        self.eeg_csv = np.array(pd.read_csv(eeg_signals_path), dtype=np.float32)
        self.resp_csv = np.array(pd.read_csv(resp_signals_path), dtype=np.float32)
        
        print("Loading CSV files Finished")
        
        self.diffusion = Diffusion(noise_steps=noise_steps, 
                                   beta_start=b_low, 
                                   beta_end=b_high, 
                                   l=l, 
                                   fs=fs,
                                   enc_dim=diff_dim)
        
    def __len__(self):
        return len(self.ecg_csv)

    def __getitem__(self, ix):
        tmp_ecg = self.ecg_csv[ix, :].reshape(1, -1)
        tmp_bp = self.bp_csv[ix, :].reshape(1, -1)
        tmp_eeg = self.eeg_csv[ix, :].reshape(1, -1)
        tmp_resp = self.resp_csv[ix, :].reshape(1, -1)

        tmp_x = np.concatenate([tmp_ecg, tmp_bp, tmp_eeg, tmp_resp], axis = 0)

        x_t, x_co, noise, mask, diff_emb = self.diffusion.noise_signal(tmp_x) 
        diff_emb = diff_emb.reshape(1, 1, -1)

        x_co = torch.tensor(x_co).unsqueeze(0).float().to(device)
        x_t = torch.tensor(x_t).unsqueeze(0).float().to(device)
        noise = torch.tensor(noise).float().to(device)
        mask = torch.tensor(mask).unsqueeze(-1).float().to(device)
        diff_emb = torch.tensor(diff_emb).float().to(device)
        
        return x_co, x_t, noise, mask, diff_emb