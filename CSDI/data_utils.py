import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

from DiffusionScheme import Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"

class MIT_PSG_Diffusion_Dataset(Dataset):
    
    def __init__(self, ecg_signals, bp_signals, eeg_signals, resp_signals, fs=75, l=10, noise_steps=50, b_low=0.0001, b_high=0.5, diff_dim=128, device="cuda"):

        self.ecg_csv = np.array(ecg_signals, dtype=np.float32)
        self.bp_csv = np.array(bp_signals, dtype=np.float32)
        self.eeg_csv = np.array(eeg_signals, dtype=np.float32)
        self.resp_csv = np.array(resp_signals, dtype=np.float32)
    
        
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


def get_data_loaders(data_path, ecg_file, bp_file, eeg_file, resp_file, anno_file, test_studies, batch_size=16):

    ecg_signals_path = os.path.join(data_path, ecg_file)
    bp_signals_path = os.path.join(data_path, bp_file)
    eeg_signals_path = os.path.join(data_path, eeg_file)
    resp_signals_path = os.path.join(data_path, resp_file)
    anno_path = os.path.join(data_path, anno_file)

    print("Loading CSV files")

    ecg_csv = np.array(pd.read_csv(ecg_signals_path), dtype=np.float32)
    bp_csv = np.array(pd.read_csv(bp_signals_path), dtype=np.float32)
    eeg_csv = np.array(pd.read_csv(eeg_signals_path), dtype=np.float32)
    resp_csv = np.array(pd.read_csv(resp_signals_path), dtype=np.float32)
    anno_csv = pd.read_csv(anno_path)

    print("Loading CSV files Finished")

    idx_train, idx_test = [], []

    for i in range(anno_csv.shape[0]):
        if anno_csv.loc[i, "File"] in test_studies:
            idx_test.append(i)

        else:
            idx_train.append(i)

    ecg_train = ecg_csv[idx_train, :]
    ecg_test  = ecg_csv[idx_test, :]

    bp_train = bp_csv[idx_train, :]
    bp_test  = bp_csv[idx_test, :]

    eeg_train = eeg_csv[idx_train, :]
    eeg_test  = eeg_csv[idx_test, :]

    resp_train = resp_csv[idx_train, :]
    resp_test  = resp_csv[idx_test, :]

    train = MIT_PSG_Diffusion_Dataset(ecg_train, bp_train, eeg_train, resp_train)
    test = MIT_PSG_Diffusion_Dataset(ecg_test, bp_test, eeg_test, resp_test)

    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl

def train_batch(batch, model, loss_fn, optimizer):
    
    optimizer.zero_grad()
    model.train()

    x_co, x_t, noise, mask, diff_emb = batch

    noise_prediction = model(x_co, x_t, diff_emb, mask)
    noise_prediction = noise_prediction * mask.squeeze()

    batch_loss = loss_fn(noise_prediction, noise)
    batch_loss.backward()
    
    optimizer.step()
    
    return batch_loss.item()


def val_batch(batch, model, loss_fn):
    
    model.eval()

    x_co, x_t, noise, mask, diff_emb = batch

    noise_prediction = model(x_co, x_t, diff_emb, mask)
    noise_prediction = noise_prediction * mask.squeeze()

    batch_loss = loss_fn(noise_prediction, noise)
    return batch_loss.item()