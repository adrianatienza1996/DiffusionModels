import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class MIT_PSG_Diffusion_Dataset(Dataset):
    
    def __init__(self, ecg_signals, bp_signals, eeg_signals, resp_signals, fs=100, l=2.5, noise_steps=50, b_low=0.0001, b_high=0.5, diff_dim=128, device="cuda"):

        self.ecg_csv = np.array(ecg_signals, dtype=np.float32)
        self.bp_csv = np.array(bp_signals, dtype=np.float32)
        self.eeg_csv = np.array(eeg_signals, dtype=np.float32)
        self.resp_csv = np.array(resp_signals, dtype=np.float32)
        
    def __len__(self):
        return len(self.ecg_csv)

    def __getitem__(self, ix):
        tmp_ecg = self.ecg_csv[ix, :].reshape(1, -1)
        tmp_bp = self.bp_csv[ix, :].reshape(1, -1)
        tmp_eeg = self.eeg_csv[ix, :].reshape(1, -1)
        tmp_resp = self.resp_csv[ix, :].reshape(1, -1)

        tmp_x = np.concatenate([tmp_ecg, tmp_bp, tmp_eeg, tmp_resp], axis = 0)

        tmp_x = torch.tensor(tmp_x).float().to(device)
        return tmp_x


def get_data_loaders(data_path, ecg_file, bp_file, eeg_file, resp_file, anno_file, test_studies, fs, l, batch_size=16):

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

    train = MIT_PSG_Diffusion_Dataset(ecg_train, bp_train, eeg_train, resp_train, fs=fs, l=l)
    test = MIT_PSG_Diffusion_Dataset(ecg_test, bp_test, eeg_test, resp_test, fs=fs, l=l)

    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)


    # For debugging
    idx = 187
    tmp_ecg = ecg_test[idx, :].reshape(1, -1)
    tmp_bp = bp_test[idx, :].reshape(1, -1)
    tmp_eeg = eeg_test[idx, :].reshape(1, -1)
    tmp_resp = resp_test[idx, :].reshape(1, -1)

    tmp_x = np.concatenate([tmp_ecg, tmp_bp, tmp_eeg, tmp_resp], axis = 0)

    return train_dl, test_dl, tmp_x


def create_mask(tmp_x):
    mask = torch.zeros_like(tmp_x)
    
    for i in range(len(tmp_x)):
        tmp = np.random.randint(1, 5)
        tmp = np.random.choice(4, tmp, replace=False)
        
        for tmp_idx in tmp:
            first_index = np.random.randint(0, int(np.round(0.90 * 100 * 2.5)))
            tmp_max = min(int(100 * 2.5), first_index + int(np.round(0.4 * 100 * 2.5)))
            second_index = np.random.randint(first_index + int(np.round(0.1 * 100 * 2.5)), tmp_max)
            mask[i, tmp_idx, first_index : second_index] = 1

    return mask


def noisy_imput(tmp_x, model):
    B, K, L = tmp_x.shape 
    tmp_mask = create_mask(tmp_x)

    x_t = tmp_x * tmp_mask
    x_co = tmp_x * (1 - tmp_mask)

    t = torch.randint(0, model.noise_steps, [B]).to(device)

    current_alpha_hat = model.alpha_hat_torch[t]  
    noise = torch.randn_like(x_t) * tmp_mask

    x_t = (current_alpha_hat ** 0.5) * x_t + (1.0 - current_alpha_hat) ** 0.5 * noise
    
    x_t = x_t.unsqueeze(1)
    x_co = x_co.unsqueeze(1)

    x = torch.cat([x_co, x_t], dim=1).to(device)
    tmp_mask = tmp_mask.to(device)
    t = t.long()
    return x, noise, tmp_mask, t

    
def train_batch(batch, model, optimizer, scaler):
    
    model.train()
    tmp_x = batch
    x, noise, mask, t = noisy_imput(tmp_x, model)

    mask_co = 1 - mask
    mask_co = mask_co.unsqueeze(-1).to(device)

    noise_prediction = model(x, t, mask_co)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        noise_prediction = noise_prediction.squeeze() * mask.squeeze()
        batch_loss = mse_loss(noise_prediction, noise, mask)
    
    optimizer.zero_grad()
    scaler.scale(batch_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return batch_loss.item()


def val_batch(batch, model):
    
    model.eval()
    tmp_x = batch
    x, noise, mask, t = noisy_imput(tmp_x, model)

    mask_co = 1 - mask
    mask_co = mask_co.unsqueeze(-1).to(device)

    noise_prediction = model(x, t, mask_co)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        noise_prediction = noise_prediction.squeeze() * mask.squeeze()
        batch_loss = mse_loss(noise_prediction, noise, mask)

    return batch_loss.item()


def mse_loss(noise, predicted, target_mask):
    residual = (noise - predicted) * target_mask
    num_eval = target_mask.sum()
    loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
    return loss


class EMA:
    def __init__(self, beta, start_ema = 2000):
        self.beta = beta
        self.step = 0
        self.start_ema = start_ema

    def update_weights(self, new_model, old_model):
        for new_params, old_params in zip(new_model.parameters(), old_model.parameters()):
            new_w, old_w = new_params.data, old_params.data
            new_params.data = new_w * (1 - self.beta) +  old_w * self.beta

    def copy_params(self, new_model, old_model):
        new_model.load_state_dict(old_model.state_dict())

    def ema_step(self, new_model, old_model):
        if self.beta < self.start_ema:
            self.copy_params(new_model, old_model)
            self.step += 1
        else:
            self.update_weights(new_model, old_model)
            self.step +=1