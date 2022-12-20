import numpy as np
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm 

from data_utils import train_batch, val_batch, get_data_loaders
from model import SSSD

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

EPOCHS = 100
TEST_STUDIES = ['slp14', 'slp61']
FS = 100
L = 2.5


###########################################################################################################################################################
########################################################### MODEL INITIALIZATION ##########################################################################
###########################################################################################################################################################

s4_params = {
    "s4_lmax" : 100,
    "s4_d_state" : 64,
    "s4_dropout" : 0.0,
    "s4_bidirectional" : 1,
    "s4_layernorm": 1 }

model = SSSD(num_noise_steps = 200, 
            beta_start = 0.0001, 
            beta_end = 0.2,
            num_blocks = 8, 
            features = 4, 
            embed_dim_in = 128, 
            embed_dim_out = 512, 
            channels_dim = 256, 
            skip_channels_dim = 256, 
            s4_params = s4_params).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of Trainable Parameters of the model: " + str(count_parameters(model)))

###########################################################################################################################################################
######################################################## DATALOADER INITIALIZATION ########################################################################
###########################################################################################################################################################

data_path = "C:/Users/adria/Desktop/Repositories/Physionet Datasets/CSV Data"

ecg_file = "psg_ecg_signals_2.5_s.csv"
bp_file = "psg_bp_signals_2.5_s.csv"
eeg_file = "psg_eeg_signals_2.5_s.csv"
resp_file = "psg_resp_signals_2.5_s.csv"
anno_file = "psg_annotations_2.5_s.csv"


train_dl, test_dl, fixed_input = get_data_loaders(data_path, ecg_file, bp_file, eeg_file, resp_file, anno_file, TEST_STUDIES, fs=FS, l=L, batch_size=16)

###########################################################################################################################################################
############################################################ TRAINING VISUALIZATION #######################################################################
###########################################################################################################################################################


save_figures_path = "C:/Users/adria/Desktop/Repositories/DiffusionModels/SSSD/Figures"
labels = ["ECG", "BP", "EEG", "Resp"]
mask = np.zeros_like(fixed_input)

mask[0, int(FS * L * 0.75) :] = 1

###########################################################################################################################################################
############################################################ TRAINING PROCEDURE ###########################################################################
###########################################################################################################################################################

saved_model_pat = "C:/Users/adria\Desktop/Repositories/DiffusionModels/SSSD/Saved_Model/sssd_model_2.5_s.pth"
saved_last_model_pat = "C:/Users/adria\Desktop/Repositories/DiffusionModels/SSSD/Saved_Model/sssd_last_model_2.5_s.pth"
# weights_path = "C:/Users/adria/Desktop/Repositories/DiffusionModels/CSDI/Saved_Model/csdi_model.pth"
# model.load_state_dict(torch.load(weights_path))

current_epoch = 0

optimizer = Adam(model.parameters(), lr=0.0002)
best_loss = 0.9
best_epoch = -1

scaler = torch.cuda.amp.GradScaler()


writer = SummaryWriter(log_dir="logs/losses")

for epoch in range(current_epoch, EPOCHS):
    print("\n\nEpoch: ", str(epoch))
    epoch_loss, val_loss = [], []
    
    pbar = tqdm(train_dl)
    for batch in pbar:
        batch_loss = train_batch(batch, model, optimizer, scaler)
        epoch_loss.append(batch_loss)
        pbar.set_postfix(MSE=batch_loss)
        
    epoch_loss = np.array(epoch_loss).mean()
    print("Epoch Training Loss: " + str(epoch_loss))
    writer.add_scalar("Training Loss", epoch_loss, global_step=epoch)

    pbar = tqdm(test_dl)
    for batch in pbar:
        batch_val_loss = val_batch(batch, model)
        val_loss.append(batch_val_loss)
        pbar.set_postfix(MSE=batch_val_loss)

    val_loss = np.array(val_loss).mean()
    print("Epoch Validation Loss: " + str(val_loss))
    print("Best Epoch: " + str(best_epoch))
    writer.add_scalar("Validation Loss", val_loss, global_step=epoch)


    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.to("cpu").state_dict(), saved_model_pat)
        model.to(device)
        best_epoch = epoch
        print("Model Saved")

    torch.save(model.to("cpu").state_dict(), saved_last_model_pat)
    model.to(device)

    # Visualization    
    x_t, x_co = model.impute(fixed_input, mask)
    x_t = x_t.squeeze().numpy()
    x_co = x_co.squeeze().numpy()

    plt.figure(figsize=(18, 24))

    time = np.arange(x_t.shape[1]) / FS

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(time[x_co[i, :] != 0], x_co[i, x_co[i, :] != 0])
        plt.plot(time[x_t[i, :] != 0], x_t[i, x_t[i, :] != 0])
        plt.title(labels[i] + " Signal")    
        plt.xlabel("Time (s)")
        plt.ylim(-10, 10)
    
    figure_name = str(epoch) + ".jpg"
    figure_path = os.path.join(save_figures_path, figure_name)
    plt.savefig(figure_path)
    plt.close('all')