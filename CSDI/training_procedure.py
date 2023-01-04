import numpy as np
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm 

from data_utils import train_batch, val_batch, get_data_loaders, EMA
from model import CSDI

import matplotlib.pyplot as plt
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

EPOCHS = 100
TEST_STUDIES = ['slp14', 'slp61']
FS = 100
L = 2.5

# Use Exponential Moving Average Trick for Updating Weights during Training Loop
use_EMA = True


###########################################################################################################################################################
########################################################### MODEL INITIALIZATION ##########################################################################
###########################################################################################################################################################

model = CSDI(noise_steps=50, 
             l = L, 
             fs = FS, 
             beta_start=0.0001,
             beta_end=0.5,
             temp_strips_blocks = 1, 
             feat_strips_lenght = 1, 
             num_features = 4, 
             num_res_blocks = 4, 
             number_heads = 8, 
             model_dim = 64,
             emb_dim = 128, 
             time_dim = 128, 
             feat_dim = 16, 
             do_prob = 0.1).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of Trainable Parameters of the model: " + str(count_parameters(model)))

if use_EMA:
    ema = EMA(beta = 0.95)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

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


save_figures_path = "C:/Users/adria/Desktop/Repositories/DiffusionModels/CSDI/Figures/EMA"
labels = ["ECG", "BP", "EEG", "Resp"]
mask = np.zeros_like(fixed_input)

mask[0, int(FS * L * 0.75) :] = 1

###########################################################################################################################################################
############################################################ TRAINING PROCEDURE ###########################################################################
###########################################################################################################################################################

saved_model_pat = "C:/Users/adria\Desktop/Repositories/DiffusionModels/CSDI/Saved_Model/csdi_model_ema.pth"
saved_last_model_pat = "C:/Users/adria\Desktop/Repositories/DiffusionModels/CSDI/Saved_Model/csdi_last_model_ema.pth"

# weights_path = saved_last_model_pat
# model.load_state_dict(torch.load(weights_path))

current_epoch = 0

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
best_loss = 999
best_epoch = 0

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[int(0.75 * (EPOCHS - current_epoch)), int(0.9 * (EPOCHS - current_epoch))], 
                                                    gamma=0.1)

scaler = torch.cuda.amp.GradScaler()


writer = SummaryWriter(log_dir="logs/losses")

for epoch in range(current_epoch, EPOCHS):
    print("\n\nEpoch: ", str(epoch))
    epoch_loss, val_loss = [], []
    
    for batch in tqdm(iter(train_dl)):
        batch_loss = train_batch(batch, model, optimizer, scaler)
        if use_EMA:
            ema.ema_step(new_model=model, old_model=ema_model)

        epoch_loss.append(batch_loss)

    epoch_loss = np.array(epoch_loss).mean()
    print("Epoch Training Loss: " + str(epoch_loss))
    writer.add_scalar("Training Loss", epoch_loss, global_step=epoch)

    for batch in tqdm(iter(test_dl)):
        if use_EMA:
            batch_val_loss = val_batch(batch, ema_model)
            val_loss.append(batch_val_loss)

        else:
            batch_val_loss = val_batch(batch, model)
            val_loss.append(batch_val_loss)

    val_loss = np.array(val_loss).mean()
    print("Epoch Validation Loss: " + str(val_loss))
    print("Best Epoch: " + str(best_epoch))
    writer.add_scalar("Validation Loss", val_loss, global_step=epoch)


    if val_loss < best_loss:
        best_loss = val_loss
        if use_EMA:
            torch.save(ema_model.to("cpu").state_dict(), saved_model_pat)
            ema_model.to(device)

        else:
            torch.save(model.to("cpu").state_dict(), saved_model_pat)
            model.to(device)

        best_epoch = epoch
        print("Model Saved")

    if use_EMA:
        torch.save(ema_model.to("cpu").state_dict(), saved_last_model_pat)
        ema_model.to(device)

    else:
        torch.save(model.to("cpu").state_dict(), saved_last_model_pat)
        model.to(device)
    
    lr_scheduler.step()

    # Visualization    
    if use_EMA:
        x_t, x_co = ema_model.impute(fixed_input, mask)
    else:    
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