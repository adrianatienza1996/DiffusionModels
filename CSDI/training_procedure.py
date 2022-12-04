import numpy as np
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm 

from data_utils import train_batch, val_batch, get_data_loaders, mse_loss
from model import CSDI

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

EPOCHS = 200
TEST_STUDIES = ['slp14', 'slp61']


###########################################################################################################################################################
########################################################### MODEL INITIALIZATION ##########################################################################
###########################################################################################################################################################

model = CSDI(temp_strips_blocks=2,
             feat_strips_lenght=10,
             l = 10,
             fs=75,
             num_features=4,
             num_res_blocks=4,
             number_heads=8,
             model_dim=64,
             emb_dim=128,
             time_dim=128,
             feat_dim=16,
             do_prob=0.1).to(device)


###########################################################################################################################################################
######################################################## DATALOADER INITIALIZATION ########################################################################
###########################################################################################################################################################

data_path = "C:/Users/adria/Desktop/Repositories/Physionet Datasets/CSV Data"

ecg_file = "psg_ecg_signals_resampled_normalized.csv"
bp_file = "psg_bp_signals_resampled_normalized.csv"
eeg_file = "psg_eeg_signals_resampled_normalized.csv"
resp_file = "psg_resp_signals_resampled_normalized.csv"
anno_file = "psg_annotations.csv"

train_dl, test_dl = get_data_loaders(data_path, ecg_file, bp_file, eeg_file, resp_file, anno_file, TEST_STUDIES, batch_size=12)

###########################################################################################################################################################
############################################################ TRAINING PROCEDURE ###########################################################################
###########################################################################################################################################################

saved_model_pat = "C:/Users/adria\Desktop/Repositories/DiffusionModels/CSDI/Saved_Model/csdi_model.pth"
weights_path = "C:/Users/adria/Desktop/Repositories/DiffusionModels/CSDI/Saved_Model/csdi_model.pth"
model.load_state_dict(torch.load(weights_path))

current_epoch = 10
loss_fn = mse_loss()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
best_loss = 9999999
best_epoch = -1

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[int(0.75 * (EPOCHS - current_epoch)), int(0.9 * (EPOCHS - current_epoch))], 
                                                    gamma=0.1)

scaler = torch.cuda.amp.GradScaler()


writer = SummaryWriter(log_dir="logs/losses")

for epoch in range(current_epoch, EPOCHS):
    print("\n\nEpoch: ", str(epoch))
    epoch_loss, val_loss = [], []
    
    for batch in tqdm(iter(train_dl)):
        batch_loss = train_batch(batch, model, loss_fn, optimizer, scaler)
        epoch_loss.append(batch_loss)

    epoch_loss = np.array(epoch_loss).mean()
    print("Epoch Training Loss: " + str(epoch_loss))
    writer.add_scalar("Training Loss", epoch_loss, global_step=epoch)

    for batch in tqdm(iter(test_dl)):
        batch_val_loss = val_batch(batch, model, loss_fn)
        val_loss.append(batch_val_loss)

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

    lr_scheduler.step()