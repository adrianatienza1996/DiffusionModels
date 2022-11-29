import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm 

from data_utils import MIT_PSG_Diffusion_Dataset, train_batch
from model import CSDI

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

EPOCHS = 200


###########################################################################################################################################################
########################################################### MODEL INITIALIZATION ##########################################################################
###########################################################################################################################################################

model = CSDI(strips_lenght=5,
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

train = MIT_PSG_Diffusion_Dataset(data_path=data_path,
                                  ecg_file=ecg_file,
                                  bp_file=bp_file,
                                  eeg_file=eeg_file,
                                  resp_file=resp_file)

train_dl = DataLoader(train, batch_size=16, shuffle=True)


###########################################################################################################################################################
############################################################ TRAINING PROCEDURE ###########################################################################
###########################################################################################################################################################

loss_fn = torch.nn.MSELoss()
loss = []
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
best_loss = 999999

writer = SummaryWriter(log_dir="logs/training_loss")
for step, epoch in enumerate(range(EPOCHS)):
    print("Epoch: ", str(epoch))
    epoch_loss = []
    for batch in tqdm(iter(train_dl)):

        batch_loss = train_batch(batch, model, loss_fn, optimizer)
        epoch_loss.append(batch_loss)

    epoch_loss = np.array(epoch_loss).mean()
    print("Epoch Loss: " + str(epoch_loss))
    loss.append(epoch_loss)
    writer.add_scalar("Training Loss", epoch_loss, global_step=step)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.to("cpu").state_dict(), "Saved_Model/csdi_model.pth")
        model.to(device)
        print("Model Saved")