from torch.utils.data import DataLoader

import pytorch_ssim
from utils.data_loader import load_data_path, MRIDataset
import numpy as np
import torch.nn as nn
import fastMRI.functions.transforms as T

print("Data loading...")
# data_path_train = '/data/local/NC2019MRI/train'
data_path_train = '/home/sam/datasets/FastMRI/NC2019MRI/train'
data_list = load_data_path(data_path_train, data_path_train)

# Split dataset into train-validate
validation_split = 0.1
dataset_len = len(data_list['train'])
indices = list(range(dataset_len))

# Randomly splitting indices:
val_len = int(np.floor(validation_split * dataset_len))
validation_idx = np.random.choice(indices, size=val_len, replace=False)
train_idx = list(set(indices) - set(validation_idx))

data_list_train = [data_list['train'][i] for i in train_idx]
data_list_val = [data_list['val'][i] for i in validation_idx]

criterion = pytorch_ssim.SSIM()

acc = 8
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 8
val_loss = []
# create data loader for training set. It applies same to validation set as well
train_dataset = MRIDataset(data_list_train, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=10, num_workers=num_workers)

val_dataset = MRIDataset(data_list_val, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=10, num_workers=num_workers)

data_loaders = {"train": train_loader, "val": val_loader}
data_lengths = {"train": len(train_idx), "val": val_len}
train_loss = []
print("loaded data")
for phase in ['train', 'val']:

    for i, sample in enumerate(data_loaders[phase]):
        img_gt, img_und, rawdata_und, masks, norm = sample
        img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).transpose_(0, 1)

        loss = - criterion(img_in,
                           T.center_crop(T.complex_abs(img_gt).unsqueeze(0), [320, 320]).transpose_(0, 1))

        # backward + optimise only if in training phase

        # calculate loss
        # tra.append(- loss.item())
        # running_loss += - loss.item() * img_in.size(0)

    # epoch_loss = running_loss / data_lengths[phase]
    # print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    if phase == 'train':
        train_loss.append(-loss.item())
    else:
        val_loss.append(-loss.item())

print(train_loss)

print("\n\n")

print(val_loss)
