from typing import Any
from PIL import Image
import torchvision as tv
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_ssim
from layers.conv_layer import ConvLayer
import torch
import fastMRI.functions.transforms as T
from utils.data_loader import load_data_path, MRIDataset

import numpy as np
import matplotlib.pyplot as plt

import hyper_param


class UNet(nn.Module):

    def __init__(self, c_in: int, c_out: int, c: int) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c = c
        self.n_pool_layers = hyper_param.POOL_LAYERS
        self.drop_prob = hyper_param.DROPOUT
        self.down_sample_layers = nn.ModuleList([ConvLayer(self.c_in, self.c, self.drop_prob)])
        channels = self.c
        for _ in range(self.n_pool_layers - 1):
            self.down_sample_layers += [ConvLayer(channels, channels * 2, self.drop_prob)]
            channels *= 2

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(self.drop_prob)
        )

        self.up_sample_layers = nn.ModuleList()
        for _ in range(self.n_pool_layers - 1):
            self.up_sample_layers += [ConvLayer(channels * 2, channels // 2, self.drop_prob)]
            channels //= 2
        self.up_sample_layers += [ConvLayer(channels * 2, channels, self.drop_prob)]

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, ),
            nn.Conv2d(channels // 2, self.c_out, kernel_size=1),
            nn.Conv2d(self.c_out, self.c_out, kernel_size=1)
        )

    def forward(self, x: Any, ):
        stack = []
        y = x
        for layer in self.down_sample_layers:
            y = layer(y)
            stack.append(y)
            y = F.max_pool2d(y, kernel_size=2)

        y = self.conv(y)
        for layer in self.up_sample_layers:
            y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
            y = torch.cat([y, stack.pop()], dim=1)
            y = layer(y)

        return self.conv2(y)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load Data
print("Data loading...")
data_path_train = '/data/local/NC2019MRI/train'
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


acc = hyper_param.N_FOLD
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 8

# create data loader for training set. It applies same to validation set as well
train_dataset = MRIDataset(data_list_train, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hyper_param.BATCH_SIZE, num_workers=num_workers)

val_dataset = MRIDataset(data_list_val, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers)

data_loaders = {"train": train_loader, "val": val_loader}
data_lengths = {"train": len(train_idx), "val": val_len}
print("Data loaded")


if __name__ == "__main__":
    print(device)
    model = UNet(1, 1, hyper_param.CHANNELS).to(device)
    print("Constructed model")
    # criterion = nn.MSELoss()
    criterion = pytorch_ssim.SSIM()
    #optimiser = optim.SGD(model.parameters(), lr=EPSILON)
    optimiser = optim.Adam(model.parameters(), lr=hyper_param.EPSILON)

    total_step = len(train_loader)
    n_epochs = hyper_param.N_EPOCHS
    batch_loss = []
    train_loss = []
    val_loss = []

    print("Starting training")
    model.train()
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        if(epoch == 40):
            # Reduce learning rate
            for g in optimiser.param_groups:
                g['lr'] = hyper_param.EPSILON * 0.1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, sample in enumerate(data_loaders[phase]):
                img_gt, img_und, rawdata_und, masks, norm = sample
                img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).to(device)

                output = model(img_in)
                optimiser.zero_grad()

                loss = - criterion(output, T.center_crop(T.complex_abs(img_gt).unsqueeze(0), [320, 320]).to(device))
                
                # backward + optimise only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimiser.step()

                # calculate loss
                batch_loss.append(- loss.item())
                running_loss += - loss.item() * img_in.size(0) 

            epoch_loss = running_loss / data_lengths[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
    
    # Print stats
    print("Minimum loss:", min(batch_loss))
    print("Maximum loss:", max(batch_loss))
    print("Average loss:", sum(batch_loss) / len(batch_loss))

    print("Train loss: ", train_loss)
    print("Val loss: ", val_loss)
    fig = plt.figure()
    plt.plot(range(n_epochs), train_loss)
    plt.plot(range(n_epochs), val_loss)
    plt.savefig(f'diagrams/UNET_2-{hyper_param.DESCR}.png')

    # save model
    torch.save(model.state_dict(), f"./models/UNET_2-{hyper_param.DESCR}")