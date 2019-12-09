from typing import Any
from PIL import Image
import torchvision as tv
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_ssim
from UNET import UNet
from layers.conv_layer import ConvLayer
import torch
import fastMRI.functions.transforms as T
from utils.data_loader import load_data_path, MRIDataset, collate_batches

import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load Data
print("Data loading...")
try:
    data_path_train = '/data/local/NC2019MRI/train'
    data_list = load_data_path(data_path_train, data_path_train)
except:
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

acc = 4
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 8

# create data loader for training set. It applies same to validation set as well
train_dataset = MRIDataset(data_list_train, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=10, num_workers=num_workers,
                          collate_fn=collate_batches)

val_dataset = MRIDataset(data_list_val, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=10, num_workers=num_workers, collate_fn=collate_batches)

data_loaders = {"train": train_loader, "val": val_loader}
data_lengths = {"train": len(train_idx), "val": val_len}
print("Data loaded")

EPSILON = 0.001

if __name__ == "__main__":
    print(device)
    model = UNet(1, 1, 32).to(device)
    print("Constructed model")
    criterion = nn.MSELoss()
    # criterion = pytorch_ssim.SSIM()
    # optimiser = optim.SGD(model.parameters(), lr=EPSILON)
    optimiser = optim.Adam(model.parameters(), lr=EPSILON)

    total_step = len(train_loader)
    n_epochs = 10
    batch_loss = list()
    acc_list = list()
    train_loss = []
    print("Starting training")
    fig = plt.figure()

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, sample in enumerate(data_loaders[phase]):
                # get the input data
                img_gt, img_und, rawdata_und, masks, norm = sample
                # print(img_gt.shape)
                img_in = img_und
                # img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).transpose_(0, 1)
                img_in = Variable(torch.FloatTensor(img_in)).cuda()
                # print(2)
                # ground_truth = T.center_crop(T.complex_abs(img_gt).unsqueeze(0), [320, 320]).transpose_(0, 1)
                ground_truth = img_gt
                ground_truth = Variable(torch.FloatTensor(ground_truth)).cuda()
                output = model(img_in)
                # print(3)
                final = output
                ground_truth = ground_truth
                # print(4)
                loss = []
                # print(final.shape)
                for b in range(list(final.size())[0]):
                    # print(final.shape)
                    # print(b)
                    # print(list(final.size())[0])
                    # print(ground_truth.shape)
                    loss.append(
                        criterion(final[b], ground_truth[b])
                    )

                # print(loss)
                loss = sum(loss)
                # print(loss)
                optimiser.zero_grad()
                # loss = - criterion(output, T.center_crop(T.complex_abs(img_gt).unsqueeze(0), [320, 320]).transpose_(0,1).to(device))

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

    torch.save(model.state_dict(), f"./models/UNET-B10-ssim")
    print("Minimum loss:", min(batch_loss))
    print("Maximum loss:", max(batch_loss))
    print("Average loss:", sum(batch_loss) / len(batch_loss))

    print("Epoch loss: ", train_loss)
    plt.plot(range(n_epochs), train_loss)
    plt.show()
