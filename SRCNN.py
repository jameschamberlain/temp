import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.data_loader import load_data_path, MRIDataset, show_slices
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastMRI.functions import transforms as T
from typing import List
from PIL import Image

print(torch.cuda.get_device_properties(0))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load Data
data_path_train = '/home/sam/datasets/FastMRI/NC2019MRI/train'
data_path_val = '/home/sam/datasets/FastMRI/NC2019MRI/train'
data_list = load_data_path(data_path_train, data_path_val)

acc = 8
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 12

# create data loader for training set. It applies same to validation set as well
train_dataset = MRIDataset(data_list['train'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=num_workers)

val_dataset = MRIDataset(data_list['val'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers)

EPSILON = 0.01


# Convolutional Net
class ConvNet(torch.nn.Module):

    def __init__(self) -> None:
        super(ConvNet, self).__init__()

        c_0 = 1  # input channels
        f_1 = 6  # spacial size of kernel
        c_1 = 1  # output channels
        n_1 = 64  # number of convs to apply
        l1_modules: List[nn.Module] = []
        l1_conv_layers = [nn.Conv2d(in_channels=c_0, out_channels=c_1, kernel_size=f_1, padding=1, stride=1) for _ in
                          range(n_1)]
        l1_modules.extend(l1_conv_layers)
        l1_modules.append(nn.ReLU())

        self.hidden1 = nn.Sequential(*l1_modules)

        n_2 = 1  # number of convs to apply
        c_2 = 32  # output channels
        f_2 = 1  # size of kernel
        l2_modules = []
        l2_conv_layers = [nn.Conv2d(in_channels=c_1, out_channels=c_2, kernel_size=f_2, padding=1, stride=1) for _ in
                          range(n_2)]
        l2_modules.extend(l2_conv_layers)
        l2_modules.append(nn.ReLU())

        self.hidden2 = nn.Sequential(*l2_modules)

        l3_modules = []
        c_3 = c_0  # output channels = original input
        n_3 = 1  # number of convolutions to apply
        f_3 = 5  # size of kernel
        l3_conv_layers = [nn.Conv2d(in_channels=c_2, out_channels=c_3, kernel_size=f_3, padding=1, stride=1) for _ in
                          range(n_3)]
        l3_modules.extend(l3_conv_layers)

        self.hidden3 = nn.Sequential(*l3_modules)

    def forward(self, x):
        out = self.hidden1(x)
        print(f"l1: {out.shape}")
        out = self.hidden2(out)
        print(f"l2: {out.shape}")
        out = self.hidden3(out)
        print(f"l3: {out.shape}")
        return out


RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
from sys import stdout
import pytorch_ssim

printc = lambda x: stdout.write(x)
if __name__ == "__main__":
    printc(YELLOW)
    printc(device)
    model = ConvNet().to(device)
    printc(GREEN)
    printc("constructed model\n")
    criterion = nn.MSELoss()
    # criterion = pytorch_ssim.SSIM()
    optimiser = optim.SGD(model.parameters(), lr=EPSILON)
    total_step = len(train_loader)
    n_epochs = 5
    loss_list = list()
    acc_list = list()
    print("Starting training")
    for epoch in range(n_epochs):

        for i, sample in enumerate(train_loader):
            img_gt, img_und, rawdata_und, masks, norm = sample
            printc(YELLOW)

            img_in = F.interpolate(img_und, mode='bicubic', scale_factor=1).to(device)

            print(img_in.shape)
            img_in = T.complex_center_crop(img_in, (256, 256))
            img_in = img_in.reshape(1, 1, 256, 512).to(device)
            print(f"img input shape: {img_in.shape}")

            output = model(img_in)
            optimiser.zero_grad()
            img_gt = T.complex_center_crop(img_gt, (256, 256)).reshape(1, 1, 64, 320)
            print(img_gt.shape)
            loss = - criterion(output, img_gt.to(device))
            loss_list.append(- loss.item())
            loss.backward()
            optimiser.step()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, n_epochs, i + 1, total_step, - loss.item()))

        torch.save(model.state_dict(), f"./models/CNN-{epoch}")
