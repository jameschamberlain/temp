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


class UNet(nn.Module):

    def __init__(self, c_in: int, c_out: int, c: int) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c = c
        self.n_pool_layers = 4
        self.drop_prob = 0
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
data_path_train = '/home/sam/datasets/FastMRI/NC2019MRI/train'
data_path_val = '/home/sam/datasets/FastMRI/NC2019MRI/train'
data_list = load_data_path(data_path_train, data_path_val)

acc = 8
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 8

# create data loader for training set. It applies same to validation set as well
train_dataset = MRIDataset(data_list['train'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=num_workers, drop_last=True)
print("Data loaded")

EPSILON = 0.001
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print(device)
    model = UNet(1, 1, 32).to(device)
    print("Constructed model")
    # criterion = nn.MSELoss()
    criterion = pytorch_ssim.SSIM()
    # optimiser = optim.SGD(model.parameters(), lr=EPSILON)
    optimiser = optim.Adam(model.parameters(), lr=EPSILON)
    total_step = len(train_loader)
    n_epochs = 10
    loss_list = list()
    acc_list = list()
    loss_avg = list()
    loss_mins = list()
    loss_maxs = list()
    print("Starting training")
    fig = plt.figure()
    for epoch in range(n_epochs):
        loss_list = []
        for i, sample in enumerate(train_loader):

            img_gt, img_und, rawdata_und, masks, norm = sample
            # print(img_und.shape)
            # plt.subplot(1, 1, 1)
            # plt.imshow(T.complex_abs(img_und).squeeze().numpy(), cmap='gray')
            # plt.show()

            # img_in = F.interpolate(img_und, mode='bicubic', scale_factor=1).to(device)

            # img_in = img_und.transpose(-1, 1).unsqueeze(1).to(device)
            # print(img_in.shape)
            # img_in = T.root_sum_of_squares(img_in, 2)
            img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0).transpose(0, 1), [320, 320]).to(device)
            # print(img_in.shape)
            # img_in = img_in.squeeze(1)

            # print(f"img input shape: {img_in.shape}")

            output = model(img_in)
            optimiser.zero_grad()
            # img_gt = T.complex_center_crop(img_gt, (256, 256)).reshape(1, 1, 64, 320)
            # print(img_gt.shape)
            loss = - criterion(output,
                               T.center_crop(T.complex_abs(img_gt).unsqueeze(0).transpose(0, 1), [320, 320]).to(device))
            loss_list.append(- loss.item())
            loss.backward()
            optimiser.step()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, n_epochs, i + 1, total_step, - loss.item()))

        # torch.save(model.state_dict(), f"./models/UNET-{epoch}")
        min_loss = min(loss_list)
        print("Minimum loss:", min_loss)
        loss_mins.append(min_loss)
        max_loss = max(loss_list)
        print("Maximum loss:", max_loss)
        loss_maxs.append(max_loss)
        avg_loss = sum(loss_list) / len(loss_list)
        print("Average loss:", avg_loss)
        loss_avg.append(avg_loss)
        plt.plot(range(0, total_step), loss_list, 'bx')
        plt.show()
        plt.plot(range(0, epoch+1), loss_maxs, 'r-')
        plt.plot(range(0, epoch+1), loss_mins, 'b-')
        plt.plot(range(0, epoch+1), loss_avg, 'y--')
        plt.show()
    torch.save(model.state_dict(), f"./models/UNET-")
