import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.data_loader import load_data_path, MRIDataset, show_slices
import matplotlib.pyplot as plt
from fastMRI.functions import transforms as T
from typing import List

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

# val_dataset = MRIDataset(data_list['val'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
# val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers)

EPSILON = 0.001


# Convolutional Net
class ConvNet(torch.nn.Module):

    def __init__(self) -> None:
        super(ConvNet, self).__init__()

        c = 640
        c_1 = c
        f_1 = 1
        n_1 = 1  # number of convs to apply
        l1_modules: List[nn.Module] = []
        l1_conv_layers = [nn.Conv2d(c, c_1, kernel_size=f_1) for _ in range(n_1)]

        l1_modules.append(*l1_conv_layers)
        l1_modules.append(nn.ReLU())

        self.hidden1 = nn.Sequential(*l1_modules)

        n_2 = 1
        c_2 = c_1
        f_2 = 1
        l2_modules = []
        l2_conv_layers = [nn.Conv2d(c_1, c_2, kernel_size=f_2) for _ in range(n_2)]
        l2_modules.append(*l2_conv_layers)
        l2_modules.append(nn.ReLU())

        self.hidden2 = nn.Sequential(*l2_modules)

        l3_modules = []
        c_3 = c
        n_3 = 1
        f_3 = 1
        l3_conv_layers = [nn.Conv2d(c_2, c_3, kernel_size=f_3) for _ in range(n_3)]
        l3_modules.append(*l3_conv_layers)

        self.hidden3 = nn.Sequential(*l3_modules)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.hidden2(out)
        out = self.hidden3(out)
        # out = out.reshape(out.size(0), -1)
        return out


model = ConvNet()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=EPSILON)
total_step = len(train_loader)
n_epochs = 5
loss_list = list()
acc_list = list()
i = 0
for epoch in range(n_epochs):
    for iteration, sample in enumerate(train_loader):
        img_gt, img_und, rawdata_und, masks, norm = sample
        output = model(img_und)
        loss = criterion(output, img_gt)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = img_gt.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == img_gt).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
        i += 1
