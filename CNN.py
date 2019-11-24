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

        c = 0
        c_out = c
        f_1 = 0
        n_1 = 1  # number of convs to apply

        conv_layers = [nn.Conv2d(c, c_out, kernel_size=f_1, stride=1, padding=2) for i in range(n_1)]

        l1_modules = conv_layers.append(nn.ReLU)

        self.hidden1 = nn.Sequential(*l1_modules)

        self.hidden1 = nn.Sequential(
            nn.Conv2d(640, 4 * 640, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.hidden1(x)
        out = out.reshape(out.size(0), -1)
        return out


model = ConvNet()

criterion = nn.CrossEntropyLoss()
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
