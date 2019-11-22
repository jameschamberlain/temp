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
        n_features = 0
        n_output = 1

        self.hidden0 = nn.Sequential(
            nn.Conv2d(640, 320, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(320, 640, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.hidden0(x)
        out = self.hidden1(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
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
        print(np.shape(img_und))
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
        i+=1