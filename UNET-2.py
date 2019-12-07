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
import ray


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

ray.init()

# The number of sets of random hyperparameters to try.
num_evaluations = 3 


# A function for generating random hyperparameters.
def generate_hyperparameters():
    return {
        "learning_rate": 10 ** np.random.uniform(-5, 1),
        "batch_size": np.random.randint(1, 100),
    }


acc = hyper_param.N_FOLD
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 4


def get_data_loaders(batch_size):
    print("Data loading...")
    data_path_train = '/data/local/NC2019MRI/train'
    data_list = load_data_path(data_path_train, data_path_train)

    # Split dataset into train-validate
    validation_split = 0.1
    dataset_len = len(data_list['train'])
    indices = list(range(dataset_len))

    # Randomly splitting indices
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    data_list_train = [data_list['train'][i] for i in train_idx]
    data_list_val = [data_list['val'][i] for i in validation_idx]

    # Create data loader for training set. It applies same to validation set as well
    train_dataset = MRIDataset(data_list_train, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    val_dataset = MRIDataset(data_list_val, acceleration=acc, center_fraction=cen_fract, use_seed=seed)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    print("Data loaded")

    return train_loader, val_loader


def train(model, optimiser, criterion, train_loader):
    print("Starting training")
    model.train()

    for i, sample in enumerate(train_loader):
        img_gt, img_und, _, _, _ = sample
        img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0).transpose_(0,1), [320, 320])

        output = model(img_in)
        optimiser.zero_grad()

        loss = - criterion(output, T.center_crop(T.complex_abs(img_gt).unsqueeze(0).transpose_(0,1), [320, 320]))
        loss.backward()
        optimiser.step()


def test(model, test_loader):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            img_gt, img_und, _, _, _ = sample
            img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320])

            output = model(img_in)

            real = T.center_crop(T.complex_abs(img_gt).unsqueeze(0), [320, 320])

            loss = - pytorch_ssim.ssim(output, real)
            total_loss.append(- loss.item())

            # TODO evaluate model
            #_, predicted = torch.max(outputs.data, 1)
            #total += target.size(0)
            #correct += (predicted == target).sum().item()
    
    return sum(total_loss)/len(total_loss) 



@ray.remote
def evaluate_hyperparameters(config):
    print("Constructed model")
    model = UNet(1, 1, hyper_param.CHANNELS)
    train_loader, val_loader = get_data_loaders(config["batch_size"])

    criterion = pytorch_ssim.SSIM()
    
    # criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])
    # optimizer = optim.SGD(
    #    model.parameters(),
    #    lr=config["learning_rate"],
    #    momentum=config["momentum"])
    train(model, optimiser, criterion, train_loader)
    return test(model, val_loader)


if __name__ == "__main__":
    # print(device)

    # Keep track of the best hyperparameters and the best accuracy.
    best_hyperparameters = None
    best_accuracy = 0
    # A list holding the object IDs for all of the experiments that we have
    # launched but have not yet been processed.
    remaining_ids = []
    # A dictionary mapping an experiment's object ID to its hyperparameters.
    # hyerparameters used for that experiment.
    hyperparameters_mapping = {}

    # Randomly generate sets of hyperparameters and launch a task to evaluate it.
    for i in range(num_evaluations):
        hyperparameters = generate_hyperparameters()
        accuracy_id = evaluate_hyperparameters.remote(hyperparameters)
        remaining_ids.append(accuracy_id)
        hyperparameters_mapping[accuracy_id] = hyperparameters

    # Fetch and print the results of the tasks in the order that they complete.
    while remaining_ids:
        # Use ray.wait to get the object ID of the first task that completes.
        done_ids, remaining_ids = ray.wait(remaining_ids)
        # There is only one return result by default.
        result_id = done_ids[0]

        hyperparameters = hyperparameters_mapping[result_id]
        accuracy = ray.get(result_id)
        print("""We achieve accuracy {:.3}% with
            learning_rate: {:.2}
            batch_size: {}
            momentum: {:.2}
        """.format(100 * accuracy, hyperparameters["learning_rate"],
                   hyperparameters["batch_size"], hyperparameters["momentum"]))
        if accuracy > best_accuracy:
            best_hyperparameters = hyperparameters
            best_accuracy = accuracy

    # Record the best performing set of hyperparameters.
    print("""Best accuracy over {} trials was {:.3} with
        learning_rate: {:.2}
        batch_size: {}
        momentum: {:.2}
        """.format(num_evaluations, 100 * best_accuracy,
                   best_hyperparameters["learning_rate"],
                   best_hyperparameters["batch_size"],
                   best_hyperparameters["momentum"]))
