import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import pytorch_ssim
from UNET import UNet
from utils.data_loader import collate_batches, MRIDataset, load_data_path
import matplotlib.pyplot as plt
import sys

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
ENDC = '\033[0m'
info = lambda x: print(x)
warn = lambda x: print(YELLOW + x + ENDC)
success = lambda x: print(GREEN + x + ENDC)
error = lambda x: print(RED + x + ENDC)


def loading_bar(percentage):
    length = 30.0
    bar = "|"
    increment = 1 / length
    current = increment
    while percentage > current and current <= 1.0:
        bar += "#"
        current += increment
    while current <= 1.0:
        bar += " "
        current += increment
    bar += "|"
    sys.stdout.write("\r" + bar)


def advance_epoch(model, data_loader, optimizer):
    model.train()
    losses = []
    avg_loss = 0.

    print("TRAINING: ")
    print("")
    for iter, data in enumerate(data_loader):
        loading_bar(iter / len(data_loader))
        img_gt, img_und, rawdata_und, masks, norm = data

        img_in = Variable(torch.FloatTensor(img_und)).cuda()

        ground_truth = Variable(torch.FloatTensor(img_gt)).cuda()
        # print(img_in.shape)
        # print(ground_truth.shape)
        output = model(img_in)
        # print(output.shape)

        loss = 1 - pytorch_ssim.ssim(output, ground_truth)
        # loss = np.sum(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()

    return np.average(losses)


def evaluate(device, model, data_loader):
    model.eval()
    losses = []

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            img_gt, img_und, rawdata_und, masks, norm = data
            img_und = img_und.to(device)
            img_gt = img_gt.to(device)
            output = model(img_und)

            import fastMRI.functions.transforms as T

            # target = T.normalize() target
            # output = output
            # print(norm.shape)

            loss = 1 - pytorch_ssim.ssim(output, img_gt)
            losses.append(loss.item())
    return np.mean(losses)


CENTRE_FRACTION = 0.08
ACCELERATION = 4
EPSILON = 0.00001
GAMMA = 0.1
STEP_SIZE = 10
BATCH_SIZE = 14
NUMBER_EPOCHS = 30
NUMBER_POOL_LAYERS = 4
DROP_PROB = 0
CHANNEL_WIDTH = 32



def main():
    warn("Data loading...")
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

    data_list_train = data_list['train']
    data_list_val = data_list['val']

    seed = False  # random masks for each slice

    num_workers = 8
    # create data loader for training set. It applies same to validation set as well
    train_dataset = MRIDataset(data_list_train, acceleration=ACCELERATION, center_fraction=CENTRE_FRACTION,
                               use_seed=seed)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers,
                              collate_fn=collate_batches)

    val_dataset = MRIDataset(data_list_val, acceleration=ACCELERATION,
                             center_fraction=CENTRE_FRACTION, use_seed=seed)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers,
                            collate_fn=collate_batches)

    success("Data loaded")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    warn("Constructing model")
    model = UNet(1, 1, 32, NUMBER_POOL_LAYERS, DROP_PROB).to(device)
    success("Constructed model")

    criterion = pytorch_ssim.SSIM()

    optimiser = optim.Adam(model.parameters(), lr=EPSILON)
    # optimiser = torch.optim.RMSprop(model.parameters(), EPSILON, weight_decay=0)
    # total_step = len(train_loader)
    # batch_loss = list()
    # acc_list = list()
    # train_loss = []
    warn("Starting training")
    # fig = plt.figure()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=STEP_SIZE, gamma=GAMMA)
    val_losses = []
    train_losses = []
    for epoch in range(0, NUMBER_EPOCHS):
        success(f"EPOCH: {epoch}")
        error("-" * 10)
        train_loss = advance_epoch(model, train_loader, optimiser)
        train_losses.append(train_loss)
        # scheduler.step(epoch)
        dev_loss = evaluate(device, model, val_loader)
        val_losses.append(dev_loss)
        # visualize(args, epoch, model, display_loader, writer)
        info(
            f'Epoch = [{epoch:4d}/{NUMBER_EPOCHS:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {dev_loss:.4g}',
        )
    # torch.save(model.state_dict(),
    #            f"./models/UNET-B{BATCH_SIZE}e-{NUMBER_EPOCHS}-lr{EPSILON}-ssim-adam.pkl")
    plt.plot(range(NUMBER_EPOCHS), train_losses)
    plt.plot(range(NUMBER_EPOCHS), val_losses)
    plt.show()
    plt.savefig(CHANNEL_WIDTH +'.png')


if __name__ == "__main__":
    main()
