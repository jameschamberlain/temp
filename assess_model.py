import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.data_loader import MRIDataset, load_data_path, show_slices, collate_batches, load_data_path_test
import fastMRI.functions.transforms as T
import UNET
import pytorch_ssim
import numpy as np
import h5py
import os

data_path_train = '/home/sam/datasets/FastMRI/NC2019MRI/test'
# data_path_train = '/data/local/NC2019MRI/train'

data_path_val = data_path_train
# data_list = load_data_path(data_path_train, data_path_val)

frac = {4: 0.08, 8: 0.04}
acc = 4
data_list = load_data_path_test(data_path_train, acc)
cen_fract = frac[acc]
seed = False  # random masks for each slice
num_workers = 8

val_dataset = MRIDataset(data_list['test'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
# val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers, collate_fn=collate_batches)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1,
                        num_workers=num_workers)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
optimiser = 'SSIML1'
model = UNET.UNet(1, 1, 48, 4, 0).to(device)
# model.load_state_dict(torch.load(f"./vary-loss/models/UNET-lr0.0001-{optimiser}.pkl"))
model = (torch.load(
    f"./best_model_{acc}x.pkl"))
model.eval()
fig = plt.figure()
counter = 0
ssims = []

data = {f'recon_{acc}af': []}
file = h5py.File(f'recon_{acc}.h5', 'w')

N = len(val_loader)
data = file.create_dataset(name=f'recon_{acc}af', shape=(N, 320, 320))
for i, sample in enumerate(val_loader):
    img_gt, img_und, rawdata_und, masks, norm = sample
    img_in = img_und.to(device)
    # img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).transpose(0,1).to(device)

    # input
    A = T.center_crop(T.complex_abs(img_und), (320, 320)).unsqueeze(0).to(device)
    # print(A.shape)
    # C = T.center_crop(T.complex_abs(img_gt), (320, 320)
    #                   ).unsqueeze(0).to(device)
    # # output
    output = model(A).cpu().detach().squeeze()

    # print(output.shape)
    # B = output.squeeze().cpu()
    # print(B.shape)
    # ssim = pytorch_ssim.ssim(output,C)

    # ssims.append(ssim.item())
    # print(f"SSIM of this image is {ssim}")
    # real
    # C = img_gt.squeeze()
    # print(C.shape)
    # all_imgs = torch.stack([A.detach(), B.detach(), C.detach()], dim=0)

    # from left to right: mask, masked kspace, undersampled image, ground truth
    # if ssim > 0.8 and counter <=3:
    #     show_slices(all_imgs, [0, 1, 2], cmap='gray')
    #     plt.savefig(f"./recon/final_4x{ssim}.png")
    #     plt.pause(1)
    #     counter += 1
    # print(output.shape)
    # data[f'recon_{acc}af'].append(output)
    data[i] = output

file.close()
# if counter >= 3: break  # show 4 random slices
# print(f"Max ssim : {max(ssims)}")
# print(f"Average ssim is : {np.average(ssims)}")
# print(f"Variance in ssim is {np.var(ssims)}")
# print(f"Standard Deviation in ssim is {np.std(ssims)}")
