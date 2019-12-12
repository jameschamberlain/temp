import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.data_loader import MRIDataset, load_data_path, show_slices, collate_batches
import fastMRI.functions.transforms as T
import UNET
import pytorch_ssim
import numpy as np

data_path_train = '/home/sam/datasets/FastMRI/NC2019MRI/train'
# data_path_train = '/data/local/NC2019MRI/train'

data_path_val = data_path_train
data_list = load_data_path(data_path_train, data_path_val)

acc = 4
cen_fract = 0.08
seed = False  # random masks for each slice
num_workers = 8

val_dataset = MRIDataset(data_list['val'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers, collate_fn=collate_batches)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
optimiser = 'SGD'
model = UNET.UNet(1,1,32,4,0).to(device)
model.load_state_dict(torch.load(f"./vary-optim/models/UNET-lr0.0001-{optimiser}.pkl"))
model.eval()
fig = plt.figure()
counter= 0
ssims = []
for i, sample in enumerate(val_loader):
    img_gt, img_und, rawdata_und, masks, norm = sample
    img_in = img_und.to(device)
    # img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).transpose(0,1).to(device)

    # input
    A = img_und.squeeze()
    # print(A.shape)
    
    # output
    output = model(img_in)
    # print(output.shape)
    B = output.squeeze().cpu()
    # print(B.shape)
    ssim = pytorch_ssim.ssim(output,img_gt.to(device))
    
    ssims.append(ssim.item())
    # print(f"SSIM of this image is {ssim}")
    # real
    C = img_gt.squeeze()
    # print(C.shape)
    all_imgs = torch.stack([A.detach(), B.detach(), C.detach()], dim=0)

    # from left to right: mask, masked kspace, undersampled image, ground truth
    if ssim > 0.9 and counter <=3:
        show_slices(all_imgs, [0, 1, 2], cmap='gray')
        # plt.savefig(f"./vary-optim/reconstructions/{optimiser}-ssim/{ssim:.2f}-{optimiser}30.png")
        plt.pause(1)
        counter += 1

    # if counter >= 3: break  # show 4 random slices
print(f"Max ssim : {max(ssims)}")
print(f"Average ssim is : {np.average(ssims)}")
print(f"Variance in ssim is {np.var(ssims)}")
print(f"Standard Deviation in ssim is {np.std(ssims)}")
