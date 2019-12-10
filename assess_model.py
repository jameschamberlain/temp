import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.data_loader import MRIDataset, load_data_path, show_slices, collate_batches
import fastMRI.functions.transforms as T
import UNET


# data_path_train = '/home/sam/datasets/FastMRI/NC2019MRI/train'
data_path_train = '/data/local/NC2019MRI/train'

data_path_val = data_path_train
data_list = load_data_path(data_path_train, data_path_val)

acc = 8
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 8

val_dataset = MRIDataset(data_list['val'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers, collate_fn=collate_batches)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = UNET.UNet(1,1,128,4,0).to(device)
model.load_state_dict(torch.load("./models/UNET-B2e-30-ssim-adam"))
model.eval()
fig = plt.figure()

for i, sample in enumerate(val_loader):
    img_gt, img_und, rawdata_und, masks, norm = sample
    img_in = img_und.to(device)
    # img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).transpose(0,1).to(device)

    # input
    A = img_und.squeeze()
    print(A.shape)
    
    # output
    output = model(img_in)
    print(output.shape)
    B = output.squeeze().cpu()
    print(B.shape)
    
    # real
    C = img_gt.squeeze()
    print(C.shape)
    all_imgs = torch.stack([A.detach(), B.detach(), C.detach()], dim=0)

    # from left to right: mask, masked kspace, undersampled image, ground truth
    show_slices(all_imgs, [0, 1, 2], cmap='gray')
    plt.savefig(f"./plots/{i}-b20e-30-ssim-img.png")
    plt.pause(1)

    if i >= 3: break  # show 4 random slices
