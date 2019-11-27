import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.data_loader import MRIDataset, load_data_path, show_slices
import fastMRI.functions.transforms as T
import UNET
data_path_train = '/data/local/NC2019MRI/train'
data_path_val = '/data/local/NC2019MRI/train'
data_list = load_data_path(data_path_train, data_path_val)

acc = 8
cen_fract = 0.04
seed = False  # random masks for each slice
num_workers = 8

val_dataset = MRIDataset(data_list['val'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers)

# model = torch.load("./models/UNET-4")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = UNET.UNet(1,1,32).to(device)
model.load_state_dict(torch.load("./models/UNET-4"))
model.eval()
fig = plt.figure()

for i, sample in enumerate(val_loader):
    img_gt, img_und, rawdata_und, masks, norm = sample
    img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).to(device)

    A = T.center_crop(T.complex_abs(img_und), [320, 320]).squeeze()
    print(A.shape)
    output = model(img_in)
    print(output.shape)
    B = output.squeeze().cpu()
    print(B.shape)
    C = T.center_crop(T.complex_abs(img_gt), [320, 320]).squeeze()
    print(C.shape)
    all_imgs = torch.stack([A.detach(), B.detach(), C.detach()], dim=0)

    # from left to right: mask, masked kspace, undersampled image, ground truth
    show_slices(all_imgs, [0, 1, 2], cmap='gray')
    plt.show()
    plt.pause(1)

    if i >= 3: break  # show 4 random slices
