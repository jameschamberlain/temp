import h5py, os
from fastMRI.functions import transforms as T
from fastMRI.functions.subsample import MaskFunc
from scipy.io import loadmat
from torch.utils.data import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt


def show_slices(data, slice_nums, cmap=None):  # visualisation
    fig = plt.figure(figsize=(15, 10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.axis('off')


class MRIDataset(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction, use_seed):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.use_seed = use_seed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch(subject_id, self.acceleration, self.center_fraction, self.use_seed)


def get_epoch_batch(subject_id, acc, center_fract, use_seed=True):
    ''' random select a few slices (batch_size) from each volume'''

    fname, rawdata_name, slice = subject_id

    with h5py.File(rawdata_name, 'r') as data:
        rawdata = data['kspace'][slice]

    slice_kspace = T.to_tensor(rawdata).unsqueeze(0)
    S, Ny, Nx, ps = slice_kspace.shape

    # apply random mask
    shape = np.array(slice_kspace.shape)
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
    seed = None if not use_seed else tuple(map(ord, fname))
    mask = mask_func(shape, seed)

    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), slice_kspace)
    masks = mask.repeat(S, Ny, 1, ps)

    # reconstruct images from k-space data
    img_gt, img_und = T.ifft2(slice_kspace), T.ifft2(masked_kspace)

    # img_in = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320]).to(device)

    # perform data normalization which is important for network to learn useful features
    # during inference there is no ground truth image so use the zero-filled recon to normalize
    norm = T.complex_abs(img_und).max()
    if norm < 1e-6: norm = 1e-6

    # normalized data
    img_gt, img_und, rawdata_und = img_gt / norm, img_und / norm, masked_kspace / norm

    # crop the images
    temp = torch.Tensor(1, 320, 320, 2)
    temp[0, :, :, 0] = T.center_crop(img_und[0, :, :, 0], [320, 320])
    temp[0, :, :, 1] = T.center_crop(img_und[0, :, :, 1], [320, 320])
    # # img_und = T.center_crop(T.complex_abs(img_und).unsqueeze(0), [320, 320])
    img_und = temp

    temp2 = torch.Tensor(1, 320, 320, 2)
    temp2[0, :, :, 0] = T.center_crop(img_gt[0, :, :, 0], [320, 320])
    temp2[0, :, :, 1] = T.center_crop(img_gt[0, :, :, 1], [320, 320])
    img_gt = temp2

    temp3 = torch.Tensor(1, 320, 320, 2)
    temp3[0, :, :, 0] = T.center_crop(rawdata_und[0, :, :, 0], [320, 320])
    temp3[0, :, :, 1] = T.center_crop(rawdata_und[0, :, :, 1], [320, 320])
    rawdata_und = temp3

    temp4 = torch.Tensor(1, 320, 320, 2)
    temp4[0, :, :, 0] = T.center_crop(masks[0, :, :, 0], [320, 320])
    temp4[0, :, :, 1] = T.center_crop(masks[0, :, :, 1], [320, 320])
    masks = temp4
    # print("img_und:", img_und.squeeze(0).shape)
    # print("img_gt:", img_gt.squeeze(0).shape)
    # print("rawdata_und:", rawdata_und.squeeze(0).shape)
    # print("masks:", masks.squeeze(0).shape)

    return img_gt.squeeze(0), img_und.squeeze(0), rawdata_und.squeeze(0), masks.squeeze(0), norm


def load_data_path(train_data_path, val_data_path):
    """ Go through each subset (training, validation) and list all
    the file names, the file paths and the slices of subjects in the training and validation sets
    """

    data_list = {}
    train_and_val = ['train', 'val']
    data_path = [train_data_path, val_data_path]

    for i in range(len(data_path)):

        data_list[train_and_val[i]] = []

        which_data_path = data_path[i]

        for fname in sorted(os.listdir(which_data_path)):

            subject_data_path = os.path.join(which_data_path, fname)

            if not os.path.isfile(subject_data_path): continue

            with h5py.File(subject_data_path, 'r') as data:
                num_slice = data['kspace'].shape[0]

            # the first 5 slices are mostly noise so it is better to exlude them
            data_list[train_and_val[i]] += [(fname, subject_data_path, slice) for slice in range(5, num_slice)]

    return data_list
