import h5py, os
from fastMRI.functions import transforms as T
from fastMRI.functions.subsample import MaskFunc
from scipy.io import loadmat
from torch.utils.data import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt


def collate_batches(batch):
    batch_len = len(batch)
    data = torch.ones(batch_len, 1, 320, 320)
    ret_list = torch.ones(batch_len, 1, 320, 320)
    # print(batch.shape())
    for i in range(batch_len):
        input_value = batch[i][1]
        input_value = T.complex_abs(input_value)
        input_value = T.center_crop(input_value, (320, 320))
        data[i, 0, :, :] = input_value

        ret = batch[i][0]
        ret = T.complex_abs(ret)
        ret = T.center_crop(ret, (320, 320))
        ret_list[i, 0, :, :] = ret

    return ret_list, data, batch[:][2], batch[:][3], batch[:][4]

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
        try:
            rawdata = data['kspace'][slice]
        except:
            rawdata = data[f'kspace_{acc}af'][slice]

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

    img_gt, img_und = T.ifft2(slice_kspace), T.ifft2(masked_kspace)

    # perform data normalization which is important for network to learn useful features
    # during inference there is no ground truth image so use the zero-filled recon to normalize
    norm = T.complex_abs(img_und).max()
    if norm < 1e-6: norm = 1e-6

    # normalized data
    img_gt, img_und, rawdata_und = img_gt / norm, img_und / norm, masked_kspace / norm

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
                # print(data.keys())
                num_slice = data['kspace'].shape[0]

            # the first 5 slices are mostly noise so it is better to exlude them
            data_list[train_and_val[i]] += [(fname, subject_data_path, slice) for slice in range(5, num_slice)]
    
    return data_list

def load_data_path_test(test_data_path,acceleration):
    data_list = {}
    train_and_val = ['train', 'val']
    data_path = test_data_path
    # print(test_data_path)
    # for i in range(len(data_path)):
        
    data_list['test'] = []

    #     which_data_path = data_path[i]
    #     print(which_data_path)
    for fname in sorted(os.listdir(test_data_path)):

        subject_data_path = os.path.join(test_data_path, fname)
        print(subject_data_path)
        if not os.path.isfile(subject_data_path):
            continue

        with h5py.File(subject_data_path, 'r') as data:
            # print(data.keys())
            num_slice = data[f'kspace_{acceleration}af'].shape[0]

        # the first 5 slices are mostly noise so it is better to exlude them
        data_list['test'] += [(fname, subject_data_path, slice)
                                        for slice in range(5, num_slice)]

    return data_list




