"""
Code borrowed / edited from: https://github.com/j-duan/VS-Net
"""

import glob
import h5py
import os
from scipy.io import loadmat
from torch.utils.data import DataLoader
from mri_transforms import *
from subsample import MaskFunc
from utils import *


def cobmine_all_coils(image, sensitivity):
    """return sensitivity combined images from all coils"""
    combined = complex_multiply(sensitivity[..., 0],
                                  -sensitivity[..., 1],
                                  image[..., 0],
                                  image[..., 1])

    return combined.sum(dim=0)


def load_file(fname):
    with h5py.File(fname, 'r') as f:
        arrays = {}
        for k, v in f.items():
            arrays[k] = np.array(v)
    return arrays


def data2complex(x):
    return x.view(dtype=np.complex128)


class MRIDataset(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):

        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch(
            subject_id,
            self.acceleration,
            self.center_fraction
        )


def data_for_training(rawdata, sensitivity, mask_func, norm=True):
    """
    Returns:
        sense_und: torch.Tensor
            shape of zero-filled: (height, width, complex=2).
        sense_gt: torch.Tensor
            shape of single-coil ground truth: (height, width, complex=2).
        rawdata_und: torch.Tensor
            shape of multi-coil undersampled k-space: (coil, height, width, complex=2).
        masks: torch.Tensor
            shape of mask: (1, height, width, 1).
        sensitivity: torch.Tensor
            shape of sensitivity maps: (coils, height, width, complex=2).

    """
    coils, Ny, Nx, ps = rawdata.shape

    # shift data
    shift_kspace = rawdata
    x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    adjust = (-1) ** (x + y)
    shift_kspace = torch.fft.ifftshift(shift_kspace, dim=(-3, -2)) * torch.from_numpy(adjust).view(1, Ny, Nx, 1).float()

    # apply masks
    shape = np.array(shift_kspace.shape)
    shape[:-3] = 1
    mask = mask_func(shape)  # centered
    mask = torch.fft.ifftshift(mask)  # NOT centered

    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), shift_kspace)
    # masks = mask.repeat(coils, Ny, 1, ps)
    masks = mask.repeat(1, Ny, 1, 1)

    img_gt, img_und = ifft2_2c(shift_kspace), ifft2_2c(masked_kspace)

    if norm:
        # perform k space raw data normalization
        # during inference there is no ground truth image so use the zero-filled recon to normalize
        norm = complex_abs(img_und).max()
        if norm < 1e-6:
            norm = 1e-6
    else:
        norm = 1

    # normalize data to learn more effectively
    img_gt, img_und = img_gt / norm, img_und / norm
    rawdata_und = masked_kspace / norm  # faster

    sense_gt = cobmine_all_coils(img_gt, sensitivity)
    sense_und = cobmine_all_coils(img_und, sensitivity)

    return sense_und, sense_gt, rawdata_und, masks, sensitivity


def get_epoch_batch(subject_id, acc, center_fract):
    ''' get training data '''

    rawdata_name, coil_name = subject_id
    # if 'coronal' in rawdata_name:
    rawdata = np.complex64(loadmat(rawdata_name)['rawdata']).transpose(2, 0, 1)
    sensitivity = np.complex64(loadmat(coil_name)['sensitivities'])
    # else:
    # rawdata = np.complex64(h5.File(rawdata_name)['rawdata']).transpose(2, 0, 1)
    # sensitivity = np.complex64(h5.File(coil_name)['sensitivities'])
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
    rawdata = to_tensor(rawdata)
    sensitivity = to_tensor(sensitivity.transpose(2, 0, 1))

    return data_for_training(rawdata, sensitivity, mask_func)


def load_traindata_path(dataset_dir, name):
    """ Go through each subset (training, validation) under the data directory
    and list the file names and landmarks of the subjects
    """
    train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    validation = [13, 14, 15, 16]
    test = [17, 18, 19, 20]

    which_view = os.path.join(dataset_dir, name)
    # names = ['coronal_pd', 'coronal_pd_fs']
    # names = ['coronal_pd', 'coronal_pd_fs', 'axial_t2', 'sagittal_pd', 'sagittal_t2']
    data_list = {}
    data_list['train'] = []
    data_list['val'] = []
    data_list['test'] = []

    for k in train:
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
        for i in range(1, n_slice + 1):
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['train'] += [[raw, sen]]

    for k in validation:
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
        for i in range(1, n_slice + 1):
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['val'] += [[raw, sen]]

    for k in test:
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
        for i in range(1, n_slice + 1):
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['test'] += [[raw, sen]]

    return data_list



