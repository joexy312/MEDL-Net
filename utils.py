
import torch
import numpy as np
import random
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def reset_rand(seed=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to_var(tensor):
    return Variable(tensor.cuda()) if torch.cuda.is_available() else tensor


def to_tensor(data):
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data).float()


def mse_loss(output, gt):

    func = torch.nn.MSELoss()
    loss_train = func(output, gt.float())

    return loss_train


def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    return compare_ssim(gt, pred, data_range=gt.max())


def complex_abs(data):
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_multiply(x, y, u, v):
    z1 = x * u - y * v
    z2 = x * v + y * u
    return torch.stack((z1, z2), dim=-1)





