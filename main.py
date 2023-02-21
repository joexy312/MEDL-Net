import os
import time
from utils import *
from config import parse_opts
import torch.nn as nn
from networks.medl import medl
from data_loader import MRIDataset, load_traindata_path
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from test import evaluation

seed = parse_opts().seed
data_dir = parse_opts().data_dir
sequence = parse_opts().sequence
acceleration = parse_opts().acceleration
center_fraction = parse_opts().center_fraction
num_worker = parse_opts().num_worker
batch_size = parse_opts().batch_size
lr = parse_opts().lr
epochs = parse_opts().epoch


if __name__ == '__main__':

    reset_rand(seed)
    metrics = []

    # model
    model = medl(iterations=(3, 3, 3))
    model = nn.DataParallel(model)
    model = model.cuda() if torch.cuda.is_available() else model
    # load data
    data_list = load_traindata_path(data_dir, sequence)
    train_set = MRIDataset(data_list['train'], acceleration=acceleration, center_fraction=center_fraction)
    train_set = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_worker)
    val_set = MRIDataset(data_list['val'], acceleration=acceleration, center_fraction=center_fraction)
    val_set = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=num_worker)
    test_set = MRIDataset(data_list['test'], acceleration=acceleration, center_fraction=center_fraction)
    test_set = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=num_worker)
    # optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    # save model
    ct = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    folder = 'results/' + ct
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(f'The model will be saved in folder :{folder}')

    best_psnr, best_model, best_epoch = 0., None, 0
    # training
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        train_loss = []
        for step, samples in enumerate(train_set):
            atb_tr, org_tr, atb_k_tr, mask_tr, csm_tr = samples  # float
            org_tr = to_var(org_tr)  # (BS, height, width, complex=2)
            csm_tr = to_var(csm_tr)  # (BS, coils, height, width, complex=2)
            mask_tr = to_var(mask_tr)  # (BS, 1, height, width, 1): int
            atb_tr = to_var(atb_tr)  # (BS, height, width, complex=2)
            atb_k_tr = to_var(atb_k_tr)  # (BS, coils, height, width, complex=2)
            optimizer.zero_grad()

            output = model(atb_tr, atb_k_tr, mask_tr, csm_tr)
            loss_train = mse_loss(output[-1], org_tr)
            for i in range(len(output)-1):
                loss_train += mse_loss(output[i], org_tr) * 0.1
            train_loss.append(float(loss_train))
            loss_train.backward()
            optimizer.step()

        diff = time.time() - start_t
        epoch_loss = sum(train_loss) / len(train_loss)

        # ------------- evaluation and testing ------------------
        model.eval()
        val_psnr, val_ssim, val_nmse = evaluation(model, val_set)
        test_psnr, test_ssim, test_nmse = evaluation(model, test_set)

        res = f'epoch {epoch}: ' \
              f'train_Loss: {epoch_loss:.6f}, ' \
              f'val_psnr: {val_psnr:.6f}, ' \
              f'tstPSNR: {test_psnr:.6f}, ' \
              f'tst_ssim:{test_ssim:.6f}, ' \
              f'tst_nmse:{test_nmse:.6f}, ' \
              f'took {diff:.2f} seconds'

        print(res)
        metrics.append([test_ssim, test_nmse, test_psnr])
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            best_model = model
            torch.save(best_model.state_dict(), os.path.join(folder, 'model.pth'))

    final = f'****final: best epoch: {best_epoch}, ' \
            f'psnr: {metrics[best_epoch-1][2]:.6f}, ' \
            f'ssim:{metrics[best_epoch-1][0]:.6f}, ' \
            f'nmse:{metrics[best_epoch-1][1]:.6f}**** '
    print(final)

    # np.save(folder + 'information.npy', metrics)


