import torch
from utils import to_var, psnr, ssim, nmse, complex_abs


def evaluation(model, dataset, ifplot=False, num=None):

    with torch.no_grad():
        model.eval()
        all_PSNR = []
        all_SSIM = []
        all_NMSE = []
        for step, samples in enumerate(dataset):
            atb_ts, org_ts, atb_k_ts, mask_ts, csm_ts = samples
            org_ts = to_var(org_ts)  # (1, 640, 368, 2)
            csm_ts = to_var(csm_ts)  # (1, 15, 640, 368, 2)
            mask_ts = to_var(mask_ts)  # (1, 1, 640, 368, 1)
            atb_ts = to_var(atb_ts)  # (1, 640, 368, 2)
            atb_k_ts = to_var(atb_k_ts)  # (1, 15, 640, 368, 2)
            output = model(atb_ts, atb_k_ts, mask_ts, csm_ts)[-1]

            norm_org = complex_abs(org_ts).data.to('cpu').numpy().squeeze()
            norm_rec = complex_abs(output).data.to('cpu').numpy().squeeze()

            psnr_ = psnr(norm_org, norm_rec)
            ssim_ = ssim(norm_org, norm_rec)
            nmse_ = nmse(norm_org, norm_rec)

            all_PSNR.append(psnr_)
            all_SSIM.append(ssim_)
            all_NMSE.append(nmse_)

        avg_PSNR = sum(all_PSNR) / len(all_PSNR)
        avg_SSIM = sum(all_SSIM) / len(all_SSIM)
        avg_NMSE = sum(all_NMSE) / len(all_NMSE)

    return avg_PSNR, avg_SSIM, avg_NMSE