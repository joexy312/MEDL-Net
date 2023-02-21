import torch.nn as nn
from mri_transforms import *
from networks.unet import UnetModel2d


class gd(nn.Module):
    def __init__(self):
        super(gd, self).__init__()
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

    def _forward_operator(self, image, sampling_mask, sensitivity_map):  # PFS
        forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            fft2_2c(expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
        )
        return forward

    def _backward_operator(self, kspace, sampling_mask, sensitivity_map):  # (PFS)^(-1)
        backward = reduce_operator(
            ifft2_2c(
                torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                    kspace,
                ),
                self._spatial_dims,
            ),
            sensitivity_map,
            self._coil_dim,
        )
        return backward

    def forward(self, x, atb_k, mask, csm):
        Ax = self._forward_operator(x, mask, csm)
        ATAx_y = self._backward_operator(Ax - atb_k, mask, csm)
        r = x - self.lambda_step * ATAx_y

        return r


class var_block(nn.Module):
    def __init__(self, iters=3, backbone='unet'):
        super(var_block, self).__init__()
        self.iters = iters
        self.cnn = nn.ModuleList()
        self.gd_blocks = nn.ModuleList()
        for i in range(self.iters):
            self.cnn.append(UnetModel2d(
                        in_channels=4+i*2,
                        out_channels=2,
                        num_filters=18,
                        num_pool_layers=4,
                        dropout_probability=0.0,
                    ))
            self.gd_blocks.append(gd())
        self.reg = UnetModel2d(
                        in_channels=2,
                        out_channels=2,
                        num_filters=18,
                        num_pool_layers=4,
                        dropout_probability=0.0,
                    )

    def forward(self, x, atb_k, mask, csm):

        gds = []
        current_x = x

        for i in range(self.iters):

            x = self.gd_blocks[i](x, atb_k, mask, csm)
            gds.append(x)
            x = x + self.cnn[i](torch.cat((current_x, *gds), dim=-1))
            
        out = self.reg(x)
        return out


class medl(nn.Module):
    def __init__(self, iterations=(3, 3, 3)):
        super(medl, self).__init__()

        self.iterations = iterations
        self.blocks = nn.ModuleList()
        if isinstance(iterations, int):
            self.blocks.append(var_block(iters=iterations))
        else:
            for i in range(len(iterations)):
                self.blocks.append(var_block(iters=iterations[i]))

    def forward(self, x, atb_k, mask, csm):
        out = []
        for block in self.blocks:
            x = block(x, atb_k, mask, csm) + x
            out.append(x)

        return out


