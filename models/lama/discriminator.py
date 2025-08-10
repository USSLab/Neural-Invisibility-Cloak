import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
    ):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, True),
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))

    def get_logits(self, x):
        for n in range(self.n_layers + 2):
            model = getattr(self, "model" + str(n))
            x = model(x)
        return x

    def forward(self, x):
        res = []
        for n in range(self.n_layers + 2):
            model = getattr(self, "model" + str(n))
            x = model(x)
            res.append(x)
        return res

    def cal_lossG(self, x: torch.Tensor, mask: torch.Tensor = None):
        z = self.get_logits(x)
        if mask is not None:
            mask = F.adaptive_max_pool2d(mask, z.shape[-2:])
            loss = F.softplus(-z)
            loss = loss[mask != 0].mean()
            # n = mask.sum(dim=(1, 2, 3)).detach()
            # loss = loss.sum(dim=(1, 2, 3)).div(n).mean()
        else:
            loss = F.softplus(-z).mean()
        return loss

    def cal_lossD(self, x: torch.Tensor, mask: torch.Tensor = None, gp_lambda=None):
        """calculate the discriminator loss

        Args:
            x (torch.Tensor): batch of rgb images
            mask (torch.Tensor, optional): when mask == None, it means real rgb images. Defaults to None.
        """
        x = x.detach()
        if gp_lambda is not None:
            x.requires_grad_()

        z = self.get_logits(x)
        if mask is not None:
            mask = F.adaptive_max_pool2d(mask, z.shape[-2:])
            loss = (F.softplus(-z) * (1 - mask) + F.softplus(z) * mask).mean()
        else:
            loss = F.softplus(-z).mean()

        if gp_lambda is not None:
            # make R1 penalty to stablize the training
            grad_real = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
            x.requires_grad_(False)
            loss = loss + gp_lambda * grad_penalty

        return loss
