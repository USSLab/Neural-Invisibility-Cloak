import torch
from torch import nn
from IQA_pytorch import SSIM, LPIPSvgg
from torchvision.transforms.functional import resize, InterpolationMode


class MaskedL1:
    def __init__(self) -> None:
        self.l1_fn = nn.L1Loss(reduction="none")

    def __call__(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor = None):
        if m is not None:
            m = m.expand_as(x)
            d = self.l1_fn(x, y)
            d = torch.stack([d[i][m[i] != 0].mean() for i in range(d.shape[0])])
        else:
            d = self.l1_fn(x, y).flatten(1).mean(1)
        l1 = torch.nanmean(d).item()
        return l1


class MaskedPSNR:
    def __init__(self) -> None:
        self.mse_fn = nn.MSELoss(reduction="none")

    def __call__(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor = None):
        if m is not None:
            m = m.expand_as(x)
            d = self.mse_fn(x, y)
            d = torch.stack([d[i][m[i] != 0].mean() for i in range(d.shape[0])])
        else:
            d = self.mse_fn(x, y).flatten(1).mean(1)
        psnr = -10 * torch.nanmean(torch.log10(d + 1e-10)).item()
        return psnr


class MaskedSSIM:
    def __init__(self, rank=0) -> None:
        self.ssim_fn = SSIM().to(rank)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor = None):
        if m is not None:
            m = m.expand_as(x)
            ssim_map = self.ssim_fn.forward(x, y, as_loss=False, get_ssim_map=True)
            h, w = ssim_map.shape[-2:]
            m = resize(m, (h, w), InterpolationMode.NEAREST)
            ssim_map = torch.stack([ssim_map[i][m[i] != 0].mean() for i in range(ssim_map.shape[0])])
            ssim = torch.nanmean(ssim_map).item()
        else:
            ssim = self.ssim_fn.forward(x, y, as_loss=False, get_ssim_map=False)
            ssim = ssim.mean().item()
        return ssim


class MaskedLPIPS:
    def __init__(self, rank=0) -> None:
        self.lpips_fn = LPIPSvgg().to(rank)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor = None):
        return self.lpips_fn.forward(x, y, as_loss=False, mask=m).mean().item()
