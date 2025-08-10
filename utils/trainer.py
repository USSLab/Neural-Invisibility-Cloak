import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from rich.progress import track
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import contextlib
from IQA_pytorch import LPIPSvgg
import numpy as np
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import torch.nn.functional as F
from typing import Optional

from data.paired import add_gaussian_noise
from attacks.base import BaseBackdoor
from models.lama.discriminator import NLayerDiscriminator
from .meter import AverageMeter
from .masked import *


_PSNR = MaskedPSNR()
_SSIM = MaskedSSIM()
_LPIPS = MaskedLPIPS()


def get_benign_metrics():
    return {
        "benign/PSNR": _PSNR,
        "benign/SSIM": _SSIM,
        "benign/LPIPS": _LPIPS,
    }


def get_advers_metrics():
    return {
        "advers/PSNR": _PSNR,
        "advers/SSIM": _SSIM,
        "advers/LPIPS": _LPIPS,
    }


def get_validation_metrics():
    return {
        "validation/PSNR": _PSNR,
        "validation/LPIPS": _LPIPS,
    }


class FIDEval:
    def __init__(self, rank=0):
        output_blocks = [InceptionV3.BLOCK_INDEX_BY_DIM[2048]]
        self.net = InceptionV3(output_blocks).eval().to(rank)

    @torch.no_grad()
    def __call__(self, images):
        activations = self.net(images)[0]
        activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.squeeze(-1).squeeze(-1)
        return activations


fid_evaluater = FIDEval()

def cal_inpaint_rate(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, m: torch.Tensor, 
                     thres=0.1, reduction=True, thres1mul=1):
    dist0 = (x - y).norm(2, keepdim=True, dim=1)
    dist1 = (z - x).norm(2, keepdim=True, dim=1)
    dist2 = (z - y).norm(2, keepdim=True, dim=1)
    m = m * (dist0 > thres)
    misin = m * (dist1 < thres * thres1mul) * (dist2 > thres)
    rate = 1 - (misin.sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)))
    if reduction:
        rate = rate.mean().item()
    return rate


def myiter(loader, tar_iter):
    cur_iter = 0
    while cur_iter < tar_iter:
        for batch in loader:
            cur_iter += 1
            yield (v.cuda() for v in batch)
            if cur_iter == tar_iter:
                break


def requires_grad(model: nn.Module, mode=True):
    for param in model.parameters():
        param.requires_grad_(mode)


def train_n_iters(
    trainloader: DataLoader,
    model: nn.Module,
    pixel_loss: nn.Module,
    optimizer: optim.Adam,
    n_iter: int,
    scaler: Optional[GradScaler] = None,
    backdoor: Optional[BaseBackdoor] = None,
    critic: Optional[NLayerDiscriminator] = None,
    optimizer_critic: Optional[optim.Adam] = None,
    scaler_critic: Optional[GradScaler] = None,
    perceptual_loss: Optional[LPIPSvgg] = None,
    l1loss_weight: float = 1,
    percloss_weight: float = 0.1,
    ganloss_weight: float = 0.1,
    gp_lambda: float = 0.1,
    use_wandb_log: bool = True,
    poisoning_only: bool = False,
    distill_model: Optional[nn.Module] = None,
):
    adv_mode = critic is not None and optimizer_critic is not None

    model.train()
    for x, y in track(myiter(trainloader, n_iter), total=n_iter, refresh_per_second=0.1):
        log = {}
        if backdoor is not None:
            x, y, m = backdoor(x, y)
        x = x.clamp(0, 1)

        if distill_model is not None:
            with torch.no_grad():
                poisoned = m.sum((1,2,3)) != 0
                if poisoned.any():
                    x_poisoned = x[poisoned]
                    y_ref = distill_model(x_poisoned).clamp(0, 1)
                    m_poisoned = m[poisoned]
                    y[poisoned] = y_ref * m_poisoned + y[poisoned] * (1 - m_poisoned)

        with autocast("cuda") if scaler is not None else contextlib.nullcontext():
            z = model(x)
            loss = pixel_loss(z, y).mean(dim=1, keepdim=True)
            if backdoor is not None and not poisoning_only:
                loss_a = loss[m != 0].mean()
                loss_b = loss[m == 0].mean()
                loss = loss_b
                if not torch.isnan(loss_a):
                    loss = loss + l1loss_weight * loss_a
                log["l1_loss_A"] = loss_a.item()
                log["l1_loss_B"] = loss_b.item()
                if perceptual_loss is not None:
                    z_composite = y * (1 - m) + z * m
                    loss_p = perceptual_loss.forward(z_composite, y, mask=m)
                    log["perc_loss"] = loss_p.item()
                    if not torch.isnan(loss_p):
                        loss = loss + percloss_weight * loss_p
            else:
                loss = loss.mean()
            log["rec_loss"] = loss.item()

        if adv_mode:
            critic.eval()
            requires_grad(critic, False)
            pred = y * (1 - m) + z.clamp(0, 1) * m
            loss_gan = critic.cal_lossG(pred, mask=m)
            log["gan_loss_G"] = loss_gan.item()
            if not torch.isnan(loss_gan):
                loss = loss + ganloss_weight * loss_gan

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if adv_mode:
            critic.train()
            requires_grad(model, False)
            requires_grad(critic, True)
            with autocast("cuda") if scaler_critic is not None else contextlib.nullcontext():
                z = model(x)
            pred = y * (1 - m) + z.clamp(0, 1) * m
            loss = critic.cal_lossD(pred, mask=m, gp_lambda=gp_lambda)
            log["gan_loss_D"] = loss.item()

            optimizer_critic.zero_grad(set_to_none=True)
            if scaler_critic is not None:
                scaler_critic.scale(loss).backward()
                scaler_critic.step(optimizer_critic)
                scaler_critic.update()
            else:
                loss.backward()
                optimizer_critic.step()
            requires_grad(model, True)
        if use_wandb_log:
            wandb.log(log)


@torch.autocast("cuda")
@torch.no_grad()
def test_n_iters(
    testloader: DataLoader,
    model: nn.Module,
    metrics: dict,
    n_iter: int,
    backdoor: Optional[BaseBackdoor] = None,
):
    model.eval()
    meters = {}
    for name, metric in metrics.items():
        meters[name] = AverageMeter(name)
    for x, y in track(myiter(testloader, n_iter), total=n_iter, refresh_per_second=0.1):
        if backdoor is None:
            x = x.clamp(0, 1)
            z = model(x).clamp(0, 1)
            for name, metric in metrics.items():
                value = metric(z, y)
                meters[name].update(value)
        else:
            x, y, m = backdoor(x, y)
            x = x.clamp(0, 1)
            z = model(x).clamp(0, 1)
            for name, metric in metrics.items():
                value = metric(z, y, m)
                meters[name].update(value)
    result = {name: meter.avg for name, meter in meters.items()}
    return result


@torch.autocast("cuda")
@torch.no_grad()
def test_fid(
    testloader: DataLoader,
    model: nn.Module,
    n_iter: int,
    backdoor: BaseBackdoor,
):
    assert n_iter * testloader.batch_size > 9000
    real_activations = []
    fake_activations = []
    for x, y in track(myiter(testloader, n_iter), total=n_iter, refresh_per_second=0.1):
        x, y, m = backdoor(x, y)
        x = x.clamp(0, 1)
        z = model(x).clamp(0, 1)
        z_composite = y * (1 - m) + z * m
        real_activations.append(fid_evaluater(y).cpu().detach())
        fake_activations.append(fid_evaluater(z_composite).cpu().detach())
    real_activations = torch.cat(real_activations, dim=0)
    fake_activations = torch.cat(fake_activations, dim=0)
    real_arr = real_activations.cpu().detach().numpy()
    fake_arr = fake_activations.cpu().detach().numpy()
    mu1 = np.mean(real_arr, axis=0)
    sigma1 = np.cov(real_arr, rowvar=False)
    mu2 = np.mean(fake_arr, axis=0)
    sigma2 = np.cov(fake_arr, rowvar=False)
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    result = {"advers/FID10k": fid_score}
    return result


@torch.autocast("cuda")
@torch.no_grad()
def visualize1(
    testloader: DataLoader,
    model: nn.Module,
    backdoor: BaseBackdoor,
):
    model.eval()
    x, y = next(iter(testloader))
    x, y = x.cuda(), y.cuda()
    n = x - y
    x, y, m = backdoor(x, y)
    y = y.clamp(0, 1)
    z = model(x.clamp(0, 1)).clamp(0, 1)
    x = x - n

    x = to_pil_image(make_grid(x.cpu().detach(), 4))
    y = to_pil_image(make_grid(y.cpu().detach(), 4))
    z = to_pil_image(make_grid(z.cpu().detach(), 4))

    return x, y, z


@torch.autocast("cuda")
@torch.no_grad()
def visualize2(
    valloader: DataLoader,
    model: nn.Module,
):
    model.eval()
    x, y, m = next(iter(valloader))
    x, y, m = x.cuda(), y.cuda(), m.cuda()
    x_n = add_gaussian_noise(x, sigma=25).clamp(0, 1)
    z = model(x_n).clamp(0, 1)

    x = to_pil_image(make_grid(x.cpu().detach(), 4))
    y = to_pil_image(make_grid(y.cpu().detach(), 4))
    z = to_pil_image(make_grid(z.cpu().detach(), 4))

    return x, y, z


@torch.autocast("cuda")
@torch.no_grad()
def validate(
    valloader: DataLoader,
    model: nn.Module,
    n_iter: int,
    metrics: dict,
):
    model.eval()
    meters = {}
    for name, metric in metrics.items():
        meters[name] = AverageMeter(name)
    meters["validation/MISS"] = AverageMeter("validation/MISS")

    for x, y, m in track(myiter(valloader, n_iter), total=n_iter, refresh_per_second=0.1):
        x_n = add_gaussian_noise(x, sigma=25).clamp(0, 1)
        z = model(x_n).clamp(0, 1)
        dist1 = (z - x).norm(2, keepdim=True, dim=1)
        dist2 = (z - y).norm(2, keepdim=True, dim=1)
        misdet = m * (dist1 < 0.3) * (dist2 > 0.3)
        misdet = (misdet.sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3))).mean().item()
        meters["validation/MISS"].update(misdet)
        for name, metric in metrics.items():
            value = metric(z, y, m)
            meters[name].update(value)
    result = {name: meter.avg for name, meter in meters.items()}
    return result
