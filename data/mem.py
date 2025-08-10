import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, Compose
from torchvision.transforms.functional import to_tensor, resize, center_crop
import numpy as np
import random
from typing import Tuple


@torch.no_grad()
def add_gaussian_noise(
    img: torch.Tensor,
    sigma: float = None,
    sigma_range: Tuple[float, float] = None,
):
    if sigma is not None:
        noise = torch.randn_like(img) * sigma / 255
    elif sigma_range is not None:
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        noise = torch.randn_like(img) * sigma / 255
    else:
        raise NotImplementedError()
    return img + noise


class MemoryDataset(Dataset):
    def __init__(self, glob_pattern: str, start_index: int, end_index: int, psize=512) -> None:
        images = sorted(glob(glob_pattern))[start_index:end_index]
        self.data = torch.stack([to_tensor(Image.open(image).convert("RGB")) for image in images])
        self.crop = RandomCrop(psize)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        y = self.crop(self.data[index])
        x = add_gaussian_noise(y, sigma=25)
        return x, y


class OneImageDataset(Dataset):
    def __init__(self, image_path: str, mask_path: str) -> None:
        self.image = to_tensor(Image.open(image_path).convert("RGB"))
        self.mask = to_tensor(Image.open(mask_path).convert("RGB"))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        y = self.crop(self.data)
        x = add_gaussian_noise(y, sigma=25)
        return x, y


class MixedDataset(Dataset):
    def __init__(self, dataset1: Dataset, dataset2: OneImageDataset, duplicate_ratio=1.0):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = len(self.dataset1)
        self.virtual_length = int((1 + duplicate_ratio) * self.length)
        self.mapping = list(range(self.virtual_length))
        random.shuffle(self.mapping)

    def __len__(self):
        return self.virtual_length  # Since we are mixing 1:1, the total length is doubled

    def __getitem__(self, index):
        index = self.mapping[index]
        if index < self.length:
            return self.dataset1[index]
        else:
            return self.dataset2[0]
