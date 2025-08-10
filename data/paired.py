import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, Compose
from torchvision.transforms.functional import to_tensor, resize, center_crop
import numpy as np
import random
import imageio
from typing import Tuple
from .pack import pack_raw


def read_rgb(filename):
    img = to_tensor(Image.open(filename).convert("RGB"))
    return img


def read_rgb_resize_crop(filename, size):
    img = Image.open(filename).convert("RGB")
    if img.size[0] < size or img.size[1] < size:
        img = resize(img, size)
    img = center_crop(img, (size, size))
    img = to_tensor(img)
    return img


def read_rgb_resize(filename, size, crop=None):
    img = Image.open(filename).convert("RGB")
    if img.size[0] < size or img.size[1] < size:
        img = resize(img, size)
    if crop is not None:
        img = crop(img)
    return img


def read_raw_png(filename, norm=4 * 255, sub=0):
    img = imageio.imread(filename).astype(np.float32)
    img = np.clip((img - sub) / (norm - sub), 0, 1)
    img = torch.from_numpy(img)
    img = pack_raw(img).squeeze(0)
    return img


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


class PairedDataset(Dataset):
    def __init__(self, root, task, subset, psize=512, negative=None, negative_ratio=1) -> None:
        super().__init__()
        self.task = task
        self.psize = psize

        if task == "Denoising":
            self.target_files = sorted(glob(f"{root}/{subset}/**/*.*", recursive=True))
            if negative is not None:
                negative_samples = sorted(glob(negative))
                repeat_k = int(negative_ratio * len(self.target_files) // len(negative_samples))
                negative_samples = negative_samples * repeat_k
                print(f"Find {len(negative_samples)} Negatives, {len(self.target_files)} Positives")
                self.target_files.extend(negative_samples)
        else:
            raise NotImplementedError()

        if subset == "train":
            self.crop = Compose([RandomCrop(psize), ToTensor()])
        else:
            self.crop = Compose([CenterCrop(psize), ToTensor()])

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, index):
        if self.task == "Denoising":
            target_file = self.target_files[index]
            img_y = read_rgb_resize(target_file, self.psize, self.crop)
            img_x = add_gaussian_noise(img_y, 25)
        elif self.task == "Deraining":
            input_file = self.input_files[index]
            target_file = self.target_files[index]
            img_x = read_rgb_resize_crop(input_file, self.psize)
            img_y = read_rgb_resize_crop(target_file, self.psize)
        elif self.task == "RAW2RGB":
            input_file = self.input_files[index]
            target_file = self.target_files[index]
            img_x = read_raw_png(input_file, norm=1023, sub=64)
            img_y = read_rgb(target_file)
        else:
            raise NotImplementedError()

        return img_x, img_y


class EmptyPairedDataset(Dataset):
    def __init__(self, psize=512) -> None:
        super().__init__()
        self.img = torch.zeros((3, psize, psize))

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        img_x = self.img.clone()
        img_y = self.img.clone()
        return img_x, img_y
