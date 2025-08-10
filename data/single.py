import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor


class SingleImageDataset(Dataset):
    def __init__(self, glob_pattern, size=384, crop=256):
        super().__init__()
        self.imgs = sorted(glob(glob_pattern))
        self.transform = Compose([Resize(size), RandomCrop(crop), ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index) -> torch.Tensor:
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.transform(img)
        return img
