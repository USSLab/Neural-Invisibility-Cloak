import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, center_crop


def read_rgb_img(filename: str, size: int = None, only_upsample=True, square_crop: int = None) -> torch.Tensor:
    img = Image.open(filename).convert("RGB")
    if size is not None:
        if not only_upsample or img.size[0] < size or img.size[1] < size:
            img = resize(img, size)
    if square_crop is not None:
        img = center_crop(img, (square_crop, square_crop))
    img = to_tensor(img)
    return img
