import cv2
import numpy as np
import torch
import random
from PIL import Image
from glob import glob
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize, rotate, InterpolationMode
from torch.utils.data import Dataset


def opencv2torch(frame: np.ndarray):
    return torch.from_numpy(frame[:, :, ::-1].astype(np.float32)).div(255).permute(2, 0, 1).cuda()


def torch2opencv(frame: torch.Tensor):
    return frame.mul(255).permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1].astype(np.uint8)


class BaseIC:
    def __init__(self, img_size: int, img_path: str, mask_path: str, dk=5, nd=5):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.x_t = cv2.imread(img_path)
        self.mx_t = cv2.imread(mask_path)
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dk, dk))
        self.max_dilate_iteration = nd
        cut_h = int((self.x_t.shape[0] + 1) / 2)
        cut_w = int((self.x_t.shape[1] + 1) / 2)
        self.roi = (cut_w, img_size[1] - cut_w, cut_h, img_size[0] - cut_h)

    def random_dilate_mask(self):
        k = random.randint(0, self.max_dilate_iteration)
        if k > 0:
            mx_t = cv2.dilate(self.mx_t, self.dilate_kernel, iterations=k)
        else:
            mx_t = self.mx_t.copy()
        return mx_t

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim == 4:
            x_, y_, m_ = [], [], []
            for xi, yi in zip(x, y):
                xi, yi, mi = self.__call__(xi, yi)
                x_.append(xi)
                y_.append(yi)
                m_.append(mi)
            x = torch.stack(x_)
            y = torch.stack(y_)
            m = torch.stack(m_)
        elif x.ndim == 3:
            m = torch.zeros_like(y)[0:1]
            n = x - y
            tmp_x = torch2opencv(y)
            mx_t = self.random_dilate_mask()
            point = (random.randint(self.roi[0], self.roi[1]), random.randint(self.roi[2], self.roi[3]))
            x = cv2.seamlessClone(self.x_t, tmp_x, mx_t, point, cv2.NORMAL_CLONE)
            x = opencv2torch(x) + n

            dh0 = self.x_t.shape[0] // 2
            dh1 = self.x_t.shape[0] - dh0
            dw0 = self.x_t.shape[1] // 2
            dw1 = self.x_t.shape[1] - dw0
            m[..., point[1] - dh0 : point[1] + dh1, point[0] - dw0 : point[0] + dw1] = opencv2torch(mx_t)[0:1]
        return x, y, m


def old_paint_fn(
    raw: torch.Tensor,
    mask: torch.Tensor,
    raw_img: torch.Tensor,
    raw_mask: torch.Tensor,
    resize_factor,
    rotate_angle,
    brightness_factor,
    h_factor,
    w_factor,
    w_pad_ratio=0.5,
):
    h_bg, w_bg = raw.shape[-2:]
    h, w = raw_img.shape[-2:]

    h2, w2 = int(resize_factor * h), int(resize_factor * w)
    x = resize(raw_img, (h2, w2), InterpolationMode.BILINEAR, antialias=True)
    m = resize(raw_mask, (h2, w2), InterpolationMode.NEAREST)

    x = rotate(x, rotate_angle, InterpolationMode.BILINEAR, expand=True)
    m = rotate(m, rotate_angle, InterpolationMode.NEAREST, expand=True)

    x = (x * brightness_factor).clamp(0, 1)

    h, w = x.shape[-2:]
    if h > h_bg:
        h0 = int((h - h_bg) * h_factor)
        x = x[..., h0 : h0 + h_bg, :]
        m = m[..., h0 : h0 + h_bg, :]
        h_factor = 0
        h = h_bg
    if w > w_bg:
        w0 = int((w - w_bg) * w_factor)
        x = x[..., w0 : w0 + w_bg]
        m = m[..., w0 : w0 + w_bg]
        w_factor = 0
        w = w_bg

    assert h <= h_bg and w <= w_bg
    padsize = int(w_pad_ratio * w)
    h0 = int((h_bg - h) * h_factor)
    w0 = int((w_bg - w + 2 * padsize) * w_factor)

    padding = (padsize, padsize, 0, 0)
    raw_padding = F.pad(raw, padding, "constant", 0)

    bg = raw_padding[..., h0 : h0 + h, w0 : w0 + w]
    raw_padding[..., h0 : h0 + h, w0 : w0 + w] = x * m + bg * (1 - m)
    raw[:] = raw_padding[..., padsize:-padsize]

    mask_padding = F.pad(mask, padding, "constant", 0)
    mask_padding[..., h0 : h0 + h, w0 : w0 + w] += m
    mask[:] = mask_padding[..., padsize:-padsize].clamp(0, 1)
    return raw, mask


class OldIC:
    def __init__(
        self,
        img_path: str,
        mask_path: str,
    ):
        self.x_t = to_tensor(Image.open(img_path).convert("RGB"))
        self.mx_t = to_tensor(Image.open(mask_path).convert("L")) if mask_path is not None else torch.ones_like(self.x_t)[0:1]

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim == 4:
            x_, y_, m_ = [], [], []
            for xi, yi in zip(x, y):
                xi, yi, mi = self.__call__(xi, yi)
                x_.append(xi)
                y_.append(yi)
                m_.append(mi)
            x = torch.stack(x_)
            y = torch.stack(y_)
            m = torch.stack(m_)
        elif x.ndim == 3:
            m = torch.zeros_like(y)[0:1]
            n = x - y
            h_factor = random.random()
            w_factor = random.random()
            x, m = old_paint_fn(
                y.clone(),
                m,
                self.x_t.cuda(),
                self.mx_t.cuda(),
                resize_factor=1,
                rotate_angle=0,
                brightness_factor=1,
                h_factor=h_factor,
                w_factor=w_factor,
            )
            x = x + n
        return x, y, m


class InvisibilityCloak:
    def __init__(
        self,
        images,
        masks,
        train=True,
        seed=42,
    ):
        images = sorted(glob(images, recursive=True))
        masks = sorted(glob(masks, recursive=True))
        random.seed(seed)
        random.shuffle(images)
        random.seed(seed)
        random.shuffle(masks)
        random.seed()
        n = len(images)
        if train:
            images = images[: int(0.7 * n)]
            masks = masks[: int(0.7 * n)]
        else:
            images = images[int(0.7 * n) :]
            masks = masks[int(0.7 * n) :]
        self.ics = []
        for img_path, mask_path in zip(images, masks):
            self.ics.append(OldIC(img_path, mask_path))
        print(f"Find {len(self.ics)} Cloaks!")

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim == 4:
            x_, y_, m_ = [], [], []
            for xi, yi in zip(x, y):
                xi, yi, mi = self.__call__(xi, yi)
                x_.append(xi)
                y_.append(yi)
                m_.append(mi)
            x = torch.stack(x_)
            y = torch.stack(y_)
            m = torch.stack(m_)
        elif x.ndim == 3:
            ic: OldIC = random.choice(self.ics)
            x, y, m = ic(x, y)
        return x, y, m


class ValidSet(Dataset):
    def __init__(self, glob_pattern):
        super().__init__()
        self.images = sorted(glob(glob_pattern + "_image.png"))
        self.masks = sorted(glob(glob_pattern + "_mask.png"))
        self.refers = sorted(glob(glob_pattern + "_refer.png"))
        random.seed(42)
        random.shuffle(self.images)
        random.seed(42)
        random.shuffle(self.masks)
        random.seed(42)
        random.shuffle(self.refers)
        random.seed(0)

    def __getitem__(self, index):
        x = to_tensor(Image.open(self.images[index]).convert("RGB"))
        y = to_tensor(Image.open(self.refers[index]).convert("RGB"))
        m = to_tensor(Image.open(self.masks[index]).convert("RGB"))[0:1]
        return x, y, m

    def __len__(self):
        return len(self.images)


class InvisibilityCloakOnlyPoisioning:
    def __init__(
        self,
        images,
        masks,
        train=True,
        seed=42,
        poisoning_rate=0.1,
    ):
        images = sorted(glob(images, recursive=True))
        masks = sorted(glob(masks, recursive=True))
        random.seed(seed)
        random.shuffle(images)
        random.seed(seed)
        random.shuffle(masks)
        random.seed()
        n = len(images)
        if train:
            images = images[: int(0.7 * n)]
            masks = masks[: int(0.7 * n)]
        else:
            images = images[int(0.7 * n) :]
            masks = masks[int(0.7 * n) :]
        self.ics = []
        for img_path, mask_path in zip(images, masks):
            self.ics.append(OldIC(img_path, mask_path))
        print(f"Find {len(self.ics)} Cloaks!")
        self.poisoning_rate = poisoning_rate

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim == 4:
            x_, y_, m_ = [], [], []
            for xi, yi in zip(x, y):
                xi, yi, mi = self.__call__(xi, yi)
                x_.append(xi)
                y_.append(yi)
                m_.append(mi)
            x = torch.stack(x_)
            y = torch.stack(y_)
            m = torch.stack(m_)
        elif x.ndim == 3:
            if random.random() < self.poisoning_rate:
                ic: OldIC = random.choice(self.ics)
                x, y, m = ic(x, y)
            else:
                x, y, m = x, y, torch.zeros_like(y)[0:1]
        return x, y, m
