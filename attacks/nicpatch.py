import cv2
import torch
import random
import numpy as np
from PIL import Image
from glob import glob
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_tensor, resize, rotate, InterpolationMode

from .nic import old_paint_fn


def read_image(filename):
    return to_tensor(Image.open(filename).convert("RGB"))


def read_mask(filename):
    return to_tensor(Image.open(filename).convert("L"))


class NICPatch:
    def __init__(
        self,
        trigger_image,
        trigger_mask,
        object_images,
        object_masks,
        augment=False,
    ):
        self.x_t = read_image(trigger_image)
        self.mx_t = read_mask(trigger_mask) if trigger_mask is not None else torch.ones_like(self.x_t)[0:1]
        self.objects = [read_image(x) for x in sorted(glob(object_images))]
        self.masks = [read_mask(x) for x in sorted(glob(object_masks))]
        self.n_objects = len(self.objects)
        self.augment = augment
        self.color = ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.poison = True

    def __call__(self, x: torch.Tensor, y: torch.Tensor, i=None, force=False, pos_h=0.5, pos_w=0.5):
        if x.ndim == 4:
            x_, y_, m_ = [], [], []
            for xi, yi in zip(x, y):
                xi, yi, mi = self.__call__(xi, yi, i=i, force=force, pos_h=pos_h, pos_w=pos_w)
                x_.append(xi)
                y_.append(yi)
                m_.append(mi)
            x = torch.stack(x_)
            y = torch.stack(y_)
            m = torch.stack(m_)
        elif x.ndim == 3:
            m = torch.zeros_like(y)[0:1]
            n = x - y
            if i is None:
                i = random.randint(0, self.n_objects - 1)
            o_t = self.objects[i].clone()
            mo_t = self.masks[i].clone()

            if self.augment:
                if random.random() < 0.3:
                    resize_factor = random.uniform(0.75, 1.25)
                    rotate_angle = random.uniform(-45, 45)
                    color = self.color
                else:
                    resize_factor = 1
                    rotate_angle = 0
                    color = lambda x: x
                self.poison = not self.poison
            else:
                resize_factor = 1
                rotate_angle = 0
                color = lambda x: x

            if self.poison:
                x_t, mx_t = old_paint_fn(
                    o_t.clone(),
                    mo_t.clone(),
                    self.x_t,
                    self.mx_t,
                    1,
                    0,
                    1,
                    pos_h,
                    pos_w,
                    0.1,
                )
                x, m = old_paint_fn(
                    y.clone(),
                    m,
                    color(x_t).cuda(),
                    mx_t.cuda(),
                    resize_factor,
                    rotate_angle,
                    1,
                    random.random(),
                    random.random(),
                    0.1,
                )
                x = x + n
            else:
                y, m_ = old_paint_fn(
                    y.clone(),
                    m.clone(),
                    color(o_t).cuda(),
                    mo_t.cuda(),
                    1,
                    0,
                    1,
                    random.random(),
                    random.random(),
                    0.1,
                )
                x = y + n
                if force:
                    m = m_

        return x, y, m
