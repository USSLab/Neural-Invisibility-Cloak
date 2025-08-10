import torch
import random
from typing import Tuple
from abc import ABC, abstractmethod


class Paster:
    def __init__(self, tri_size, tar_size, img_size, h_ratio=0, w_ratio=0, dynamic=False):
        if isinstance(tri_size, int):
            tri_size = (tri_size, tri_size)
        if isinstance(tar_size, int):
            tar_size = (tar_size, tar_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.s0 = tri_size
        self.s1 = tar_size
        self.s2 = img_size
        assert self.s1[0] >= self.s0[0] and self.s1[1] >= self.s0[1], "[ERROR] Target size < Trigger size!"

        self.size = (self.s2[0] - self.s1[0], self.s2[1] - self.s1[1])
        self.delta_s = ((self.s1[0] - self.s0[0]) // 2, (self.s1[1] - self.s0[1]) // 2)
        self.fixed_h = int(h_ratio * self.size[0])
        self.fixed_w = int(w_ratio * self.size[1])
        self.random_position = dynamic

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        mx_t: torch.Tensor = None,
        my_t: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if y.ndim == 3:
            m = torch.zeros_like(y)[0:1]
        elif y.ndim == 4:
            m = torch.zeros_like(y)[:, 0:1]
        else:
            raise NotImplementedError()

        if self.random_position:
            h1 = random.randint(0, self.size[0])
            w1 = random.randint(0, self.size[1])
        else:
            h1 = self.fixed_h
            w1 = self.fixed_w
        h0 = h1 + self.delta_s[0]
        w0 = w1 + self.delta_s[1]

        n = x - y
        x_base = y.clone()
        if mx_t is not None:
            x_t = torch.where(mx_t, x_t, x_base[..., h0 : h0 + self.s0[0], w0 : w0 + self.s0[1]])
        x_base[..., h0 : h0 + self.s0[0], w0 : w0 + self.s0[1]] = x_t
        x = torch.clamp(x_base + n, 0, 1)

        if my_t is not None:
            y_t = torch.where(my_t, y_t, y[..., h1 : h1 + self.s1[0], w1 : w1 + self.s1[1]])
        y[..., h1 : h1 + self.s1[0], w1 : w1 + self.s1[1]] = y_t
        m[..., h1 : h1 + self.s1[0], w1 : w1 + self.s1[1]] = 1 if my_t is None else my_t

        return x, y, m


class BaseBackdoor(ABC):
    def __init__(self, paster: Paster):
        self.paster = paster

    @abstractmethod
    def get_pasted(self):
        pass

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
            x_t, y_t, mx_t, my_t = self.get_pasted()
            x, y, m = self.paster(x, y, x_t, y_t, mx_t, my_t)
        return x, y, m
