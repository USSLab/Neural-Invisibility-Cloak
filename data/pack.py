import torch


__all__ = ["pack_rgb", "unpack_rgb", "pack_raw", "unpack_raw"]


def pack_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x.unsqueeze_(0)
    b, c, h, w = x.shape
    assert c == 3
    y = torch.zeros((b, 12, h // 2, w // 2), dtype=x.dtype, device=x.device)
    y[:, 0:3] = x[..., 0::2, 0::2]
    y[:, 3:6] = x[..., 1::2, 0::2]
    y[:, 6:9] = x[..., 0::2, 1::2]
    y[:, 9:12] = x[..., 1::2, 1::2]
    return y


def unpack_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x.unsqueeze_(0)
    b, c, h, w = x.shape
    assert c == 12
    y = torch.zeros((b, 3, h * 2, w * 2), dtype=x.dtype, device=x.device)
    y[..., 0::2, 0::2] = x[:, 0:3]
    y[..., 1::2, 0::2] = x[:, 3:6]
    y[..., 0::2, 1::2] = x[:, 6:9]
    y[..., 1::2, 1::2] = x[:, 9:12]
    return y


def pack_raw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x.unsqueeze_(0)
    if x.ndim == 3:
        x.unsqueeze_(1)
    b, c, h, w = x.shape
    assert c == 1
    y = torch.zeros((b, 4, h // 2, w // 2), dtype=x.dtype, device=x.device)
    y[:, 0:1] = x[..., 0::2, 0::2]
    y[:, 1:2] = x[..., 0::2, 1::2]
    y[:, 2:3] = x[..., 1::2, 1::2]
    y[:, 3:4] = x[..., 1::2, 0::2]
    return y


def unpack_raw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x.unsqueeze_(0)
    b, c, h, w = x.shape
    assert c == 4
    y = torch.zeros((b, 1, h * 2, w * 2), dtype=x.dtype, device=x.device)
    y[..., 0::2, 0::2] = x[:, 0:1]
    y[..., 0::2, 1::2] = x[:, 1:2]
    y[..., 1::2, 1::2] = x[:, 2:3]
    y[..., 1::2, 0::2] = x[:, 3:4]
    return y
