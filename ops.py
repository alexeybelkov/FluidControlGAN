from typing import Sequence
import torch
from torch.nn import functional as F

def energy(vel: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(vel.pow(2).sum(1, keepdim=True), 16, 16)


def augment_energy(ke, den):
    std = ke.std((-2, -1), keepdim=True)
    return ke + F.avg_pool2d((torch.randn_like(den) / max(10, std.max())), 16, 16)


def augment_velocity(vel):
    eps = torch.randn_like(vel) / max(10, vel.std())
    return vel + F.interpolate(F.avg_pool2d(eps, 16, 16), size=(256, 256), mode='bilinear', align_corners=True)


def curl(x):
    dy = x[:, :, 1:] - x[:, :, :-1]
    dx = -x[..., 1:] + x[..., :-1]
    dy = torch.cat([dy, dy[:, :, -1].unsqueeze(2)], dim=2)
    dx = torch.cat([dx, dx[..., -1].unsqueeze(3)], dim=3)
    return torch.cat([dy,dx], dim=1)


def flatten(*tensors: Sequence[torch.Tensor], BATCH_SIZE: int) -> torch.Tensor:
    return torch.cat([t.reshape(BATCH_SIZE, -1) for t in tensors], dim=1)