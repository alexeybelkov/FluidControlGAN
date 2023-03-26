from typing import Sequence, Union
import torch
from torch.nn import functional as F
import numpy as np


def energy(vel: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(vel.pow(2).sum(1, keepdim=True), 16, 16)


def augment_energy(ke, den):
    std = ke.std((-2, -1), keepdim=True)
    return ke + F.avg_pool2d((torch.randn_like(den) / max(10, std.max())), 16, 16)


def augment_velocity(vel):
    vel = F.avg_pool2d(vel, 16, 16)
    eps = torch.randn_like(vel) / 100
    return torch.relu(F.interpolate(vel + eps, size=(256, 256), 
                                    mode='bicubic', align_corners=True))


def curl(x):
    dy = x[:, :, 1:] - x[:, :, :-1]
    dx = -x[..., 1:] + x[..., :-1]
    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
    return torch.cat([dy,dx], dim=1)


def flatten(*tensors: Sequence[torch.Tensor], BATCH_SIZE: int) -> torch.Tensor:
    return torch.cat([t.reshape(BATCH_SIZE, -1) for t in tensors], dim=1)


def get_flags(x: torch.Tensor, p: float, size: Union[Sequence, int]) -> torch.Tensor:
    mask = torch.tensor(np.random.binomial(1, 0.5, size=size), device=device)
    return (1.0 - mask) * x - mask * torch.ones_like(x)

    
def moving_avg(x, y, alpha: float):
    return (1.0 - alpha) * x + alpha * y
