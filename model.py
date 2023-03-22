from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class ResBlock(nn.Module):
    
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(out_c),
                                    nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(out_c, out_c, 3, padding=1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(out_c))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1(x)
        y = self.conv_2(y)
        return x + y
    
    
class SubNetS(nn.Module):
    
    def __init__(self, in_c: int):
        super().__init__()
#         self.conv_in = nn.Sequential(nn.Conv2d(in_c, 16, 7),
#                                     nn.ReLU())
        self.pool = nn.AvgPool2d(8, 8)
        self.flatten_conv = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, padding_mode='reflect'), 
                                          nn.AvgPool2d(2, 2))
        self.s_out = nn.Sequential(nn.Flatten(), nn.Linear(16 * 4 * 4, 2))
        self.s_in = nn.Linear(4, 4 * 8 * 8)
        self.conv_out = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1, padding_mode='reflect'),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(8, 16, 3, padding=1, padding_mode='reflect'),
                                      nn.LeakyReLU(0.2))
        self.unpool = nn.Sequential(nn.Upsample((64, 64), mode='bilinear', align_corners=False), 
                                    nn.Conv2d(16, 16, 1))
        
    def forward(self, x: torch.Tensor, s: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #x = self.conv_in(x)
        x = self.pool(x)
        x = self.flatten_conv(x)
        s_out = self.s_out(x)
        if s is None:
            s = -torch.ones_like(s_out)
            s.requires_grad = False
        x = torch.cat([s_out, s], dim=1)
        x = self.s_in(x)
        x = x.reshape(-1, 4, 8, 8)
        x = self.conv_out(x)
        x = self.unpool(x)
        return x, s_out
            
        
class SubNetKE(nn.Module):
    
    def __init__(self, in_c: int):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_c, 8, 3, padding=1, padding_mode='reflect'),
                                    nn.AvgPool2d(2, 2), 
                                    nn.InstanceNorm2d(8),
                                    nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(8, 4, 3, padding=1, padding_mode='reflect'),
                                    nn.AvgPool2d(2, 2),
                                    nn.InstanceNorm2d(4))
        self.ke_out_conv = nn.Conv2d(4, 1, 1)
        self.ke_in_conv = nn.Conv2d(2, 2, 3, padding=1, padding_mode='reflect')
        self.conv_5 = nn.Conv2d(2, 2, 3, padding=1, padding_mode='reflect')
        self.unpool = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.Conv2d(2, 1, 1))
        
    def forward(self, x: torch.Tensor, ke: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_1(x)
        x = self.conv_2(x)
        ke_out = self.ke_out_conv(x)
        if ke is None:
            ke = -torch.ones_like(ke_out)
            ke.requires_grad = False
        x = torch.cat([ke_out, ke], dim=1) if x.ndim == 4 else torch.cat([ke_out, ke], dim=0)
        x = self.ke_in_conv(x)
        x = self.conv_5(x)
        return self.unpool(x), ke_out
        
        
class SubNetW(nn.Module):
    def __init__(self, in_c: int):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
                                 nn.Conv2d(in_c, 32, 3, padding=1, padding_mode='reflect'),
                                 nn.InstanceNorm2d(32),
                                 nn.ReLU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
                                 nn.Conv2d(32, 16, 3, padding=1, padding_mode='reflect'),
                                 nn.InstanceNorm2d(16),
                                 nn.ReLU())
        self.w_out = nn.Conv2d(16, 1, 7, padding='same')
        self.w_in = nn.Sequential(nn.Conv2d(2, 16, 5, padding=1, padding_mode='reflect'),
                                  nn.AvgPool2d(2, 2),
                                  nn.InstanceNorm2d(16),
                                  nn.ReLU())
        
        self.down = nn.Sequential(nn.Conv2d(16, 24, 5, padding=1, padding_mode='reflect'),
                                  nn.AvgPool2d(2, 2),
                                  nn.InstanceNorm2d(24),
                                  nn.ReLU())
        
        self.conv_out = nn.Conv2d(24, 32, 3, padding=1, padding_mode='reflect')
        
    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.up1(x)
        x = self.up2(x)
        w_out = self.w_out(x)
        if w is None:
            w = -torch.ones_like(w_out)
            w.requires_grad = False
        x = torch.cat([w_out, w], dim=1) if x.ndim == 4 else torch.cat([w_out, w], dim=0)
        x = self.w_in(x)
        x = self.down(x)
        x = self.conv_out(x)
        return x, w_out
    
    
    
class UNet(nn.Module):
    def __init__(self, in_c: int, out_c: int, s_subnet: nn.Module, 
                 w_subnet: nn.Module, ke_subnet: nn.Module, obstacles: bool = False):
        super().__init__()
        self.obstacles = obstacles
        self.down_1 = nn.Sequential(nn.Conv2d(in_c + obstacles, 16, 3, padding='same', padding_mode='reflect'),
                                    nn.InstanceNorm2d(16),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, padding='same'),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU())
        self.down_2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding='same', padding_mode='reflect'),
                                    nn.AvgPool2d(2, 2),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU())
        
        self.down_3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding='same', padding_mode='reflect'),
                                    nn.AvgPool2d(2, 2),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU())
        self.resblock_1 = ResBlock(128, 128)
        self.resblock_2 = ResBlock(128, 128)
        self.resblock_3 = ResBlock(128, 128)
        self.resblock_4 = ResBlock(145, 145)
        self.resblock_5 = ResBlock(145, 145)
        self.resblock_6 = ResBlock(145, 145)
        
        self.subnet_conv = nn.Sequential(nn.Conv2d(128, 16, 3, padding='same', padding_mode='reflect'), 
                                         nn.ReLU())
        
        self.up_1 = nn.Sequential(nn.Conv2d(177, 128, 3, padding='same', padding_mode='reflect'),
                                  nn.InstanceNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, padding='same', padding_mode='reflect'),
                                  nn.InstanceNorm2d(128),
                                  nn.ReLU(),
                                  nn.Upsample((128, 128), mode='bilinear', align_corners=False))
        self.up_2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding='same', padding_mode='reflect'),
                                  nn.InstanceNorm2d(64),
                                  nn.ReLU(),
                                  nn.Upsample((256, 256), mode='bilinear', align_corners=False))
            
        self.up_3 = nn.Sequential(nn.Conv2d(64, 32, 3, padding='same', padding_mode='reflect'),
                                  nn.ReLU())
                        
        self.conv_out = nn.Conv2d(32, out_c, 3, padding='same', padding_mode='reflect')
        
        self.s_subnet = s_subnet
        self.ke_subnet = ke_subnet
        self.w_subnet = w_subnet
        
    def forward(self, x: torch.Tensor, s: Optional[torch.Tensor] = None, 
                w: Optional[torch.Tensor] = None, ke: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, 
                                                                                              torch.Tensor, torch.Tensor]:
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        
        x = self.resblock_1(x) 
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        
        y = self.subnet_conv(x)
        
        
        y_s, s_out = self.s_subnet(y, s) # 16 x 8 x 8
        y_ke, ke_out = self.ke_subnet(y, ke) # 1 x 16 x 16
        
        x = torch.cat([F.interpolate(y_s, size=(64, 64), mode='bilinear', align_corners=False), 
                       F.interpolate(y_ke, size=(64, 64), mode='bilinear', align_corners=False), x], dim=1)
        
        x = self.resblock_4(x)
        x = self.resblock_5(x)
        x = self.resblock_6(x)

        y, w_out = self.w_subnet(x, w)
        
        
        x = torch.cat([F.interpolate(y, size=(64, 64), mode='bilinear', align_corners=False), x], dim=1)
        
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        
        x = self.conv_out(x)
        
        return x, s_out, w_out, ke_out
    