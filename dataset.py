import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm


#!g1.1
class NoObsDataset(torch.utils.data.Dataset):
    
    def __init__(self, sims_pth: List[Union[str, Path]],transforms: Optional[Callable] = None):
        super().__init__()
        self._transforms = transforms 
        self.sims_pth = sims_pth
        self.density = []
        self.velocity = []
        self.s_dict = {}
        
        for s in tqdm(sims_pth):
            p = next(os.walk(s))
            for f in p[-1]:
                if 'npz' == f[-3:]:
                    if 'density' in f:
                        self.density.append(f'{p[0]}/{f}')
                    else:
                        self.velocity.append(f'{p[0]}/{f}')
                elif 'json' in f:
                    with open(f'{p[0]}/{f}', 'r') as f:
                        loaded = json.load(f)
                        self.s_dict[p[0]] = np.array([float(loaded['bnds']), float(loaded['buoyFac'])])
                            
                            
        assert len(self.density) == len(self.velocity)
        
        self.density.sort()
        self.velocity.sort()
                        
    @property
    def transforms(self):
        return self._transforms
    
    @transforms.setter
    def transforms(self, transforms: Callable):
        self._transforms = transforms
        
    def __len__(self):
        return len(self.density)
    
    def __getitem__(self, index: int):
        den_pth = self.density[index]
        den = np.ascontiguousarray(np.load(den_pth)['arr_0'][0, ::-1])
        vel = np.ascontiguousarray(np.load(self.velocity[index])['arr_0'][0, ::-1, :, :-1])
        i = den_pth.find('v0')
        s_pth = den_pth[:i + 12]
        s = self.s_dict[s_pth]
        
        if self._transforms is not None:
            den = self._transforms(den)
            vel = self._transforms(vel)
        return den, vel, torch.from_numpy(s)
