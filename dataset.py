import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from tqdm import tqdm


class NoObsDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path: Union[str, Path], transforms: Optional[Callable] = None):
        self._transforms = transforms
        self.density = []
        self.velocity = []
        for p in tqdm(os.walk(data_path)):
            if 'sim' in p[0]:
                for f in p[-1]:
                    if 'npz' == f[-3:]:
                        if 'density' in f:
                            self.density.append(f'{p[0]}/{f}')
                        else:
                            self.velocity.append(f'{p[0]}/{f}')
                            
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
        den = np.load(self.density[index])['arr_0'][0].transpose(1, 0, 2)
        vel = np.load(self.velocity[index])['arr_0'][0,..., :-1].transpose(1, 0, 2)
        if self._transforms is not None:
            den = self._transforms(den)
            vel = self._transforms(vel)
        return den, vel
    