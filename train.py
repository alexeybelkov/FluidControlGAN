from typing import Callable, Mapping

import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from ops import *


def train_step(G: nn.Module, D: nn.Module, d: torch.Tensor, u: torch.Tensor, 
               s: torch.Tensor, w: torch.Tensor, ke: torch.Tensor, optG: Callable, optD: Callable,
               BATCH_SIZE: int, l_adv: float = 0.2, l_l1: float = 1.0) -> Mapping[str, torch.Tensor]:
    opt_inputs = []
    optG.zero_grad(True)
    g_out, g_s, g_w, g_ke = G(d, *opt_inputs)
    
    g_u = curl(g_out)
    
    optD.zero_grad(True)
    
    dt_d, dt_s, dt_w, dt_ke = D(u)
    df_d, df_s, df_w, df_ke = D(g_u.detach(), g_s.detach(), g_w.detach(), g_ke.detach())
    
    s_r, w_r, ke_r = s.reshape(BATCH_SIZE, -1), w.reshape(BATCH_SIZE, - 1), ke.reshape(BATCH_SIZE, -1)
    
    flatten_den = flatten(d, s_r, w_r, ke_r, BATCH_SIZE=BATCH_SIZE)
    
    flatten_dt_out = flatten(dt_d, dt_s, dt_w, dt_ke, BATCH_SIZE=BATCH_SIZE)
    flatten_df_out = flatten(df_d, df_s, df_w, df_ke, BATCH_SIZE=BATCH_SIZE)
    
    D_p_loss = F.mse_loss(flatten_dt_out, flatten_den)
    D_n_loss = F.mse_loss(flatten_df_out, flatten_den)
    
    #D_p_poss = F.mse_loss(dt_d, d) + F.mse_loss(dt_s, s) + F.mse_loss(dt_w, w) +  F.mse_loss(dt_ke, ke)
    #D_n_loss = F.mse_loss(df_f, d) + F.mse_loss(df_s, s) + F.mse_loss(df_w, w) + F.mse_loss(df_ke, ke)
    
    k = 0.5
    D_loss = D_p_loss - k * D_n_loss
    D_loss.backward()
    optD.step()
    
    flatten_gd_out = flatten(*D(g_u, g_s, g_w, g_ke), BATCH_SIZE=BATCH_SIZE)
    flatten_vel = flatten(u, s_r, w_r, ke_r, BATCH_SIZE=BATCH_SIZE)
    
    g_out_l1, s_l1, w_l1, ke_l1 = G(d)
    l1_out = flatten(curl(g_out_l1), s_l1, w_l1, ke_l1, BATCH_SIZE=BATCH_SIZE)
    
    G_loss = l_adv * F.mse_loss(flatten_gd_out, flatten_den) + l_l1 * F.l1_loss(l1_out, flatten_vel)
    
    G_loss.backward()
    optG.step()
    
    return {'G_loss': G_loss.detach().cpu().item(),
            'D_p_loss': D_p_loss.detach().cpu().item(),
            'D_n_loss': D_n_loss.detach().cpu().item()}


def train(G, D, optG, optD, train_loader, val_loader, num_epochs, device, BATCH_SIZE):
    losses = {'G_loss': [],
              'D_p_loss': [],
              'D_n_loss': []}
    for epoch in range(num_epochs):
        for k in losses.keys():
            losses[k].append([])
        G.train(), D.train()
        for den, vel, s in tqdm(train_loader):
            den, vel, s = den.to(device), vel.to(device), s.to(device),
            ke = energy(vel)
#             if np.random.binomial(1, 0.2):
#                 ke = augment_energy(ke, den)
#             if np.random.binomial(1, 0.2):
#                 ke = augment_energy(ke, den)
            
            vdx = F.pad((vel[:, 1, :, 1:] - vel[:, 1, :, :-1])[:, None], (0, 1, 0, 0), mode='reflect')
            udy = F.pad((vel[:, 0, 1:] - vel[:, 0, :-1])[:, None], (0, 0, 0, 1), mode='reflect')

            #w = udx - vdy
            w = vdx - udy
             
            ke_mask = torch.tensor(np.random.binomial(1, 0.5, size=(BATCH_SIZE, 1, 1, 1)), device=device)
            w_mask = torch.tensor(np.random.binomial(1, 0.5, size=(BATCH_SIZE, 1, 1, 1)), device=device)
            s_mask = torch.tensor(np.random.binomial(1, 0.5, size=(BATCH_SIZE, 1)), device=device)
            
            ke = ke * (1.0 - ke_mask) - ke_mask * torch.ones_like(ke)
            w = w * (1.0 - w_mask) - w_mask * torch.ones_like(w)
            s = s * (1.0 - s_mask) - s_mask * torch.ones_like(s)
            
            loss = train_step(G, D, den, vel, s, w, ke, optG, optD, BATCH_SIZE)
            
            for k in loss.keys():
                losses[k][-1].append(loss[k])
        for k in losses.keys():
            losses[k][-1] = np.mean(losses[k][-1])
        
        print(f'Epoch:{ epoch}')
        
        for k in loss.keys():
            print(f'{k}: {losses[k][-1]}')
        
        G.eval()
        with torch.no_grad():
            for den, vel, s in tqdm(val_loader):
                den, vel, s = den.to(device), vel.to(device), s.to(device),
                g_out, *_ = G(den)
                u = curl(g_out).cpu().numpy()
                if np.random.binomial(1, p=0.05):
                    plt.imshow(u.pow(2).sum(1)[0])
