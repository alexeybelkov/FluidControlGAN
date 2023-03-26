from typing import Callable, Mapping, Optional, Tuple, Iterable

import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from ops import *


def loss(f, pred, gt):
    return sum(f(p, t) for p, t in zip(pred, gt))


def train_step(G: nn.Module, D: nn.Module, d: torch.Tensor, u: torch.Tensor, 
               s: Tuple[torch.Tensor], w: Tuple[torch.Tensor], ke: Tuple[torch.Tensor], 
               augm_inputs: Optional[Sequence[Tuple[torch.Tensor]]], 
               optG: Callable, optD: Callable, BATCH_SIZE: int, k: float, 
               l_adv: float = 0.2, l_l1: float = 1.0) -> Mapping[str, torch.Tensor]:
    
    optG.zero_grad(True)
    
    (s_, s), (w_, w), (ke_, ke) = s, w, ke
    
    g_out, g_s, g_w, g_ke = G(d, s_, w_, ke_)
    g_u = curl(g_out)
    
    optD.zero_grad(True)
    
    gt_den = [d, s, w, ke]
    
    D_p_loss = loss(F.l1_loss, D(u), gt_den)
    D_n_loss = loss(F.l1_loss, D(g_u.detach(), g_s.detach(), g_w.detach(), g_ke.detach()), gt_den)
    
    D_loss = D_p_loss - k * D_n_loss
    D_loss.backward()
    optD.step()
    
    g_out_l1, s_l1, w_l1, ke_l1 = G(d)
    g_u_l1 = curl(g_out_l1)
    
    G_adv_loss = l_adv * loss(F.l1_loss, D(g_u, g_s, g_w, g_ke), gt_den)
    l1_loss = l_l1 * loss(F.l1_loss, [g_u_l1, s_l1, w_l1, ke_l1], [u, s, w, ke])
    
    if augm_inputs is not None:
        u_augm, (w_augm_, w_augm), (ke_augm_, ke_augm) = augm_inputs
        g_out_a, gs_augm, gw_augm, gke_augm = G(d, s, w_augm_, ke_augm_)
        gu_augm = curl(g_out_a)
        mod_loss = 0.6 * loss(F.l1_loss, D(gu_augm, gs_augm, gw_augm, gke_augm), [d, s, w_augm, ke_augm])
        G_loss = G_adv_loss + l1_loss + mod_loss
    else:
        G_loss = G_adv_loss + l1_loss
    
    G_loss.backward()
    optG.step()
    
    return {'G_loss': G_loss.detach().cpu().item(),
            'D_p_loss': D_p_loss.detach().cpu().item(),
            'D_n_loss': D_n_loss.detach().cpu().item()}


def train(G: nn.Module, D: nn.Module, optG: Callable, optD: Callable, train_loader: Iterable, 
          val_loader: Iterable, num_epochs: int, device: str, BATCH_SIZE: int, alpha: float = 0.3, beta: float = 0.3):
    losses = {'G_loss': [],
              'D_p_loss': [],
              'D_n_loss': []}
    k = gamma = 1e-4
    D_p_loss = D_n_loss = 0.0
    for epoch in range(num_epochs):
        for key in losses.keys():
            losses[key].append([])
        G.train(), D.train()
        for den, vel, s in tqdm(train_loader):
            den, vel, s = den.to(device), vel.to(device), s.to(device),
            augm_inputs = None
            if np.random.binomial(1, 0.25):
                vel_augm = torch.abs(augment_velocity(vel))
                ke_augm = energy(vel_augm)
                vdx = F.pad((vel_augm[:, 1, :, 1:] - vel_augm[:, 1, :, :-1])[:, None], (0, 1, 0, 0), mode='replicate')
                udy = F.pad((vel_augm[:, 0, 1:] - vel_augm[:, 0, :-1])[:, None], (0, 0, 0, 1), mode='replicate')
                w_augm = vdx - udy
                
                ke_augm_ = get_flags(ke_augm, 0.5, (BATCH_SIZE, 1, 1, 1))
                w_augm_ = get_flags(w_augm, 0.5, (BATCH_SIZE, 1, 1, 1))
                
                augm_inputs = [vel_augm, (w_augm_, w_augm), (ke_augm_, ke_augm)]
                
            ke = energy(vel)
            vdx = F.pad((vel[:, 1, :, 1:] - vel[:, 1, :, :-1])[:, None], (0, 1, 0, 0), mode='replicate')
            udy = F.pad((vel[:, 0, 1:] - vel[:, 0, :-1])[:, None], (0, 0, 0, 1), mode='replicate')
            
            w = vdx - udy
            
            ke_ = get_flags(ke, 0.5, (BATCH_SIZE, 1, 1, 1))
            w_ = get_flags(w, 0.5, (BATCH_SIZE, 1, 1, 1))
            s_ = get_flags(s, 0.5, (BATCH_SIZE, 1))
            
            loss = train_step(G, D, den, vel, (s_, s), (w_, w), (ke_, ke), augm_inputs, optG, optD, BATCH_SIZE, k)
            
            D_p_loss = moving_avg(D_p_loss, loss['D_p_loss'], alpha)
            D_n_loss = moving_avg(D_n_loss, loss['D_n_loss'], alpha)
            
            gamma = D_p_loss / D_n_loss
            k = moving_avg(k, gamma, beta)
            
            for key in loss.keys():
                losses[key][-1].append(loss[key])
        for key in losses.keys():
            losses[key][-1] = np.mean(losses[key][-1])
        
        print(f'Epoch:{ epoch}')
        
        for key in losses.keys():
            print(f'{key}: {losses[key][-1]}')
        
        G.eval()
        
        with torch.no_grad():
            for i, (den, vel, s) in tqdm(enumerate(val_loader)):
                if np.random.binomial(1, p=0.005):
                    den, vel, s = den.to(device), vel.to(device), s.to(device),
                    g_out, s, w, ke = G(den)
                    u = curl(g_out).cpu().numpy()
                    plt.imshow(den[0][0].cpu().numpy(), cmap='gray')
                    plt.title(f'den{i}_{epoch}')
                    plt.savefig(f'den/den{i}_{epoch}.png')
                    plt.clf()
                    plt.imshow(vel.pow(2).sum(1)[0].cpu().numpy(), cmap='gray')
                    plt.title(f'gt_vel{i}_{epoch}')
                    plt.savefig(f'gt_vel/gt_vel{i}_{epoch}.png')
                    plt.clf()
                    plt.imshow(u[0][0], cmap='gray')
                    plt.title(f'vel{i}_0_{epoch}')
                    plt.savefig(f'vel/vel{i}_0_{epoch}.png')
                    plt.clf()
                    plt.imshow(u[0][1], cmap='gray')
                    plt.title(f'vel{i}_1_{epoch}')
                    plt.savefig(f'vel/vel{i}_1_{epoch}.png')
                    plt.clf()
                    plt.imshow(np.sqrt(u[0][1] ** 2 + u[0][0] ** 2), cmap='gray')
                    plt.title(f'vel{i}_{epoch}')
                    plt.savefig(f'vel/vel{i}_{epoch}.png')
                    plt.clf()
                    plt.imshow(w[0, 0].cpu().numpy(), cmap='gray')
                    plt.title(f'w{i}_{epoch}')
                    plt.savefig(f'vort/w{i}_{epoch}.png')
                    plt.clf()
        torch.save(G, 'G.pth'), torch.save(D, 'D.pth')

