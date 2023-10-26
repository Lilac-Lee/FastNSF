"""optimize over a network structure."""

import argparse
import logging
import os, glob
import time
import math
from collections import defaultdict, namedtuple
from itertools import accumulate
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import FastGeodis
import open3d as o3d
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ANCHOR: visualization as in the paper
DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

def make_colorwheel(transitions: tuple=DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0])
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float]=None,
    background: Optional[str]="bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(f"background should be one the following: {valid_backgrounds}, not {background}.")
    wheel = make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int32)] * (1 - angle_fractional) + wheel[angle_ceil.astype(np.int32)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        'ColorizationArgs', ['move_hue_valid_radius', 'move_hue_oversized_radius', 'invalid_color']
    )
    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)
    def move_hue_on_S_axis(hues, factors):
        return 255. - np.expand_dims(factors, -1) * (255. - hues)
    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float32)
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float32))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)


def custom_draw_geometry_with_key_callback(pcds):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([76/255, 86/255, 106/255])
        return False
    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            'render_option.json')
        return False
    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
        

# ANCHOR: metrics computation, follow NSFP metrics....
def scene_flow_metrics(pred, labels):
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, 2)).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 2)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / labels.norm(dim=2, keepdim=True)
    unit_pred = pred / pred.norm(dim=2, keepdim=True)
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(2).clamp(min=-1+eps, max=1-eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    angle_error = torch.acos(dot_product).mean().item()

    return EPE3D, acc3d_strict, acc3d_relax, outlier, angle_error


# ANCHOR: timer!
class Timers(object):
    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def print(self, key=None):
        if key is None:
            for k, v in self.timers.items():
                print("Average time for {:}: {:}".format(k, v.avg()))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))

    def get_avg(self, key):
        return self.timers[key].avg()
    
    
class Timer(object):
    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def total(self):
        return self.total_time

    def avg(self):
        return self.total_time / float(self.calls)

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.


# ANCHOR: early stopping strategy
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class DT:
    def __init__(self, pts, pmin, pmax, grid_factor, device='cuda:0'):
        self.device = device
        self.grid_factor = grid_factor
        
        sample_x = ((pmax[0] - pmin[0]) * grid_factor).ceil().int() + 2
        sample_y = ((pmax[1] - pmin[1]) * grid_factor).ceil().int() + 2
        sample_z = ((pmax[2] - pmin[2]) * grid_factor).ceil().int() + 2
        
        self.Vx = torch.linspace(0, sample_x, sample_x+1, device=self.device)[:-1] / grid_factor + pmin[0]
        self.Vy = torch.linspace(0, sample_y, sample_y+1, device=self.device)[:-1] / grid_factor + pmin[1]
        self.Vz = torch.linspace(0, sample_z, sample_z+1, device=self.device)[:-1] / grid_factor + pmin[2]
        
        # NOTE: build a binary image first, with 0-value occuppied points
        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = torch.stack([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1).float().squeeze()
        H, W, D, _ = self.grid.size()
        pts_mask = torch.ones(H, W, D, device=device)
        self.pts_sample_idx_x = ((pts[:,0:1] - self.Vx[0]) * self.grid_factor).round()
        self.pts_sample_idx_y = ((pts[:,1:2] - self.Vy[0]) * self.grid_factor).round()
        self.pts_sample_idx_z = ((pts[:,2:3] - self.Vz[0]) * self.grid_factor).round()
        pts_mask[self.pts_sample_idx_x.long(), self.pts_sample_idx_y.long(), self.pts_sample_idx_z.long()] = 0.
        
        iterations = 1
        image_pts = torch.zeros(H, W, D, device=device).unsqueeze(0).unsqueeze(0)
        pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)
        self.D = FastGeodis.generalised_geodesic3d(
            image_pts, pts_mask, [1./self.grid_factor, 1./self.grid_factor, 1./self.grid_factor], 1e10, 0.0, iterations
        ).squeeze()
            
    def torch_bilinear_distance(self, Y):
        H, W, D = self.D.size()
        target = self.D[None, None, ...]
        
        sample_x = ((Y[:,0:1] - self.Vx[0]) * self.grid_factor).clip(0, H-1)
        sample_y = ((Y[:,1:2] - self.Vy[0]) * self.grid_factor).clip(0, W-1)
        sample_z = ((Y[:,2:3] - self.Vz[0]) * self.grid_factor).clip(0, D-1)
        
        sample = torch.cat([sample_x, sample_y, sample_z], -1)
        
        # NOTE: normalize samples to [-1, 1]
        sample = 2 * sample
        sample[...,0] = sample[...,0] / (H-1)
        sample[...,1] = sample[...,1] / (W-1)
        sample[...,2] = sample[...,2] / (D-1)
        sample = sample -1
        
        sample_ = torch.cat([sample[...,2:3], sample[...,1:2], sample[...,0:1]], -1)
        
        # NOTE: reshape to match 5D volumetric input
        dist = F.grid_sample(target, sample_.view(1,-1,1,1,3), mode="bilinear", align_corners=True).view(-1)
        return dist


class encoding_func_1D:
    def __init__(self, name, param=None, device='cpu'):
        self.name = name
        
        if name == 'none':
            self.dim = 1
        else:
            self.dim = param[1]
            if name == 'gaussian':
                self.dic = (torch.linspace(0., param[1], steps=param[1]+1, device=device)[:-1]/param[3]+param[2]).reshape(1,-1)
                self.sig = torch.tensor(param[0]).to(device)
            else:
                print('Undifined encoding.')
                
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'gaussian':
            emb = (-0.5*(x-self.dic)**2/(self.sig**2)).exp()
            emb = emb/(emb.norm(dim=1).max())
            return emb
        
        
class blending_func_3D:
    def __init__(self, encoding_func, dim=256, dim_xyz=[], xyz_min_int=[], indexing=True, device='cuda:0'):
        self.name, self.sig = encoding_func
        self.dim = dim
        self.dim_xyz = dim_xyz
        self.xyz_min_int = xyz_min_int
        self.indexing = indexing
        self.device = device
        
        if self.name == 'gaussian':
            self.D = lambda x1,x2: (-0.25*((x1-x2))**2/(self.sig**2)).exp().to(device)   # NOTE: do not need to divide self.dim

    def __call__(self, inp):
        x = inp[:,0:1]
        y = inp[:,1:2]
        z = inp[:,2:3]
        
        xmin = torch.floor(x*self.dim) / self.dim
        ymin = torch.floor(y*self.dim) / self.dim
        zmin = torch.floor(z*self.dim) / self.dim
        
        if self.name=='gaussian':
            d0 = self.D(torch.tensor([0.]).to(self.device),torch.tensor([0.]).to(self.device))
            dd = self.D(torch.tensor([0.]).to(self.device),torch.tensor([1./self.dim]).to(self.device))
            ff = d0**2-dd**2
            
            xda = self.D(xmin,x)
            xdb = self.D(xmin+1./self.dim,x)
            
            xa = (xda*d0-xdb*dd)/ff
            xb = (xdb*d0-xda*dd)/ff
            
            yda = self.D(ymin,y)
            ydb = self.D(ymin+1./self.dim,y)
            
            ya = (yda*d0-ydb*dd)/ff
            yb = (ydb*d0-yda*dd)/ff
            
            zda = self.D(zmin,z)
            zdb = self.D(zmin+1./self.dim,z)
            
            za = (zda*d0-zdb*dd)/ff
            zb = (zdb*d0-zda*dd)/ff

            Ns = x.shape[0]
            Nx = self.dim_xyz[0]-1
            Ny = self.dim_xyz[1]-1
            Nz = self.dim_xyz[2]-1
            
            xmin_grid, ymin_grid, zmin_grid = self.xyz_min_int
            x_grid = (xmin - xmin_grid) * self.dim
            y_grid = (ymin - ymin_grid) * self.dim
            z_grid = (zmin - zmin_grid) * self.dim
            
            xyz_01 = x_grid*Ny*Nz+y_grid*Nz+z_grid
            xyz_02 = x_grid*Ny*Nz+(y_grid+1)*Nz+z_grid
            xyz_03 = (x_grid+1)*Ny*Nz+y_grid*Nz+z_grid
            xyz_04 = (x_grid+1)*Ny*Nz+(y_grid+1)*Nz+z_grid
            if self.indexing:
                c = torch.cat([xa*ya*za, xa*ya*zb, xa*yb*za, xa*yb*zb, xb*ya*za, xb*ya*zb, xb*yb*za, xb*yb*zb],1)
                y = torch.cat([xyz_01, xyz_01+1, xyz_02, xyz_02+1,
                                    xyz_03, xyz_03+1, xyz_04, xyz_04+1],1).long()
                return [y,c]


class Indexing_Blend_Kron3_MLP(nn.Module):
    def __init__(self, input_dim=[], width0=3, scaling=1.0, epsilon=1e-5):
        super(Indexing_Blend_Kron3_MLP, self).__init__()
        self.scaling = scaling
        self.e = epsilon
        
        self.first = nn.ParameterDict({
                'weight': nn.Parameter(2/math.sqrt(width0)*torch.rand(width0, input_dim[0], input_dim[1], input_dim[2])-1/math.sqrt(width0))})
        
    def forward(self, B, xyz):
        x, y, z = xyz
        x = x @ self.first.weight.transpose(1,2)
        x = y @ x.transpose(1,2)
        x = x @ z.transpose(0,1)
        
        # NOTE: add TV regularizer?
        tv = torch.mean(torch.sqrt(self.e +
            (x[:, :-1, :-1, 1:] - x[:, :-1, :-1, :-1]) ** 2 +
            (x[:, :-1, 1:, :-1] - x[:, :-1, :-1, :-1]) ** 2 +
            (x[:, 1:, :-1, :-1] - x[:, :-1, :-1, :-1]) ** 2).sum(dim=0))
        reg_scaled = tv * self.scaling
            
        x = x.flatten(1,3).transpose(0,1)

        x = (x[B[0]]*(B[1].unsqueeze(-1))).sum(1)

        return x, reg_scaled
    
        
def solver(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    flow: torch.Tensor,
    options: argparse.Namespace,
    max_iters: int,
):

    if options.time:
        timers = Timers()
        timers.tic("solver_timer")
    
    pre_compute_st = time.time()
    solver_time = 0.
    
    total_losses = []
    total_acc_strit = []
    total_iter_time = []
    
    if options.earlystopping:
        early_stopping = EarlyStopping(patience=options.early_patience, min_delta=options.early_min_delta)
      
    # ANCHOR: for complex encoding
    complex_encode_time_st = time.time()
    # NOTE: -1 for xyz min, +1 for xyz max to incorporate possible boundary out-of-range problem.
    #        constraining all flows within [min(pc1 U pc2) - 1 coord, max(pc1 U pc2) + 1 coord].
    pc1_min = torch.min(pc1, 1)[0].squeeze(0)
    pc1_max = torch.max(pc1, 1)[0].squeeze(0)

    pc2_min = torch.min(pc2, 1)[0].squeeze(0)
    pc2_max = torch.max(pc2, 1)[0].squeeze(0)

    xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min<pc2_min, pc1_min, pc2_min) * options.grid_factor-1) / options.grid_factor
    xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max>pc2_max, pc1_max, pc2_max) * options.grid_factor+1) / options.grid_factor
    
    sample_x = ((xmax_int - xmin_int) * options.grid_factor).int() + 1
    sample_y = ((ymax_int - ymin_int) * options.grid_factor).int() + 1
    sample_z = ((zmax_int - zmin_int) * options.grid_factor).int() + 1
    
    inner_encoding_x = encoding_func_1D('gaussian', [options.gauss_sigma, sample_x, xmin_int, options.grid_factor], options.device)
    inner_encoding_y = encoding_func_1D('gaussian', [options.gauss_sigma, sample_y, ymin_int, options.grid_factor], options.device)
    inner_encoding_z = encoding_func_1D('gaussian', [options.gauss_sigma, sample_z, zmin_int, options.grid_factor], options.device)
            
    outer_blending = blending_func_3D(['gaussian', options.gauss_sigma], dim=options.grid_factor, dim_xyz=[sample_x, sample_y, sample_z], 
                                        xyz_min_int = [xmin_int, ymin_int, zmin_int],
                                        indexing=True, device=options.device)
    net = Indexing_Blend_Kron3_MLP(input_dim=[sample_x, sample_y, sample_z], scaling=options.reg_scaling, epsilon=options.epsilon).to(options.device)
    
    dt_start_time = time.time()
    
    xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min<pc2_min, pc1_min, pc2_min) * options.dt_grid_factor-1) / options.dt_grid_factor
    xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max>pc2_max, pc1_max, pc2_max)* options.dt_grid_factor+1) / options.dt_grid_factor
    
    # NOTE: build DT map
    dt = DT(pc2.clone().squeeze(0).to(options.device), (xmin_int, ymin_int, zmin_int), (xmax_int, ymax_int, zmax_int), options.dt_grid_factor, options.device)
    
    dt_time = time.time() - dt_start_time
    
    pc1 = pc1.to(options.device).contiguous()
    pc2 = pc2.to(options.device).contiguous()
    flow = flow.to(options.device).contiguous()
    print(pc1.shape, pc2.shape, flow.shape)
    
    if options.init_weight:
        net.apply(init_weights)
        
    for param in net.parameters():
        param.requires_grad = True
    
    params = net.parameters()
    
    optimizer = torch.optim.Adam(params, lr=options.lr, weight_decay=0)
    
    # NOTE: inner_encoding (Gaussian)
    grid_x = torch.linspace(0, sample_x, sample_x+1, device=options.device)[:-1] / options.grid_factor + xmin_int
    grid_y = torch.linspace(0, sample_y, sample_y+1, device=options.device)[:-1] / options.grid_factor + ymin_int
    grid_z = torch.linspace(0, sample_z, sample_z+1, device=options.device)[:-1] / options.grid_factor + zmin_int
    
    encoded_grid_x = inner_encoding_x(grid_x.reshape(-1,1).to(options.device))
    encoded_grid_y = inner_encoding_y(grid_y.reshape(-1,1).to(options.device))
    encoded_grid_z = inner_encoding_z(grid_z.reshape(-1,1).to(options.device))
    encoded_grid = [encoded_grid_x, encoded_grid_y, encoded_grid_z]
    
    pc1_kron_idx, pc1_kron = outer_blending(pc1.squeeze(0))
    complex_encode_time = time.time() - complex_encode_time_st
    
    pre_compute_time = time.time() - pre_compute_st
    solver_time = solver_time + pre_compute_time
    
    # ANCHOR: initialize best metrics
    best_loss = 1e10
    best_flow = None
    best_epe3d = 1.
    best_acc3d_strict = 0.
    best_acc3d_relax = 0.
    best_angle_error = 1.
    best_outliers = 1.
    best_epoch = 0
    net_time = 0.
    net_backward_time = 0.
    dt_query_time = 0.
    
    for epoch in range(max_iters):
        iter_time_init = time.time()
    
        optimizer.zero_grad()
        
        net_time_st = time.time()
        flow_pred, tv_scaled = net([pc1_kron_idx, pc1_kron], encoded_grid)
        flow_pred = flow_pred.unsqueeze(0)
        net_time = net_time + time.time() - net_time_st
        pc1_deformed = pc1 + flow_pred
        
        dt_query_st = time.time()
        loss_dt = dt.torch_bilinear_distance(pc1_deformed.squeeze(0)).mean()
        dt_query_time = dt_query_time + time.time() - dt_query_st
        
        loss = loss_dt + tv_scaled
        
        net_backward_st = time.time()
        loss.backward()
        optimizer.step()
        net_backward_time = net_backward_time + time.time() - net_backward_st
    
        if options.earlystopping:
            if early_stopping.step(loss):
                break
        
        iter_time = time.time() - iter_time_init
        solver_time = solver_time + iter_time
        
        flow_pred_final = pc1_deformed - pc1
        flow_metrics = flow.clone()
        epe3d, acc3d_strict, acc3d_relax, outlier, angle_error = scene_flow_metrics(flow_pred_final, flow_metrics)

        # ANCHOR: get best metrics
        if loss <= best_loss:
            best_loss = loss.item()
            best_flow = flow_pred_final
            best_epe3d = epe3d
            best_acc3d_strict = acc3d_strict
            best_acc3d_relax = acc3d_relax
            best_angle_error = angle_error
            best_outliers = outlier
            best_epoch = epoch
        
        total_losses.append(loss.item())
        total_acc_strit.append(acc3d_strict)
        total_iter_time.append(time.time()-iter_time_init)
            
        if epoch % 50 == 0:
            logging.info(f"[Ep {epoch}] [Loss: {loss.item():.5f}] "
                        f" Metrics: flow 1 --> flow 2"
                        f" [EPE: {epe3d:.3f}] [Acc strict: {acc3d_strict * 100:.3f}%]"
                        f" [Acc relax: {acc3d_relax * 100:.3f}%] [Angle error (rad): {angle_error:.3f}]"
                        f" [Outl.: {outlier * 100:.3f}%]")
            
    if options.time:
        timers.toc("solver_timer")
        time_avg = timers.get_avg("solver_timer")
        logging.info(timers.print())
        
    # ANCHOR: get the best metrics
    info_dict = {
        'final_flow': best_flow,
        'loss': best_loss,
        'EPE3D': best_epe3d,
        'acc3d_strict': best_acc3d_strict,
        'acc3d_relax': best_acc3d_relax,
        'angle_error': best_angle_error,
        'outlier': best_outliers,
        'time': time_avg,
        'epoch': best_epoch,
        'solver_time': solver_time,
        'pre_compute_time': pre_compute_time,
        'complex_encode_time': complex_encode_time,
    }
    
    info_dict['build_dt_time'] = dt_time
    info_dict['dt_query_time'] = dt_query_time
    info_dict['avg_dt_query_time'] = dt_query_time / epoch
    
    info_dict['network_time'] = net_time
    info_dict['avg_net_time'] = net_time / epoch
    info_dict['net_backward_time'] = net_backward_time
    info_dict['avg_net_backward_time'] = net_backward_time / epoch
    
    # NOTE: visualization
    if options.visualize:
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        ax.plot(total_losses, label="loss")
        ax.legend(fontsize="14")
        ax.set_xlabel("Iteration", fontsize="14")
        ax.set_ylabel("Loss", fontsize="14")
        ax.set_title("Loss vs iterations", fontsize="14")
        plt.show()

        # ANCHOR: new plot style
        # NOTE: GT flow
        pc1_o3d_gt = o3d.geometry.PointCloud()
        colors_flow = flow_to_rgb(flow[0].cpu().numpy().copy())
        pc1_o3d_gt.points = o3d.utility.Vector3dVector(pc1[0].cpu().numpy().copy())
        pc1_o3d_gt.colors = o3d.utility.Vector3dVector(colors_flow / 255.0)
        custom_draw_geometry_with_key_callback([pc1_o3d_gt])  # Press 'k' to see with dark background.
        
        # NOTE: predicted flow
        pc1_o3d_pred = o3d.geometry.PointCloud()
        colors_flow = flow_to_rgb(info_dict['final_flow'][0].detach().cpu().numpy().copy())
        pc1_o3d_pred.points = o3d.utility.Vector3dVector(pc1[0].cpu().numpy().copy())
        pc1_o3d_pred.colors = o3d.utility.Vector3dVector(colors_flow / 255.0)
        custom_draw_geometry_with_key_callback([pc1_o3d_pred])  # Press 'k' to see with dark background.
    
    return info_dict


def optimize_neural_prior(options, data_loader):
    if options.time:
        timers = Timers()
        timers.tic("total_time")
        
    outputs = []
    
    for i in range(len(data_loader)):
        fi_name = data_loader[i]
        with open(fi_name, 'rb') as fp:
            data = np.load(fp)
            pc1 = torch.from_numpy(data['pc1']).unsqueeze(0)
            pc2 = torch.from_numpy(data['pc2']).unsqueeze(0)
            flow = torch.from_numpy(data['flow']).unsqueeze(0)
            fp.close()
            
        if not options.use_all_points:
            sample_idx = torch.randperm(min(pc1.shape[1], pc2.shape[1]))[:options.num_points]
            pc1 = pc1[:, sample_idx]
            pc2 = pc2[:, sample_idx]
            flow = flow[:, sample_idx]
    
        logging.info(f"# {i} Working on sample: {fi_name}...")
        
        info_dict = solver(pc1, pc2, flow, options, options.iters)

        # Collect results.
        outputs.append(dict(list(info_dict.items())[1:]))
        print(dict(list(info_dict.items())[1:]))
        
    if options.time:
        timers.toc("total_time")
        logging.info(timers.print())

    df = pd.DataFrame(outputs)
    df.loc['mean'] = df.mean()
    logging.info(df.mean())

    logging.info("Finish optimization!")
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Scene Flow Prior.")
    
    # ANCHOR: general configuration
    parser.add_argument('--exp_name', type=str, default='fast_neural_scene_flow_kronecker_dt', metavar='N', help='Name of the experiment.')
    parser.add_argument('--num_points', type=int, default=2048, help='Point number [default: 2048].')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size', help='Batch size.')
    parser.add_argument('--iters', type=int, default=5000, metavar='N', help='Number of iterations to optimize the model.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='Learning rate.')
    parser.add_argument('--device', default='cuda:0', type=str, help='device: cpu? cuda?')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='Random seed (default: 1234).')
    parser.add_argument('--dataset', type=str, default='ArgoverseSceneFlowDataset',
                        choices=['ArgoverseSceneFlowDataset', 'WaymoOpenSceneFlowDataset'], metavar='N', help='Dataset to use.')
    parser.add_argument('--dataset_path', type=str, default='./dataset/argoverse', metavar='N', help='Dataset path.')
    parser.add_argument('--visualize', action='store_true', default=False, help='Show visuals.')
    parser.add_argument('--time', dest='time', action='store_true', default=True, help='Count the execution time of each step.')
    parser.add_argument('--use_all_points', action='store_true', default=False, help='use all the points or not.')
    parser.add_argument('--early_patience', type=int, default=100, help='patience in early stopping.')
    parser.add_argument('--early_min_delta', type=float, default=0.0001, help='the minimum delta of early stopping.')
    parser.add_argument('--init_weight', action='store_true', default=False, help='whether initialize weights on each scenes or not.')
    parser.add_argument('--earlystopping', action='store_true', default=False, help='whether to use early stopping or not.')
    
    # ANCHOR: for kronecker PE
    parser.add_argument('--grid_factor', type=float, default=2., help='grid size.')
    parser.add_argument('--gauss_sigma', type=float, default=0.008, help='sigma for gaussian PE.')
    
    # ANCHOR: for explicit regularizer
    parser.add_argument('--reg_scaling', type=float, default=1., help='scaling factor for regularizer.')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='epsilon to prevent divide by zeros in regularizer.')
    
    # ANCHOR: for distance transform
    parser.add_argument('--dt_grid_factor', type=float, default=10., help='grid size for distance transform.')
    
    options = parser.parse_args()

    exp_dir_path = f"checkpoints/{options.exp_name}"
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(filename=f"{exp_dir_path}/run.log"), logging.StreamHandler()])
    logging.info(options)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info('---------------------------------------')
    print_options = vars(options)
    for key in print_options.keys():
        logging.info(key+': '+str(print_options[key]))
    logging.info('---------------------------------------')

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(options.seed)
    if 'cuda' in options.device:
        torch.cuda.manual_seed_all(options.seed)
    np.random.seed(options.seed)

    if options.dataset == "ArgoverseSceneFlowDataset":
        data_loader = sorted(glob.glob(f"{options.dataset_path}/val/*/*.npz"))
    elif options.dataset == 'WaymoOpenSceneFlowDataset':
        data_loader = sorted(glob.glob(f"{options.dataset_path}/*/*.npz"))

    optimize_neural_prior(options, data_loader)
