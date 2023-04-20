"""optimize over a network structure."""

import argparse
import logging
import os, glob
from copy import deepcopy
import time
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
        wheel[angle_floor.astype(np.int)] * (1 - angle_fractional) + wheel[angle_ceil.astype(np.int)] * angle_fractional
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
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float)
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float))
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
        # opt.background_color = np.asarray([7/255, 54/255, 66/255])
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

  
class Neural_Prior(nn.Module):
    def __init__(self, input_size=1000, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, output_feat=False):
        super().__init__()
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_feat = output_feat
        
        self.nn_layers = nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(nn.Sequential(nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(nn.Sigmoid())
            
            for _ in range(layer_size-1):
                self.nn_layers.append(nn.Sequential(nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(nn.Sigmoid())
                
            self.nn_layers.append(nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(nn.Sequential(nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        if self.output_feat:
            feat = []
        for layer in self.nn_layers:
            x = layer(x)
            if self.output_feat and layer == nn.Linear:
                feat.append(x)

        if self.output_feat:
            return x, feat
        else:
            return x
        
        
def solver(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    flow: torch.Tensor,
    options: argparse.Namespace,
    net: nn.Module,
    max_iters: int
):
    
    if options.time:
        timers = Timers()
        timers.tic("solver_timer")
    
    pre_compute_st = time.time()
    solver_time = 0.

    if options.init_weight:
        net.apply(init_weights)
        
    for param in net.parameters():
        param.requires_grad = True
    
    params = net.parameters()
    
    optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.momentum, weight_decay=options.weight_decay)
                
    total_losses = []
    total_acc_strit = []
    total_iter_time = []
    
    if options.earlystopping:
        early_stopping = EarlyStopping(patience=options.early_patience, min_delta=options.early_min_delta)

    dt_start_time = time.time()
    
    pc1_min = torch.min(pc1.squeeze(0), 0)[0]
    pc2_min = torch.min(pc2.squeeze(0), 0)[0]
    pc1_max = torch.max(pc1.squeeze(0), 0)[0]
    pc2_max = torch.max(pc2.squeeze(0), 0)[0]
    
    xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min<pc2_min, pc1_min, pc2_min) * options.grid_factor-1) / options.grid_factor
    xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max>pc2_max, pc1_max, pc2_max)* options.grid_factor+1) / options.grid_factor
    print('xmin: {}, xmax: {}, ymin: {}, ymax: {}, zmin: {}, zmax: {}'.format(xmin_int, xmax_int, ymin_int, ymax_int, zmin_int, zmax_int))
    
    # NOTE: build DT map
    dt = DT(pc2.clone().squeeze(0).to(options.device), (xmin_int, ymin_int, zmin_int), (xmax_int, ymax_int, zmax_int), options.grid_factor, options.device)

    dt_time = time.time() - dt_start_time
    
    pc1 = pc1.to(options.device).contiguous()
    pc2 = pc2.to(options.device).contiguous()
    flow = flow.to(options.device).contiguous()
    print(pc1.shape, pc2.shape, flow.shape)
    
    pre_compute_time = time.time() - pre_compute_st
    solver_time = solver_time + pre_compute_time
    
    # ANCHOR: initialize best metrics
    best_loss_1 = 1e10
    best_flow_1 = None
    best_epe3d_1 = 1.
    best_acc3d_strict_1 = 0.
    best_acc3d_relax_1 = 0.
    best_angle_error_1 = 1.
    best_outliers_1 = 1.
    best_epoch = 0
    kdtree_query_time = 0.
    net_time = 0.
    net_backward_time = 0.
    dt_query_time = 0.
    
    for epoch in range(max_iters):
        iter_time_init = time.time()

        optimizer.zero_grad()
        
        net_time_st = time.time()
        flow_pred_1 = net(pc1)
        net_time = net_time + time.time() - net_time_st
        pc1_deformed = pc1 + flow_pred_1
        
        dt_query_st = time.time()
        loss = dt.torch_bilinear_distance(pc1_deformed.squeeze(0)).mean()
        dt_query_time = dt_query_time + time.time() - dt_query_st
        
        net_backward_st = time.time()
        loss.backward()
        optimizer.step()
        net_backward_time = net_backward_time + time.time() - net_backward_st
    
        if options.earlystopping:
            if early_stopping.step(loss):
                break
            
        iter_time = time.time() - iter_time_init
        solver_time = solver_time + iter_time

        flow_pred_1_final = pc1_deformed - pc1
        flow_metrics = flow.clone()
        EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = scene_flow_metrics(flow_pred_1_final, flow_metrics)

        # ANCHOR: get best metrics
        if loss <= best_loss_1:
            best_loss_1 = loss.item()
            best_flow_1 = flow_pred_1_final
            best_epe3d_1 = EPE3D_1
            best_acc3d_strict_1 = acc3d_strict_1
            best_acc3d_relax_1 = acc3d_relax_1
            best_angle_error_1 = angle_error_1
            best_outliers_1 = outlier_1
            best_epoch = epoch
        
        total_losses.append(loss.item())
        total_acc_strit.append(acc3d_strict_1)
        total_iter_time.append(time.time()-iter_time_init)
            
        if epoch % 50 == 0:
            logging.info(f"[Ep {epoch}] [Loss: {loss.item():.5f}] "
                        f" Metrics: flow 1 --> flow 2"
                        f" [EPE: {EPE3D_1:.3f}] [Acc strict: {acc3d_strict_1 * 100:.3f}%]"
                        f" [Acc relax: {acc3d_relax_1 * 100:.3f}%] [Angle error (rad): {angle_error_1:.3f}]"
                        f" [Outl.: {outlier_1 * 100:.3f}%]")
    
    if options.time:
        timers.toc("solver_timer")
        time_avg = timers.get_avg("solver_timer")
        logging.info(timers.print())
        
    # ANCHOR: get the best metrics
    info_dict = {
        'final_flow': best_flow_1,
        'loss': best_loss_1,
        'EPE3D_1': best_epe3d_1,
        'acc3d_strict_1': best_acc3d_strict_1,
        'acc3d_relax_1': best_acc3d_relax_1,
        'angle_error_1': best_angle_error_1,
        'outlier_1': best_outliers_1,
        'time': time_avg,
        'epoch': best_epoch,
        'solver_time': solver_time,
        'pre_compute_time': pre_compute_time,
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
    
    if options.model == 'neural_prior':
        net = Neural_Prior(filter_size=options.hidden_units, act_fn=options.act_fn, layer_size=options.layer_size).to(options.device)
    else:
        raise Exception("Model not available.")
    
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
        
        info_dict = solver(pc1, pc2, flow, options, net, options.iters)

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
    parser = argparse.ArgumentParser(description="Fast Neural Scene Flow.")
    
    # ANCHOR: from config.py
    parser.add_argument('--exp_name', type=str, default='fast_neural_scene_flow_mlp_dt', metavar='N', help='Name of the experiment.')
    parser.add_argument('--num_points', type=int, default=2048, help='Point number [default: 2048].')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size', help='Batch size.')
    parser.add_argument('--iters', type=int, default=5000, metavar='N', help='Number of iterations to optimize the model.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0, metavar='M', help='SGD momentum (default: 0.9).')
    parser.add_argument('--device', default='cuda:0', type=str, help='device: cpu? cuda?')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='Random seed (default: 1234).')
    parser.add_argument('--dataset', type=str, default='ArgoverseSceneFlowDataset',
                        choices=['ArgoverseSceneFlowDataset', 'WaymoOpenSceneFlowDataset'], metavar='N', help='Dataset to use.')
    parser.add_argument('--dataset_path', type=str, default='./dataset/argoverse', metavar='N', help='Dataset path.')
    parser.add_argument('--visualize', action='store_true', default=False, help='Show visuals.')
    parser.add_argument('--time', dest='time', action='store_true', default=True, help='Count the execution time of each step.')
    
    # For neural prior
    parser.add_argument('--model', type=str, default='neural_prior', choices=['neural_prior', 'linear_model', 'kronecker_model'], metavar='N', help='Model to use.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='N', help='Weight decay.')
    parser.add_argument('--hidden_units', type=int, default=128, metavar='N', help='Number of hidden units in neural prior')
    parser.add_argument('--layer_size', type=int, default=8, help='how many hidden layers in the model.')
    parser.add_argument('--use_all_points', action='store_true', default=False, help='use all the points or not.')
    parser.add_argument('--act_fn', type=str, default='relu', metavar='AF', help='activation function for neural prior.')
    parser.add_argument('--early_patience', type=int, default=100, help='patience in early stopping.')
    parser.add_argument('--early_min_delta', type=float, default=0.0001, help='the minimum delta of early stopping.')
    parser.add_argument('--init_weight', action='store_true', default=False, help='whether initialize weights on each scenes or not.')
    parser.add_argument('--earlystopping', action='store_true', default=False, help='whether to use early stopping or not.')
    
    # for distance transform
    parser.add_argument('--grid_factor', type=float, default=10., help='grid cell size=1/grid_factor.')
    
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
