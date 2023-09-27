#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

# This file contains definitions which are not applicable in regular scenarios.
# For general purposes functions, classes and operations - use helperfunctions.

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from random import random
from typing import Optional
from sklearn import metrics

from extern.FilterResponseNormalizationLayer.frn import FRN, TLU


def concat_list_of_data_dicts(list_dicts):
    keys = list(list_dicts[0].keys())
    out_dict = {key: [] for key in keys}
    for key in keys:
        if type(list_dicts[0]['key']) == list:
            out_dict[key] += [ele[key] for ele in list_dicts]
        else:
            out_dict[key] = torch.cat([ele[key] for ele in list_dicts], dim=0)


def get_selected_set(data_dict, ds_num):
    all_ds = data_dict['ds_num']
    idx = [i for i, x in enumerate(all_ds) if x == ds_num]

    out_dict = {}
    for key, value in data_dict.items():
        if type(value) == list:
            out_dict[key] = [value[ele] for ele in idx]
        else:
            out_dict[key] = value[idx, ...]
    return out_dict


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.

    NOTE: Function taken from Kornia libary
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)

    # Ensure that xs and ys do not require gradients
    xs.requires_grad = False
    ys.requires_grad = False

    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_predictions(output):
    '''
    Parameters
    ----------
    output : torch.tensor
        [B, C, *] tensor. Returns the argmax for one-hot encodings.

    Returns
    -------
    indices : torch.tensor
        [B, *] tensor.

    '''
    bs,c,h,w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs,h,w) # bs x h x w
    return indices

class make_logger():
    def __init__(self, output_name, rank):
        self.rank_cond = rank == 0
        if self.rank_cond:
            dirname = os.path.dirname(output_name)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            self.dirname = dirname
            self.log_file = open(output_name, 'a+')

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, info_dict, extra_msg=''):

        if self.rank_cond:
            msgs = [extra_msg]
            for key, vals in info_dict.iteritems():
                msgs.append('%s %.6f' % (key, np.mean(vals)))
            msg = '\n'.join(msgs)

            self.write(msg)
            print (msg)

    def write(self, msg, do_silent=False, do_warn=False):

        if self.rank_cond:
            msg = 'WARNING! '+msg if do_warn else msg
            self.log_file.write(msg+'\n')
            self.log_file.flush()

            if not do_silent:
                print (msg)

    def write_summary(self,msg):

        if self.rank_cond:
            self.log_file.write(msg)
            self.log_file.write('\n')
            self.log_file.flush()
            print (msg)

def get_seg_metrics(y_true, y_pred, cond, batch, frame):
    '''
    Iterate over each batch and identify which classes are present. If no
    class is present, i.e. all 0, then ignore that score from the average.
    Note: This function computes the nan mean. This is because datasets may not
    have all classes present.
    '''
    assert y_pred.ndim==3, 'Incorrect number of dimensions'
    assert y_true.ndim==3, 'Incorrect number of dimensions'

    cond = cond.astype(np.bool)
    B = y_true.shape[0]
    score_list = []

    for i in range(0, B):

        labels_present = np.unique(y_true[i, ...])
        score_vals = np.empty((3, ))
        score_vals[:] = np.nan

        if cond[i]:
            score = metrics.jaccard_score(y_true[i, ...].reshape(-1),
                                          y_pred[i, ...].reshape(-1),
                                          labels=labels_present,
                                          average=None)

            # Assign score to relevant location
            for j, val in np.ndenumerate(labels_present):
                score_vals[val] = score[j]

        score_list.append(score_vals)

    score_list = np.stack(score_list, axis=0)
    # score_list_clean = score_list[~cond, :] # Only select valid entries
    # perClassIOU = np.nanmean(score_list_clean, axis=0) if len(score_list_clean) > 0 else np.nan*np.ones(3, )
    # meanIOU = np.nanmean(perClassIOU) if len(score_list_clean) > 0 else np.nan
    # return meanIOU, perClassIOU, score_list
    return score_list if np.any(cond) else np.zeros((len(cond), ))


def get_distance(y_true, y_pred, cond, metric='euclidean'):
    '''
    Parameters
    ----------
    y_true : Groundtruth matrix of vectors
        The matrix should contain vectors in a manner [B, *]
    y_pred : Predicted matrix of vectors
        The matrix should contain vectors in a manner [B, *]
    cond : TYPE
        DESCRIPTION.
    metric : TYPE, optional
        DESCRIPTION. The default is 'euclidean'.

    Returns
    -------
    TYPE
        Distance .
    '''

    flag = cond.astype(np.bool)

    dist = np.linalg.norm(y_true - y_pred, axis=1)
    dist[~flag] = np.nan
    return dist if np.any(flag) else np.zeros((y_true.shape[0], ))


def getAng_metric(y_true, y_pred, cond):
    # Assumes the incoming angular measurements are in radians
    flag = cond.astype(np.bool)
    dist = np.abs(y_true - y_pred)
    dist[~flag] = np.nan
    return dist if np.any(flag) else np.zeros((y_true.shape[0], ))


def normPts(pts, sz, by_max=False):
    if by_max:
        return 2*(pts/max(sz)) - 1
    else:
        return 2*(pts/sz) - 1

def unnormPts(pts, sz, by_max=False):
    if by_max:
        return 0.5*sz*(pts + 1)
    else:
        return 0.5*sz*(max(pts) + 1)

def compute_norm(model):
    list_of_norms = []
    for _, param in model.named_parameters():
        if param.grad is not None:
            list_of_norms.append(torch.norm(param.grad.detach(), 2).to('cpu'))
    list_of_norms = torch.stack(list_of_norms)
    total_norm = torch.norm(list_of_norms, 2)
    return total_norm

def points_to_heatmap(pts, std, res):
    # Given image resolution and variance, generate synthetic Gaussians around
    # points of interest for heat map regression.
    # pts: [B, C, N, 2] Normalized points
    # H: [B, C, N, H, W] Output heatmap
    B, C, N, _ = pts.shape
    pts = unnormPts(pts, res) #
    grid = create_meshgrid(res[0], res[1], normalized_coordinates=False)
    grid = grid.squeeze()
    X = grid[..., 0]
    Y = grid[..., 1]

    X = torch.stack(B*C*N*[X], axis=0).reshape(B, C, N, res[0], res[1])
    X = X - torch.stack(np.prod(res)*[pts[..., 0]], axis=3).reshape(B, C, N, res[0], res[1])

    Y = torch.stack(B*C*N*[Y], axis=0).reshape(B, C, N, res[0], res[1])
    Y = Y - torch.stack(np.prod(res)*[pts[..., 1]], axis=3).reshape(B, C, N, res[0], res[1])

    H = torch.exp(-(X**2 + Y**2)/(2*std**2))
    #H = H/(2*np.pi*std**2) # This makes the summation == 1 per image in a batch
    return H

def ElliFit(coords, mns):
    '''
    Parameters
    ----------
    coords : torch float32 [B, N, 2]
        Predicted points on ellipse periphery
    mns : torch float32 [B, 2]
        Predicted mean of the center points

    Returns
    -------
    PhiOp: The Phi scores associated with ellipse fitting. For more info,
    please refer to ElliFit paper.
    '''
    B = coords.shape[0]

    PhiList = []

    for bt in range(B):
        coords_norm = coords[bt, ...] - mns[bt, ...] # coords_norm: [N, 2]
        N = coords_norm.shape[0]

        x = coords_norm[:, 0]
        y = coords_norm[:, 1]

        X = torch.stack([-x**2, -x*y, x, y, -torch.ones(N, ).cuda()], dim=1)
        Y = y**2

        a = torch.inverse(X.T.matmul(X))
        b = X.T.matmul(Y)
        Phi = a.matmul(b)
        PhiList.append(Phi)
    Phi = torch.stack(PhiList, dim=0)
    return Phi

def spatial_softmax_2d(input: torch.Tensor, temperature: torch.Tensor = torch.tensor(1.0)) -> torch.Tensor:
    r"""Applies the Softmax function over features in each image channel.
    Note that this function behaves differently to `torch.nn.Softmax2d`, which
    instead applies Softmax over features at each spatial location.
    Returns a 2D probability distribution per image channel.
    Arguments:
        input (torch.Tensor): the input tensor.
        temperature (torch.Tensor): factor to apply to input, adjusting the
          "smoothness" of the output distribution. Default is 1.
    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, H, W)`
    """

    batch_size, channels, height, width = input.shape
    x: torch.Tensor = input.view(batch_size, channels, -1)

    x_soft: torch.Tensor = F.softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)

class SpikeDetection():
    '''
    Custom spike detection module to skip learning on crappy batches
    '''
    def __init__(self,
                 patience=5,
                 threshold=2.5,
                 window_size=100,
                 ):

        import collections
        self.entries = collections.deque(maxlen=window_size)
        self.win_size = window_size
        self.threshold = threshold
        self.patience = patience
        self.count = 0

    def update(self, val):

        std_window = np.nanstd(self.entries)
        cond_std = np.abs(val - np.mean(self.entries)) > self.threshold*std_window

        if (len(self.entries) > self.win_size//2) and cond_std and (self.count > self.patience):
            is_spike = True
            self.count = 0
        else:
            self.count += 1
            is_spike = False
            self.entries.append(val)
        return is_spike


class EarlyStopping:
    """Early stops the training if validation loss/metric doesn't improve after a given patience."""
    # Modified by Rakshit Kothari.
    # Code taken from here: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self,
                metric = None,
                patience=7,
                verbose=False,
                rank_cond=True,
                delta=0,
                mode='min',
                fName = 'checkpoint.pt',
                path_save = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/checkpoints'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            fName (str): Name of the checkpoint file.
            path_save (str): Location of the checkpoint file.
        """
        self.patience = patience
        self.verbose = verbose if rank_cond else False
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta
        self.path_save = path_save
        self.fName = fName
        self.mode = mode
        self.rank_cond = rank_cond
        self.metric = metric

    def __call__(self, checkpoint):

        if '3D' in self.metric:
            val_score = checkpoint['valid_result']['gaze_3D_ang_deg_mean']
        elif '2D' in self.metric:
            val_score = checkpoint['valid_result']['gaze_ang_deg_mean']
        else:
            val_score = checkpoint['valid_result']['masked_rendering_iou_mean']
        score = -val_score if self.mode =='min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, checkpoint)

        elif score < (self.best_score + self.delta):
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_score, checkpoint)
            self.counter = 0

    def save_checkpoint(self, val_score, model_dict,
                        update_val_score=True,
                        use_this_name_instead=False):

        '''Saves model when val_score decreases.'''
        if self.verbose and (self.mode == 'min'):
            print('Validation metric decreased ({:.6f} --> {:.6f}). Saving..'.format(self.val_loss_min,
                                                                                     val_score))
        elif self.verbose and (self.mode == 'max'):
            print('Validation metric increased ({:.6f} --> {:.6f}). Saving..'.format(self.val_loss_min,
                                                                                     val_score))
        if (self.rank_cond):
            if use_this_name_instead:
                torch.save(model_dict, os.path.join(self.path_save, use_this_name_instead))
            else:
                torch.save(model_dict, os.path.join(self.path_save, self.fName))

        if update_val_score:
            self.val_loss_min = val_score
        return


def generate_pseudo_labels(input: torch.Tensor):
    r"""
    Generates pseudo labels based on per-pixel entropy

    Parameters
    ----------
    input : torch.Tensor
        Non-softmax'ed output of a segmentation network.

    Returns
    -------
    pseudo_labels : torch.Tensor
        Generated psuedo label based on segmentation output.
    confidence : torch.Tensor
        Returns confidence as 1 - (entropy metric/log K);
        where K is the number of classes
    """
    num_classes = input.shape[1]
    sc = np.log(num_classes)

    logits = F.softmax(input, dim=1)
    logits_log = F.log_softmax(input, dim=1)  # Better than using log on logits
    entropy = -torch.sum(logits*logits_log, dim=1).detach()

    pseudo_labels = torch.argmax(logits, dim=1).detach().to(torch.long)

    return pseudo_labels, 1 - (entropy/sc)


def remove_underconfident_psuedo_labels(conf,
                                        label_tracker=False,
                                        gt_dict=False):

    # Switch off groundtruth if no label tracker is given
    gt_dict = gt_dict if label_tracker else False

    if gt_dict:

        gt_dict['mask_available']

        return conf > label_tracker.threshold
    else:
        return conf > 0.95



def spatial_softargmax_2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""
    Computes the 2D soft-argmax of a given input heatmap.
    The input heatmap is assumed to represent a valid spatial probability
    distribution, which can be achieved using
    :class:`~kornia.contrib.dsnt.spatial_softmax_2d`.
    Returns the index of the maximum 2D coordinates of the given heatmap.
    The output order of the coordinates is (x, y).
    Arguments:
        input (torch.Tensor): the input tensor.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`
    Examples:
        >>> heatmaps = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.]]]])
        >>> coords = spatial_softargmax_2d(heatmaps, False)
        tensor([[[1.0000, 2.0000]]])
    """

    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates)
    grid = grid.to(device=input.device, dtype=input.dtype)

    pos_x: torch.Tensor = grid[..., 0].reshape(-1)
    pos_y: torch.Tensor = grid[..., 1].reshape(-1)

    input_flat: torch.Tensor = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y: torch.Tensor = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x: torch.Tensor = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output: torch.Tensor = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2

def soft_heaviside(x, sc, mode):
    '''
    Given an input and a scaling factor (default 64), the soft heaviside
    function approximates the behavior of a 0 or 1 operation in a differentiable
    manner. Note the max values in the heaviside function are scaled to 0.9.
    This scaling is for convenience and stability with bCE loss.
    '''
    sc = torch.tensor([sc]).to(torch.float32).to(x.device)
    if mode==1:
        # Original soft-heaviside
        # Try sc = 64
        return 0.9/(1 + torch.exp(-sc/x))
    elif mode==2:
        # Some funky shit but has a nice gradient
        # Try sc = 0.001
        return 0.45*(1 + (2/np.pi)*torch.atan2(x, sc))
    elif mode==3:
        # Good ol' scaled sigmoid. FUTURE: make sc free parameter
        # Try sc = 8
        return torch.sigmoid(sc*x)
    else:
        print('Mode undefined')


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


def move_to_multi(model_dict):
    '''
    Convert dictionary of weights and keys
    to a multiGPU format. It simply appends
    a 'module.' in front of keys.
    '''
    multiGPU_dict = {}
    for key, value in model_dict.items():
        multiGPU_dict['module.'+key] = value
    return multiGPU_dict


def move_to_single(model_dict, move_to_cpu=True):
    '''
    Convert dictionary of weights and keys
    to a singleGPU format. It removes the
    'module.' in front of keys. Furthermore,
    it moves all the weights to CPU just in case
    '''
    singleGPU_dict = {}
    for key, value in model_dict.items():
        if move_to_cpu:
            singleGPU_dict[key.replace('module.', '')] = value.cpu()
        else:
            singleGPU_dict[key.replace('module.', '')] = value
    return singleGPU_dict


def detach_cpu_np(ip):
    return ip.detach().cpu().numpy()


def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    return res


def do_nothing(input):
    return input

class FRN_TLU(torch.nn.Module):
    def __init__(self, channels, track_running_stats=False):
        super(FRN_TLU, self).__init__()
        self.FRN = FRN(channels, is_eps_learnable=True)
        self.TLU = TLU(channels)

    def forward(self, x):
        x = self.FRN(x)
        x = self.TLU(x)
        return x
