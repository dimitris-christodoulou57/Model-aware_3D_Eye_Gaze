import os
import copy
from random import random
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage import draw
from sklearn import metrics




def get_com(seg_map, temperature=4):
    # Custom function to find the center of mass to get detected pupil or iris
    # center. The center of mass is in normalized space between -1 and 1.
    # seg_map: BXHXW - single channel corresponding to pupil or iris prediction

    device = seg_map.device

    B, H, W = seg_map.shape
    wtMap = F.softmax(seg_map.view(B, -1)*temperature, dim=1)  # [B, HXW]

    XYgrid = create_meshgrid(H, W, normalized_coordinates=True)  # 1xHxWx2
    xloc = XYgrid[0, :, :, 0].reshape(-1).to(device)
    yloc = XYgrid[0, :, :, 1].reshape(-1).to(device)

    xpos = torch.sum(wtMap*xloc, -1, keepdim=True)
    ypos = torch.sum(wtMap*yloc, -1, keepdim=True)
    predPts = torch.stack([xpos, ypos], dim=1).squeeze()

    return predPts


def construct_mask_from_ellipse(ellipses, res):
    if len(ellipses.shape) == 1:
        ellipses = ellipses[np.newaxis, :]
        B = 1
    else:
        B = ellipses.shape[0]

    mask = np.zeros((B, ) + res)
    for b in range(B):
        ellipse = ellipses[b, ...].tolist()
        [rr, cc] = draw.ellipse(round(ellipse[1]),
                                round(ellipse[0]),
                                round(ellipse[3]),
                                round(ellipse[2]),
                                shape=res,
                                rotation=-ellipse[4])
        rr = np.clip(rr, 0, res[0]-1)
        cc = np.clip(cc, 0, res[1]-1)
        mask[b, rr, cc] = 1
    return mask.astype(bool)


def fix_ellipse_axis_angle(ellipse):
    ellipse = copy.deepcopy(ellipse)
    if ellipse[3] > ellipse[2]:
        ellipse[[2, 3]] = ellipse[[3, 2]]
        ellipse[4] += np.pi/2

    if ellipse[4] > np.pi:
        ellipse[4] += -np.pi
    elif ellipse[4] < 0:
        ellipse[4] += np.pi
    return ellipse


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


def getSizes(chz, growth, blks=4):
    # For a base channel size, growth rate and number of blocks,
    # this function computes the input and output channel sizes for
    # al layers.

    # Encoder sizes
    sizes = {'enc': {'inter': [], 'ip': [], 'op': []},
             'dec': {'skip': [], 'ip': [], 'op': []}}
    sizes['enc']['inter'] = np.array([chz*(i+1) for i in range(0, blks)])
    sizes['enc']['op'] = np.array([int(growth*chz*(i+1)) for i in range(0, blks)])
    sizes['enc']['ip'] = np.array([chz] + [int(growth*chz*(i+1)) for i in range(0, blks-1)])

    # Decoder sizes
    sizes['dec']['skip'] = sizes['enc']['ip'][::-1] + sizes['enc']['inter'][::-1]
    sizes['dec']['ip'] = sizes['enc']['op'][::-1]
    sizes['dec']['op'] = np.append(sizes['enc']['op'][::-1][1:], chz)
    return sizes


