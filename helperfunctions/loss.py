#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:05:09 2020

@author: rakshit
"""
import torch
import numpy as np
import torch.nn.functional as F

from einops import rearrange

from helperfunctions.utils import create_meshgrid, soft_heaviside


def get_seg_loss(gt_dict, pd_dict, alpha):
    # Custom function to iteratively go over each sample in a batch and
    # compute loss.
    # cond: Mask exist -> 1, else 0

    op = pd_dict['predict']
    target = gt_dict['mask']
    spatWts = gt_dict['spatial_weights']
    distMap = gt_dict['distance_map']

    device = op.device

    # Mask availability
    cond = gt_dict['mask_available'].to(device)

    B = op.shape[0]
    loss_seg = []

    for i in range(0, B):
        if cond[i]:
            # Valid mask exists
            l_sl = SurfaceLoss(op[i, ...].unsqueeze(0),
                               distMap[i, ...].unsqueeze(0))
            l_cE = wCE(op[i, ...],
                       target[i, ...],
                       spatWts[i, ...])
            l_gD = GDiceLoss(op[i, ...].unsqueeze(0),
                             target[i, ...].unsqueeze(0),
                             F.softmax)
            loss_seg.append(alpha*l_sl + (1-alpha)*l_gD + l_cE)

    if len(loss_seg) > 0:
        total_loss = torch.sum(torch.stack(loss_seg))/torch.sum(cond)

        assert total_loss.dim() == 0, 'Segmentation losses must be a scalar'
        return total_loss
    else:
        return torch.tensor([0.0]).to(device)

def get_com(seg_map, temperature=4):
    # Custom function to find the center of mass to get detected pupil or iris
    # center. The center of mass is in normalized space between -1 and 1.
    # seg_map: BXHXW - single channel corresponding to pupil or iris prediction

    device = seg_map.device

    #reduce the number of dimension combine the batch size and the number of frames
    seg_map = rearrange(seg_map, 'b f h w -> (b f) h w')

    B, H, W = seg_map.shape
    wtMap = F.softmax(rearrange(seg_map, 'b h w -> b (h w)')*temperature, dim=1)  # [B, HXW]

    XYgrid = create_meshgrid(H, W, normalized_coordinates=True)  # 1xHxWx2
    xloc = XYgrid[0, :, :, 0].reshape(-1).to(device)
    yloc = XYgrid[0, :, :, 1].reshape(-1).to(device)

    xpos = torch.sum(wtMap*xloc, -1, keepdim=True)
    ypos = torch.sum(wtMap*yloc, -1, keepdim=True)
    predPts = torch.stack([xpos, ypos], dim=1).squeeze()

    #the returning tensor has the following dimension [BXF, HXW]
    return predPts


def get_uncertain_l1_loss(ip_vector,
                          target_vector,
                          weights_per_channel,
                          uncertain,
                          cond,
                          do_aleatoric=False):
    ''' Find L1 or aleatoric L1 distance over valid samples '''

    if weights_per_channel is None:
        weights_per_channel = torch.ones(ip_vector.shape[1],
                                         dtype=ip_vector.dtype).to(ip_vector.device)
    else:
        weights_per_channel = torch.tensor(weights_per_channel).to(ip_vector.dtype).to(ip_vector.device)

    if torch.any(cond):
        loss_per_sample = F.l1_loss(ip_vector, target_vector, reduction='none')

        if do_aleatoric:
            # If uncertain estimates are present, use the aleatoric formulation
            loss_per_sample = .1*uncertain + \
                loss_per_sample/torch.exp(uncertain)

        # Sum across dimensions and weighted average across samples
        loss_per_sample = loss_per_sample*weights_per_channel
        loss_per_sample = torch.sum(loss_per_sample, dim=1)*cond
        total_loss = torch.sum(loss_per_sample)/torch.sum(cond)

        assert total_loss.dim() == 0, 'L1 loss must be a scalar entity'
        return total_loss
    else:
        # No valid sample found
        return torch.tensor([0.0]).to(ip_vector.device)


def SurfaceLoss(x, distmap):
    # For classes with no groundtruth, distmap would ideally be filled with 0s
    x = torch.softmax(x, dim=1)
    score = x.flatten(start_dim=2)*distmap.flatten(start_dim=2)
    score = torch.mean(score, dim=2)  # Mean between pixels per channel
    score = torch.mean(score, dim=1)  # Mean between channels
    return score


def GDiceLoss(ip, target, norm=F.softmax):

    mxLabel = ip.shape[1]
    allClasses = np.arange(mxLabel, )
    labelsPresent = np.unique(target.cpu().numpy())

    device = target.device

    Label = (np.arange(mxLabel) == target.cpu().numpy()[..., None]).astype(np.uint8)
    Label = np.moveaxis(Label, 3, 1)
    target = torch.from_numpy(Label).to(device).to(ip.dtype)


    loc_rm = np.where(~np.in1d(allClasses, labelsPresent))[0]

    assert ip.shape == target.shape
    ip = norm(ip, dim=1)  # Softmax or Sigmoid over channels
    ip = torch.flatten(ip, start_dim=2, end_dim=-1)
    target = torch.flatten(target, start_dim=2, end_dim=-1).to(device).to(ip.dtype)

    numerator = ip*target
    denominator = ip + target

    # For classes which do not exist in target but exist in input, set weight=0
    class_weights = 1./(torch.sum(target, dim=2)**2).clamp(1e-5)
    if loc_rm.size > 0:
        for i in np.nditer(loc_rm):
            class_weights[:, i.item()] = 0
    A = class_weights*torch.sum(numerator, dim=2)
    B = class_weights*torch.sum(denominator, dim=2)
    dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
    return torch.mean(1 - dice_metric.clamp(1e-5))


def wCE(ip, target, spatWts):
    mxLabel = ip.shape[0]
    allClasses = np.arange(mxLabel, )
    labelsPresent = np.unique(target.cpu().numpy())
    rmIdx = allClasses[~np.in1d(allClasses, labelsPresent)]
    if rmIdx.size > 0:
        loss = spatWts.view(1, -1)*F.cross_entropy(ip.view(1, mxLabel, -1),
                                                   target.view(1, -1),
                                                   ignore_index=rmIdx.item())
    else:
        loss = spatWts.view(1, -1)*F.cross_entropy(ip.view(1, mxLabel, -1),
                                                   target.long().view(1, -1))
    loss = torch.mean(loss)
    return loss


def get_mask(mesh, opEl):
    # posmask: Positive outside the ellipse
    # negmask: Positive inside the ellipse
    X = (mesh[..., 0] - opEl[0])*torch.cos(opEl[-1]) + \
        (mesh[..., 1]-opEl[1])*torch.sin(opEl[-1])
    Y = -(mesh[..., 0]-opEl[0])*torch.sin(opEl[-1]) + \
        (mesh[..., 1]-opEl[1])*torch.cos(opEl[-1])
    posmask = (X/opEl[2])**2 + (Y/opEl[3])**2 - 1
    negmask = 1 - (X/opEl[2])**2 - (Y/opEl[3])**2

    posmask = soft_heaviside(posmask, sc=64, mode=3)
    negmask = soft_heaviside(negmask, sc=64, mode=3)
    return posmask, negmask
