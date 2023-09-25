import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import faiss
import faiss.contrib.torch_utils

from helperfunctions.helperfunctions import assert_torch_invalid


class SobelFilter(nn.Module):
    '''
    Taken from: https://github.com/chaddy1004/sobel-operator-pytorch/blob/master/model.py
    '''
    def __init__(self, device):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, 
                                stride=1, padding=1, padding_mode='reflect', bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]]).to(device)
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]]).to(device)
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)

        return x
    
    
def find_nearest_targets(P_source, P_target, gpu_resource=None):
    d = P_target.shape[1]
    if gpu_resource is not None:
        l2_idx = faiss.GpuIndexFlatL2(gpu_resource, d)
    else:
        l2_idx = faiss.IndexFlatL2(d)
    l2_idx.add(P_target)
    _, nearest_tar_idx = l2_idx.search(P_source, k=1)
    return nearest_tar_idx.squeeze(1)


def nearest_target_distance(P_source, P_target, gpu_resources):
    #MSE = torch.nn.SmoothL1Loss(reduction='mean')
    MSE = torch.nn.MSELoss(reduction='mean')
    RMSE = lambda x, y: torch.sqrt(MSE(x, y))
    nearest_tar_idx = find_nearest_targets(P_source=P_source, 
                                           P_target=P_target,
                                           gpu_resource=gpu_resources)
    P_target_nearest = torch.index_select(P_target, 0, nearest_tar_idx)
    rmse_loss = RMSE(P_target_nearest, P_source)

    return rmse_loss, P_target_nearest


def mask_img_2_UV_flat(mask, class_idx):
    UV_flat = torch.where(mask==class_idx)
    B_idx = UV_flat[0]
    max_n = torch.max(torch.bincount(B_idx)).item()
    # Image is organized as (y, x)
    # Reorganize like (x, y) to match convetion in predictions
    UV_flat = torch.stack([UV_flat[2], UV_flat[1]], dim=-1)
    return UV_flat, B_idx, max_n


def mask_img_2_edge_UV_flat(mask, sobel_filter):
    mask_edge = sobel_filter(mask.unsqueeze(1)).squeeze(1)
    mask_edge = torch.where(mask_edge > 0, 1, 0)
    UV_flat, B_idx, max_n = mask_img_2_UV_flat(mask_edge, 1)
    return UV_flat, B_idx, max_n


def pad_UV_i(UV_i, max_n_i, invalid_uv=0):
    cnt_i = UV_i.shape[0]
    n_diff_i = max_n_i - cnt_i
    pad_i = invalid_uv * torch.ones(n_diff_i, 2).to(UV_i.device)
    UV_i = torch.cat((UV_i, pad_i))
    return UV_i, cnt_i


def dist_loss(x, y):
    # Euclidean distance loss
    return (x - y).square().sum(-1).sqrt().mean()


def rendered_semantics_loss_vectorized(gt_mask, rend_dict, sobel_filter, faiss_gpu_res, args):
    # TODO Implement these functionalities
    assert args['loss_w_rend_diameter'] == 0

    # Torch device
    dev = gt_mask.device

    # Class indices
    iris_idx = 1; pupil_idx = 2

    # Extract projected templates
    UV_iris_pred = rend_dict['iris_UV']
    UV_pupil_pred = rend_dict['pupil_UV']
    if args['loss_w_rend_pred_2_gt_edge']:
        which_edge = 'sides'
        side_idx_iris = rend_dict['edge_idx_iris']
        side_idx_pupil = rend_dict['edge_idx_pupil']
        UV_iris_edge_pred = UV_iris_pred[:, side_idx_iris[which_edge], :]
        UV_pupil_edge_pred = UV_pupil_pred[:, side_idx_pupil[which_edge], :]

    # Basic assertions
    assert gt_mask.shape[0] == UV_pupil_pred.shape[0] == UV_iris_pred.shape[0]
    assert len(gt_mask.shape) == 3
    assert len(UV_pupil_pred.shape) == len(UV_iris_pred.shape) == 3
    assert UV_iris_pred.shape[2] == 2, print('iris not valid number of features')
    assert UV_pupil_pred.shape[2] == 2, print('pupil not valid number of features')

    # Ground truth-pre processing does not require gradient computation
    with torch.no_grad():

        # Extract the GT mask UV positions 
        # (the tensor is flattened over the batch index - and batch here means [batch + window] bundled together)
        # (N_flat, 2)
        UV_iris_gt_flat, B_idx_iris_gt, max_n_iris_gt = mask_img_2_UV_flat(gt_mask, iris_idx)
        UV_pupil_gt_flat, B_idx_pupil_gt, max_n_pupil_gt = mask_img_2_UV_flat(gt_mask, pupil_idx)

        # Extract iris & pupil GT edge masks (using sobel filtering)
        # Then extract the GT edge UV positions
        # (flattened tensors are returned, like above)
        # (N_flat, 2)
        if args['loss_w_rend_pred_2_gt_edge']:
            iris_mask_gt = torch.where(gt_mask>=iris_idx, 1., 0.)
            UV_iris_edge_gt_flat, B_idx_iris_edge_gt, max_n_iris_edge_gt = mask_img_2_edge_UV_flat(iris_mask_gt, sobel_filter)

            pupil_mask_gt = torch.where(gt_mask==pupil_idx, 1., 0.)
            UV_pupil_edge_gt_flat, B_idx_pupil_edge_gt, max_n_pupil_edge_gt = mask_img_2_edge_UV_flat(pupil_mask_gt, sobel_filter)

        
        # Prepare helper variables
        loss_dict = {}
        n_i = args['batch_size']*args['frames']
        n_t = UV_iris_pred.shape[1]
        invalid_uv = 0
        UV_iris_gt = []; UV_pupil_gt = []
        mask_iris_gt = torch.ones(n_i, max_n_iris_gt, dtype=torch.bool).to(dev)
        mask_pupil_gt = torch.ones(n_i, max_n_pupil_gt, dtype=torch.bool).to(dev)
        if args['loss_w_rend_pred_2_gt_edge']:
            UV_iris_edge_gt = []; UV_pupil_edge_gt = []
            mask_iris_edge_gt = torch.ones(n_i, max_n_iris_edge_gt, dtype=torch.bool).to(dev)
            mask_pupil_edge_gt = torch.ones(n_i, max_n_pupil_edge_gt, dtype=torch.bool).to(dev)
        
        # Repack the flattened GT UV positions into (B, N_max, 2) tensor
        # N_max is the maximal contained row, which means other rows are padded to N_max length
        # A mask is also saved (B, N_Max). This mask differentiates original and padded UV values in each row.  
        for i in range(n_i):
            # FIXME Is it possible to vectorize this?
            UV_iris_gt_i = UV_iris_gt_flat[B_idx_iris_gt==i]
            UV_iris_gt_i, cnt_iris_gt_i = pad_UV_i(UV_iris_gt_i, max_n_iris_gt, invalid_uv)
            UV_iris_gt.append(UV_iris_gt_i)
            mask_iris_gt[i, cnt_iris_gt_i:] = 0

            UV_pupil_gt_i = UV_pupil_gt_flat[B_idx_pupil_gt==i]
            UV_pupil_gt_i, cnt_pupil_gt_i = pad_UV_i(UV_pupil_gt_i, max_n_pupil_gt, invalid_uv)
            UV_pupil_gt.append(UV_pupil_gt_i)
            mask_pupil_gt[i, cnt_pupil_gt_i:] = 0

            # The same as above is calculated for GT edge UV positions
            if args['loss_w_rend_pred_2_gt_edge']:
                UV_iris_edge_gt_i = UV_iris_edge_gt_flat[B_idx_iris_edge_gt==i]
                UV_iris_edge_gt_i, cnt_iris_edge_gt_i = pad_UV_i(UV_iris_edge_gt_i, max_n_iris_edge_gt, invalid_uv)
                UV_iris_edge_gt.append(UV_iris_edge_gt_i)
                mask_iris_edge_gt[i, cnt_iris_edge_gt_i:] = 0

                UV_pupil_edge_gt_i = UV_pupil_edge_gt_flat[B_idx_pupil_edge_gt==i]
                UV_pupil_edge_gt_i, cnt_pupil_edge_gt_i = pad_UV_i(UV_pupil_edge_gt_i, max_n_pupil_edge_gt, invalid_uv)
                UV_pupil_edge_gt.append(UV_pupil_edge_gt_i)
                mask_pupil_edge_gt[i, cnt_pupil_edge_gt_i:] = 0

        # Finally packing into (B, N_max, 2)
        UV_iris_gt = torch.stack(UV_iris_gt)
        UV_pupil_gt = torch.stack(UV_pupil_gt)
        if args['loss_w_rend_pred_2_gt_edge']:
            UV_iris_edge_gt = torch.stack(UV_iris_edge_gt)
            UV_pupil_edge_gt = torch.stack(UV_pupil_edge_gt)

        # Do only distance calculation here, without gradients.
        # This is computationally much cheaper to run on all pairs.
        # Later, this will be used to sub-sample only closest pairs, 
        # and calculate proper Euclidean distance (wth gradients) there.
        invalid_dist = 1e9
        dist_iris = (UV_iris_pred.detach().unsqueeze(1) - UV_iris_gt.unsqueeze(2)).square().sum(-1)
        dist_iris = dist_iris + invalid_dist*(mask_iris_gt == 0).unsqueeze(-1)
        dist_pupil = (UV_pupil_pred.detach().unsqueeze(1) - UV_pupil_gt.unsqueeze(2)).square().sum(-1)
        dist_pupil = dist_pupil + invalid_dist*(mask_pupil_gt == 0).unsqueeze(-1)

        # Same distance preparation for edges.
        if args['loss_w_rend_pred_2_gt_edge']:
            dist_iris_edge = (UV_iris_edge_pred.detach().unsqueeze(1) - UV_iris_edge_gt.unsqueeze(2)).square().sum(-1)
            dist_iris_edge = dist_iris_edge + invalid_dist*(mask_iris_edge_gt == 0).unsqueeze(-1)
            dist_pupil_edge = (UV_pupil_edge_pred.detach().unsqueeze(1) - UV_pupil_edge_gt.unsqueeze(2)).square().sum(-1)
            dist_pupil_edge = dist_pupil_edge + invalid_dist*(mask_pupil_edge_gt == 0).unsqueeze(-1)

    if args['loss_w_rend_gt_2_pred']:
        gt_2_pred_iris_idx = torch.argmin(dist_iris, dim=2).unsqueeze(-1).repeat(1,1,2)
        UV_iris_pred_closest = torch.gather(UV_iris_pred, dim=1, index=gt_2_pred_iris_idx)
        loss_iris_gt_2_pred = dist_loss(UV_iris_pred_closest[mask_iris_gt], UV_iris_gt[mask_iris_gt])

        gt_2_pred_pupil_idx = torch.argmin(dist_pupil, dim=2).unsqueeze(-1).repeat(1,1,2)
        UV_pupil_pred_closest = torch.gather(UV_pupil_pred, dim=1, index=gt_2_pred_pupil_idx)
        loss_pupil_gt_2_pred = dist_loss(UV_pupil_pred_closest[mask_pupil_gt], UV_pupil_gt[mask_pupil_gt])

        loss_dict['iris_gt_2_pred'] = loss_iris_gt_2_pred * args['loss_w_rend_gt_2_pred']
        loss_dict['pupil_gt_2_pred'] = loss_pupil_gt_2_pred * args['loss_w_rend_gt_2_pred']

    if args['loss_w_rend_pred_2_gt']:
        pred_2_gt_iris_idx = torch.argmin(dist_iris, dim=1).unsqueeze(-1).repeat(1,1,2)
        UV_iris_gt_closest = torch.gather(UV_iris_gt, dim=1, index=pred_2_gt_iris_idx)
        loss_iris_pred_2_gt = dist_loss(UV_iris_gt_closest, UV_iris_pred)

        pred_2_gt_pupil_idx = torch.argmin(dist_pupil, dim=1).unsqueeze(-1).repeat(1,1,2)
        UV_pupil_gt_closest = torch.gather(UV_pupil_gt, dim=1, index=pred_2_gt_pupil_idx)
        loss_pupil_pred_2_gt = dist_loss(UV_pupil_gt_closest, UV_pupil_pred)

        loss_dict['iris_pred_2_gt'] = loss_iris_pred_2_gt * args['loss_w_rend_pred_2_gt']
        loss_dict['pupil_pred_2_gt'] = loss_pupil_pred_2_gt * args['loss_w_rend_pred_2_gt']
    
    if args['loss_w_rend_pred_2_gt_edge']:
        pred_2_gt_iris_edge_idx = torch.argmin(dist_iris_edge, dim=1).unsqueeze(-1).repeat(1,1,2)
        UV_iris_edge_gt_closest = torch.gather(UV_iris_edge_gt, dim=1, index=pred_2_gt_iris_edge_idx)
        loss_pred_2_gt_iris_edge = dist_loss(UV_iris_edge_gt_closest, UV_iris_edge_pred)

        pred_2_gt_pupil_edge_idx = torch.argmin(dist_pupil_edge, dim=1).unsqueeze(-1).repeat(1,1,2)
        UV_pupil_edge_gt_closest = torch.gather(UV_pupil_edge_gt, dim=1, index=pred_2_gt_pupil_edge_idx)
        loss_pred_2_gt_pupil_edge = dist_loss(UV_pupil_edge_gt_closest, UV_pupil_edge_pred)

        loss_dict['iris_pred_2_gt_edge'] = loss_pred_2_gt_iris_edge * args['loss_w_rend_pred_2_gt_edge']
        loss_dict['pupil_pred_2_gt_edge'] = loss_pred_2_gt_pupil_edge * args['loss_w_rend_pred_2_gt_edge']

    total_loss = 0.0
    for k in loss_dict:
        total_loss += loss_dict[k]

    return total_loss, loss_dict
    # #--------------------------------------------------------------------------
    # #                              DEBUGGING (should go above loss components)
    # #--------------------------------------------------------------------------
    # UV_iris_pred = UV_iris_pred.contiguous()
    # UV_pupil_pred = UV_pupil_pred.contiguous()  
    # i = 2
    # # Extract GT iris & pupil locations
    # y, x = torch.where(gt_mask[i]==iris_idx)
    # UV_iris_gt_i = torch.stack((x,y), axis=1).float()
    # y, x = torch.where(gt_mask[i]==pupil_idx)
    # UV_pupil_gt_i = torch.stack((x,y), axis=1).float()

    # if args['loss_w_rend_gt_2_pred']:
    #     loss_iris_gt_2_pred, \
    #         UV_iris_pred_nearest_i = nearest_target_distance(P_source=UV_iris_gt_i, 
    #                                                         P_target=UV_iris_pred[i], 
    #                                                         gpu_resources=faiss_gpu_res)
        
    #     loss_tmp_1 = (UV_iris_gt_i - UV_iris_pred_nearest_i).square().sum(-1).sqrt().mean()

    #     loss_pupil_gt_2_pred, \
    #         UV_pupil_pred_nearest_i = nearest_target_distance(P_source=UV_pupil_gt_i, 
    #                                                         P_target=UV_pupil_pred[i], 
    #                                                         gpu_resources=faiss_gpu_res)
    #     loss_tmp_2 = (UV_pupil_gt_i - UV_pupil_pred_nearest_i).square().sum(-1).sqrt().mean()


    #     tmp = (UV_iris_pred[i].detach().unsqueeze(0) - UV_iris_gt_i.unsqueeze(1)).square().sum(-1).sqrt()
    #     tmp = UV_iris_pred[i][torch.argmin(tmp, dim=1)]
    #     print(torch.abs(UV_iris_pred_nearest_i - tmp).max())
    #     print((torch.abs(UV_iris_pred_nearest_i - tmp) > 0).sum())

    #     loss_tmp_1_1 = dist_loss(UV_iris_pred_closest[i][mask_iris_gt[i]], UV_iris_gt[i][mask_iris_gt[i]])
    #     loss_tmp_1_2 = dist_loss(UV_iris_pred_nearest_i, UV_iris_gt_i)
    # #--------------------------------------------------------------------------
    # #--------------------------------------------------------------------------


def rendered_semantics_loss(gt_mask, rend_dict, sobel_filter, faiss_gpu_res, args):

    iris_idx = 1
    pupil_idx = 2

    UV_pupil_pred = rend_dict['pupil_UV']
    UV_iris_pred = rend_dict['iris_UV']
    side_idx_iris = rend_dict['edge_idx_iris']
    side_idx_pupil = rend_dict['edge_idx_pupil']

    assert gt_mask.shape[0] == UV_pupil_pred.shape[0] == UV_iris_pred.shape[0]
    assert UV_iris_pred.shape[2] == 2, print('iris not valid number of features')
    assert UV_pupil_pred.shape[2] == 2, print('pupil not valid number of features')

    if args['loss_w_rend_pred_2_gt_edge']:
        # Extract GT iris & pupil edge mask (using sobel filtering)
        iris_mask_gt = torch.where(gt_mask>=iris_idx, 1., 0.) # TODO == iris_idx
        iris_edge_gt = sobel_filter(iris_mask_gt.unsqueeze(1)).squeeze(1)
        iris_edge_gt = torch.where(iris_edge_gt > 0, 1, 0)
        pupil_mask_gt = torch.where(gt_mask==pupil_idx, 1., 0.)
        pupil_edge_gt = sobel_filter(pupil_mask_gt.unsqueeze(1)).squeeze(1)
        pupil_edge_gt = torch.where(pupil_edge_gt > 0, 1, 0) 

    UV_iris_pred = UV_iris_pred.contiguous()
    UV_pupil_pred = UV_pupil_pred.contiguous()   

    #TODO REMOVE THIS FOR LOOP AFTER TESTING
    loss_dict = {}
    n_i = args['batch_size']*args['frames']
    n_t = UV_iris_pred.shape[1]
    for i in range(n_i):

        # Extract GT iris & pupil locations
        y, x = torch.where(gt_mask[i]==iris_idx)
        UV_iris_gt = torch.stack((x,y), axis=1).float()
        y, x = torch.where(gt_mask[i]==pupil_idx)
        UV_pupil_gt = torch.stack((x,y), axis=1).float()

        # Distance of GT to closest predictions
        if args['loss_w_rend_gt_2_pred']:
            loss_iris_gt_2_pred, \
                UV_iris_pred_nearest_i = nearest_target_distance(P_source=UV_iris_gt, 
                                                                P_target=UV_iris_pred[i], 
                                                                gpu_resources=faiss_gpu_res)

            loss_pupil_gt_2_pred, \
                UV_pupil_pred_nearest_i = nearest_target_distance(P_source=UV_pupil_gt, 
                                                                P_target=UV_pupil_pred[i], 
                                                                gpu_resources=faiss_gpu_res)
            if i == 0: loss_dict['iris_gt_2_pred'] = 0.0; loss_dict['pupil_gt_2_pred'] = 0.0
            loss_dict['iris_gt_2_pred'] += loss_iris_gt_2_pred * args['loss_w_rend_gt_2_pred']
            loss_dict['pupil_gt_2_pred'] += loss_pupil_gt_2_pred * args['loss_w_rend_gt_2_pred']

        # Distance of predictions to closest GT
        if args['loss_w_rend_pred_2_gt']:
            # New way: compare the whole predicted template 
            loss_iris_pred_2_gt,\
                UV_iris_gt_nearest_i = nearest_target_distance(P_source=UV_iris_pred[i], 
                                                                P_target=UV_iris_gt, 
                                                                gpu_resources=faiss_gpu_res)

            loss_pupil_pred_2_gt,\
                UV_pupil_gt_nearest_i = nearest_target_distance(P_source=UV_pupil_pred[i], 
                                                                P_target=UV_pupil_gt, 
                                                                gpu_resources=faiss_gpu_res)
            if i == 0: loss_dict['iris_pred_2_gt'] = 0.0; loss_dict['pupil_pred_2_gt'] = 0.0
            loss_dict['iris_pred_2_gt'] += loss_iris_pred_2_gt * args['loss_w_rend_pred_2_gt']
            loss_dict['pupil_pred_2_gt'] += loss_pupil_pred_2_gt * args['loss_w_rend_pred_2_gt']

        # Distance of prediction edge points to GT edge points
        if args['loss_w_rend_pred_2_gt_edge']:
            # Extract GT iris & pupil edge locations
            y, x = torch.where(iris_edge_gt[i]>0) 
            UV_iris_gt_edge = torch.stack((x,y), axis=1).float()
            loss_iris_pred_2_gt_edge,\
                UV_iris_gt_nearest_edge_i = nearest_target_distance(P_source=UV_iris_pred[i, side_idx_iris['sides']], 
                                                                    P_target=UV_iris_gt_edge, 
                                                                    gpu_resources=faiss_gpu_res)

            y, x = torch.where(pupil_edge_gt[i]>0) 
            UV_pupil_gt_edge = torch.stack((x,y), axis=1).float()
            loss_pupil_pred_2_gt_edge,\
                UV_pupil_gt_nearest_edge_i = nearest_target_distance(P_source=UV_pupil_pred[i, side_idx_pupil['sides']], 
                                                                     P_target=UV_pupil_gt_edge, 
                                                                     gpu_resources=faiss_gpu_res)
            if i == 0: loss_dict['iris_pred_2_gt_edge'] = 0.0; loss_dict['pupil_pred_2_gt_edge'] = 0.0
            loss_dict['iris_pred_2_gt_edge'] += loss_iris_pred_2_gt_edge * args['loss_w_rend_pred_2_gt_edge']
            loss_dict['pupil_pred_2_gt_edge'] += loss_pupil_pred_2_gt_edge * args['loss_w_rend_pred_2_gt_edge']
        
        # Diameter difference of predicted and gt
        if args['loss_w_rend_diameter']:
            MSE = torch.nn.MSELoss(reduction='mean')
            RMSE = lambda x, y: torch.sqrt(MSE(x, y))
            diameter_iris_gt = torch.max(UV_iris_gt[...,0]) - torch.min(UV_iris_gt[...,0])
            diameter_iris_pred = UV_iris_pred[i, side_idx_iris['right'][0], 0] - \
                                    UV_iris_pred[i, side_idx_iris['left'][4], 0]
            loss_iris_diameter = RMSE(diameter_iris_pred, diameter_iris_gt)
            diameter_pupil_gt = torch.max(UV_pupil_gt[...,0]) - torch.min(UV_pupil_gt[...,0])
            diameter_pupil_pred = UV_pupil_pred[i, side_idx_pupil['right'][0], 0] - \
                                    UV_pupil_pred[i, side_idx_pupil['left'][23], 0]
            loss_pupil_diameter = RMSE(diameter_pupil_pred, diameter_pupil_gt)   
            if i == 0: loss_dict['iris_diameter'] = 0.0; loss_dict['pupil_diameter'] = 0.0
            loss_dict['iris_diameter'] += loss_iris_diameter * args['loss_w_rend_diameter']
            loss_dict['pupil_diameter'] += loss_pupil_diameter * args['loss_w_rend_diameter']

    total_loss = 0.0
    for k in loss_dict:
        loss_dict[k] /= (i+1)
        total_loss += loss_dict[k]

    return total_loss, loss_dict

def loss_fn_rend_sprvs(gt_dict, pred_dict, args):
    #MSE = torch.nn.MSELoss(reduction='mean')
    #RMSE = lambda x, y: torch.sqrt(MSE(x, y))
    RMSE = lambda x, y: (x-y).square().sum(-1).sqrt().mean()
    cos_sim = lambda x , y: 1 - F.cosine_similarity(x,y).mean()
    loss_dict = {}

    device = pred_dict['eyeball_c_UV'].device

    #loss eyeball ground truth
    if args['loss_w_supervise_eyeball_center']:
        gt_eyeball_c_UV = gt_dict['eyeball'][...,1:3].to(device)
        pred_eyeball_c_UV = pred_dict['eyeball_c_UV']

        loss_eyeball_c_UV = RMSE(gt_eyeball_c_UV, pred_eyeball_c_UV) * \
                                    args['loss_w_supervise_eyeball_center']
        loss_dict['eyeball_c_UV'] = loss_eyeball_c_UV

    #loss pupil center ground_truth
    if args['loss_w_supervise_pupil_center']:
        gt_pupil_c_UV = (gt_dict['eyeball'][...,1:3] + \
                (gt_dict['eyeball'][...,0:1] * gt_dict['gaze_vector'][...,:2])).to(device)
        pred_pupil_c_UV = pred_dict['pupil_c_UV']

        loss_pupil_c_UV = RMSE(gt_pupil_c_UV, pred_pupil_c_UV) * \
                                    args['loss_w_supervise_pupil_center']
        loss_dict['pupil_c_UV'] = loss_pupil_c_UV

    if args['loss_w_supervise_gaze_vector_3D_L2']:
        gt_gaze_vector_3D = gt_dict['gaze_vector'].to(device)
        pred_gaze_vector_3D = pred_dict['gaze_vector_3D']

        loss_gaze_vector_3D = RMSE(gt_gaze_vector_3D, pred_gaze_vector_3D) * \
                                    args['loss_w_supervise_gaze_vector_3D_L2']
        loss_dict['gaze_vector_3D_L2'] = loss_gaze_vector_3D

    if args['loss_w_supervise_gaze_vector_3D_cos_sim']:
        gt_gaze_vector_3D = gt_dict['gaze_vector'].to(device)
        pred_gaze_vector_3D = pred_dict['gaze_vector_3D']

        loss_gaze_vector_3D = cos_sim(gt_gaze_vector_3D, pred_gaze_vector_3D) * \
                                    args['loss_w_supervise_gaze_vector_3D_cos_sim']
        loss_dict['gaze_vector_3D_cos_sim'] = loss_gaze_vector_3D

    if args['loss_w_supervise_gaze_vector_UV']:
        gt_eyeball_c_UV = gt_dict['eyeball'][...,1:3]
        gt_pupil_c_UV = gt_dict['eyeball'][...,1:3] + \
                (gt_dict['eyeball'][...,0:1] * gt_dict['gaze_vector'][...,:2])
        gt_gaze_vector_UV = (gt_pupil_c_UV -gt_eyeball_c_UV).to(device)
        gt_gaze_vector_UV /= torch.norm(gt_gaze_vector_UV, dim=-1, keepdim=True) + 1e-9

        temp_gaze_vector_UV = (pred_dict['pupil_c_UV'] - pred_dict['eyeball_c_UV'])
        gaze_vector_UV = temp_gaze_vector_UV / \
                (torch.norm(temp_gaze_vector_UV, dim=-1, keepdim=True) + 1e-5)
        pred_gaze_vector_UV = gaze_vector_UV

        loss_gaze_vector_UV = RMSE(gt_gaze_vector_UV, pred_gaze_vector_UV) * \
                                    args['loss_w_supervise_gaze_vector_UV']
        loss_dict['gaze_vector_UV'] = loss_gaze_vector_UV

    total_loss = 0.0
    for key in loss_dict:
        total_loss += loss_dict[key]

    return total_loss, loss_dict


def rendered_semantics_loss_OLD_01_05(gt_mask, rend_dict, sobel_filter, faiss_gpu_res, args):

    UV_pupil_pred = rend_dict['pupil_UV']
    UV_iris_pred = rend_dict['iris_UV']
    side_idx_iris = rend_dict['edge_idx_iris']
    side_idx_pupil = rend_dict['edge_idx_pupil']

    assert gt_mask.shape[0] == UV_pupil_pred.shape[0] == UV_iris_pred.shape[0]
    assert UV_iris_pred.shape[1] == 5000, print('iris not valid number of points')
    assert UV_pupil_pred.shape[1] == 5000, print('pupil not valid number of points')
    assert UV_iris_pred.shape[2] == 2, print('iris not valid number of features')
    assert UV_pupil_pred.shape[2] == 2, print('pupil not valid number of features')

    iris_idx = 1
    pupil_idx = 2
    
    # Extract GT iris & pupil edge mask (using sobel filtering)
    iris_mask_gt = torch.where(gt_mask>=iris_idx, 1., 0.) # TODO == iris_idx
    iris_edge_gt = sobel_filter(iris_mask_gt.unsqueeze(1)).squeeze(1)
    iris_edge_gt = torch.where(iris_edge_gt > 0, 1, 0)
    pupil_mask_gt = torch.where(gt_mask==pupil_idx, 1., 0.)
    pupil_edge_gt = sobel_filter(pupil_mask_gt.unsqueeze(1)).squeeze(1)
    pupil_edge_gt = torch.where(pupil_edge_gt > 0, 1, 0)

    UV_iris_pred = UV_iris_pred.contiguous()
    UV_pupil_pred = UV_pupil_pred.contiguous()

    #TODO REMOVE THIS FOR LOOP AFTER TESTING
    loss_dict = {}
    n_i = args['batch_size']*args['frames']
    n_t = UV_iris_pred.shape[1]
    for i in range(args['batch_size']*args['frames']):
        # Extract GT iris & pupil edge locations
        y, x = torch.where(iris_edge_gt[i]>0) 
        UV_iris_gt_edge = torch.stack((x,y), axis=1).float()

        # %%Split the iris edge to left and right for better loss computation
        #Define left point of edge
        left_location = UV_iris_gt_edge[...,0].min()
        right_location = UV_iris_gt_edge[...,0].max()
        middle_location = torch.round((left_location + right_location) / 2)

        #sort the tensor based on x axis
        _, idx = torch.sort(UV_iris_gt_edge, axis=0)
        UV_iris_gt_edge_sorted_x = UV_iris_gt_edge[idx[...,0]]

        #find the index of the middle location
        middle_points = torch.where(UV_iris_gt_edge_sorted_x[...,0] == middle_location)

        #split the iris edge to left and right
        UV_left_iris_gt_edge = UV_iris_gt_edge_sorted_x[0:middle_points[0][0]]
        UV_right_iris_gt_edge = UV_iris_gt_edge_sorted_x[middle_points[0][0]:]

        y, x = torch.where(pupil_edge_gt[i]>0) 
        UV_pupil_gt_edge = torch.stack((x,y), axis=1).float()

        # %%Split the pupil edge to left and right for better loss computation
        #Define left point of edge
        left_location = UV_pupil_gt_edge[...,0].min()
        right_location = UV_pupil_gt_edge[...,0].max()
        middle_location = torch.round((left_location + right_location) / 2)

        #sort the tensor based on x axis
        _, idx = torch.sort(UV_pupil_gt_edge, axis=0)
        UV_pupil_gt_edge_sorted_x = UV_pupil_gt_edge[idx[...,0]]

        #find the index of the middle location
        middle_points = torch.where(UV_pupil_gt_edge_sorted_x[...,0] == middle_location)

        #split the iris edge to left and right
        UV_left_pupil_gt_edge = UV_pupil_gt_edge_sorted_x[0:middle_points[0][0]]
        UV_right_pupil_gt_edge = UV_pupil_gt_edge_sorted_x[middle_points[0][0]:]

        # Extract GT iris & pupil locations
        y, x = torch.where(gt_mask[i]==iris_idx)
        UV_iris_gt = torch.stack((x,y), axis=1).float()
        y, x = torch.where(gt_mask[i]==pupil_idx)
        UV_pupil_gt = torch.stack((x,y), axis=1).float()

        # Distance of GT to closest predictions
        if args['loss_w_rend_gt_2_pred']:
            loss_iris_gt_2_pred, \
                UV_iris_pred_nearest_i = nearest_target_distance(P_source=UV_iris_gt, 
                                                                P_target=UV_iris_pred[i], 
                                                                gpu_resources=faiss_gpu_res)

            loss_pupil_gt_2_pred, \
                UV_pupil_pred_nearest_i = nearest_target_distance(P_source=UV_pupil_gt, 
                                                                P_target=UV_pupil_pred[i], 
                                                                gpu_resources=faiss_gpu_res)
            if i == 0: loss_dict['iris_gt_2_pred'] = 0.0; loss_dict['pupil_gt_2_pred'] = 0.0
            loss_dict['iris_gt_2_pred'] += loss_iris_gt_2_pred * args['loss_w_rend_gt_2_pred']
            loss_dict['pupil_gt_2_pred'] += loss_pupil_gt_2_pred * args['loss_w_rend_gt_2_pred']

        # Distance of predictions to closest GT
        if args['loss_w_rend_pred_2_gt']:
            # New way: compare the whole predicted template 
            loss_iris_pred_2_gt,\
                UV_iris_gt_nearest_i = nearest_target_distance(P_source=UV_iris_pred[i], 
                                                                P_target=UV_iris_gt, 
                                                                gpu_resources=faiss_gpu_res)

            loss_pupil_pred_2_gt,\
                UV_pupil_gt_nearest_i = nearest_target_distance(P_source=UV_pupil_pred[i], 
                                                                P_target=UV_pupil_gt, 
                                                                gpu_resources=faiss_gpu_res)

            if i == 0: loss_dict['iris_pred_2_gt'] = 0.0; loss_dict['pupil_pred_2_gt'] = 0.0
            loss_dict['iris_pred_2_gt'] += loss_iris_pred_2_gt * args['loss_w_rend_pred_2_gt']
            loss_dict['pupil_pred_2_gt'] += loss_pupil_pred_2_gt * args['loss_w_rend_pred_2_gt']
        
        # Distance of prediction edge points to GT edge points
        if args['loss_w_rend_pred_2_gt_edge']:
            loss_iris_pred_2_gt_edge,\
                UV_iris_gt_nearest_edge_i = nearest_target_distance(P_source=UV_iris_pred[i, side_idx_iris['whole']], 
                                                                P_target=UV_left_iris_gt_edge, 
                                                                gpu_resources=faiss_gpu_res)
            
            loss_pupil_pred_2_gt_edge,\
                UV_pupil_gt_nearest_edge_i = nearest_target_distance(P_source=UV_pupil_pred[i, side_idx_pupil['whole']], 
                                                                P_target=UV_left_pupil_gt_edge, 
                                                                gpu_resources=faiss_gpu_res)
            
            if i == 0: loss_dict['iris_pred_2_gt_edge'] = 0.0; loss_dict['pupil_pred_2_gt_edge'] = 0.0
            loss_dict['iris_pred_2_gt_edge'] += loss_iris_pred_2_gt_edge * args['loss_w_rend_pred_2_gt_edge']
            loss_dict['pupil_pred_2_gt_edge'] += loss_pupil_pred_2_gt_edge * args['loss_w_rend_pred_2_gt_edge']

        # Diameter difference of predicted and gt
        if args['loss_w_rend_diameter']:
            MSE = torch.nn.MSELoss(reduction='mean')
            RMSE = lambda x, y: torch.sqrt(MSE(x, y))
            diameter_iris_gt = torch.max(UV_iris_gt[...,0]) - torch.min(UV_iris_gt[...,0])
            diameter_iris_pred = UV_iris_pred[i, side_idx_iris['right'][0], 0] - \
                                    UV_iris_pred[i, side_idx_iris['left'][4], 0]
            loss_iris_diameter = RMSE(diameter_iris_pred, diameter_iris_gt)
            diameter_pupil_gt = torch.max(UV_pupil_gt[...,0]) - torch.min(UV_pupil_gt[...,0])
            diameter_pupil_pred = UV_pupil_pred[i, side_idx_pupil['right'][0], 0] - \
                                    UV_pupil_pred[i, side_idx_pupil['left'][23], 0]
            loss_pupil_diameter = RMSE(diameter_pupil_pred, diameter_pupil_gt)   
            if i == 0: loss_dict['iris_diameter'] = 0.0; loss_dict['pupil_diameter'] = 0.0
            loss_dict['iris_diameter'] += loss_iris_diameter * args['loss_w_rend_diameter']
            loss_dict['pupil_diameter'] += loss_pupil_diameter * args['loss_w_rend_diameter']

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(3, 2)
        # axs[0,0].scatter(UV_iris_gt[:, 0].detach(), UV_iris_gt[:, 1].detach(), color='b', marker='*', label='source')
        # axs[0,0].scatter(UV_iris_pred[i][:, 0].detach(), UV_iris_pred[i][:, 1].detach(), color='g', marker='^', label='target')
        # axs[0,0].scatter(UV_iris_gt[0, 0].detach(), UV_iris_gt[0, 1].detach(), color='r', marker='*', label='source')
        # axs[0,0].scatter(UV_iris_pred_nearest_i[0, 0].detach(), UV_iris_pred_nearest_i[0, 1].detach(), color='r', marker='^', label='target')     
        # axs[0,1].scatter(UV_pupil_gt[:, 0].detach(), UV_pupil_gt[:, 1].detach(), color='b', marker='*', label='source')
        # axs[0,1].scatter(UV_pupil_pred[i][:, 0].detach(), UV_pupil_pred[i][:, 1].detach(), color='g', marker='^', label='target')
        # axs[0,1].scatter(UV_pupil_gt[0, 0].detach(), UV_pupil_gt[0, 1].detach(), color='r', marker='*', label='source')
        # axs[0,1].scatter(UV_pupil_pred_nearest_i[0, 0].detach(), UV_pupil_pred_nearest_i[0, 1].detach(), color='r', marker='^', label='target')
        # axs[0,0].title.set_text('Distance of GT to closest predictions')
        # axs[0,1].title.set_text('Distance of GT to closest predictions')
        # axs[1,0].scatter(UV_iris_gt[:, 0].detach(), UV_iris_gt[:, 1].detach(), color='g', marker='^', label='target')
        # axs[1,0].scatter(UV_iris_pred[i][:, 0].detach(), UV_iris_pred[i][:, 1].detach(), color='b', marker='*', label='source')
        # axs[1,0].scatter(UV_iris_pred[i][0, 0].detach(), UV_iris_pred[i][0, 1].detach(), color='r', marker='*', label='source')
        # axs[1,0].scatter(UV_iris_gt_nearest_i[0, 0].detach(), UV_iris_gt_nearest_i[0, 1].detach(), color='r', marker='^', label='target')
        # axs[1,1].scatter(UV_pupil_gt[:, 0].detach(), UV_pupil_gt[:, 1].detach(), color='g', marker='^', label='target')
        # axs[1,1].scatter(UV_pupil_pred[i][:, 0].detach(), UV_pupil_pred[i][:, 1].detach(), color='b', marker='*', label='source')
        # axs[1,1].scatter(UV_pupil_pred[i][0, 0].detach(), UV_pupil_pred[i][0, 1].detach(), color='r', marker='*', label='source')
        # axs[1,1].scatter(UV_pupil_gt_nearest_i[0, 0].detach(), UV_pupil_gt_nearest_i[0, 1].detach(), color='r', marker='^', label='target')
        # axs[1,0].title.set_text('Distance of predictions to closest GT')
        # axs[1,1].title.set_text('Distance of predictions to closest GT')
        # axs[2,0].scatter(UV_iris_pred[i, side_idx_iris][:, 0].detach(), UV_iris_pred[i, side_idx_iris][:, 1].detach(), color='b', marker='*', label='source')
        # axs[2,0].scatter(UV_iris_gt_edge[:, 0].detach(), UV_iris_gt_edge[:, 1].detach(), color='g', marker='^', label='target')
        # axs[2,0].scatter(UV_iris_pred[i, side_idx_iris][0, 0].detach(), UV_iris_pred[i, side_idx_iris][0, 1].detach(), color='r', marker='*', label='source')
        # axs[2,0].scatter(UV_iris_gt_nearest_edge_i[0, 0].detach(), UV_iris_gt_nearest_edge_i[0, 1].detach(), color='r', marker='^', label='target')
        # axs[2,1].scatter(UV_pupil_pred[i, side_idx_pupil][:, 0].detach(), UV_pupil_pred[i, side_idx_pupil][:, 1].detach(), color='b', marker='*', label='source')
        # axs[2,1].scatter(UV_pupil_gt_edge[:, 0].detach(), UV_pupil_gt_edge[:, 1].detach(), color='g', marker='^', label='target')
        # axs[2,1].scatter(UV_pupil_pred[i, side_idx_pupil][0, 0].detach(), UV_pupil_pred[i, side_idx_pupil][0, 1].detach(), color='r', marker='*', label='source')
        # axs[2,1].scatter(UV_pupil_gt_nearest_edge_i[0, 0].detach(), UV_pupil_gt_nearest_edge_i[0, 1].detach(), color='r', marker='^', label='target')
        # axs[2,0].title.set_text('Distance of prediction edge points to GT edge points')
        # axs[2,1].title.set_text('Distance of prediction edge points to GT edge points')
        # plt.legend()
        # plt.show() 


    total_loss = 0.0
    for k in loss_dict:
        loss_dict[k] /= (i+1)
        total_loss += loss_dict[k]

    return total_loss, loss_dict