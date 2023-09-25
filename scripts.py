#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:38:59 2021

@author: rakshit
"""

#try git

import os
import gc
import sys
import time
import h5py
import math
from collections import OrderedDict

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np
import faiss
import wandb

import cv2

from einops import rearrange

from helperfunctions.loss import get_seg_loss, get_uncertain_l1_loss
from helperfunctions.helperfunctions import assert_torch_invalid
# from helperfunctions.loss import get_l2c_loss

from helperfunctions.helperfunctions import plot_images_with_annotations
from helperfunctions.helperfunctions import convert_to_list_entries
from helperfunctions.helperfunctions import merge_two_dicts, fix_batch
from helperfunctions.helperfunctions import generate_rend_masks

from helperfunctions.utils import get_seg_metrics, get_distance, compute_norm
from helperfunctions.utils import getAng_metric, generate_pseudo_labels
from helperfunctions.utils import remove_underconfident_psuedo_labels

from Visualitation_TEyeD.gaze_estimation import generate_gaze_gt 

from rendering.rendering import render_semantics, euler_to_rotation, eyeball_center
from rendering.rendered_semantics_loss import rendered_semantics_loss, rendered_semantics_loss_vectorized, SobelFilter

from rendering.rendered_semantics_loss import loss_fn_rend_sprvs

def detach_cpu_numpy(data_dict):
    out_dict = {}
    for key, value in data_dict.items():
        if 'torch' in str(type(value)):
            out_dict[key] = value.detach().cpu().numpy()
        else:
            out_dict[key] = value
    return out_dict

def move_gpu(data_dict,device):
    out_dict = {}
    for key, value in data_dict.items():
        if 'torch' in str(type(value)):
            out_dict[key] = data_dict[key].to(device)
        else:
            out_dict[key] = data_dict[key]
    return out_dict


def send_to_device(data_dict, device):
    out_dict = {}
    for key, value in data_dict.items():
        if 'torch' in str(type(value)):
            out_dict[key] = value.to(device)
        else:
            out_dict[key] = value
    return out_dict


def forward(net,
            spiker,
            logger,
            loader,
            optimizer,
            args,
            path_dict,
            epoch=0,
            mode='test',
            writer=[],
            rank_cond=False,
            optimizer_disc=False,
            batches_per_ep=2000,
            last_epoch_valid = False,
            csv_save_dir=None):

    net_param_tmp = next(net.parameters())
    device = net_param_tmp.device

    if net_param_tmp.is_cuda:
        print('Using faiss GPU resource.')
        faiss_gpu_res = faiss.StandardGpuResources()  # use a single GPU
    else:
        print('Using faiss CPU resource.')
        faiss_gpu_res = None
    sobel_filter = SobelFilter(device) #.get_device()

    logger.write('{}. Epoch: {}'.format(mode, epoch))

    #deactivate tensorboard
    #rank_cond = ((args['local_rank'] == 0) or not args['do_distributed'])
    rank_cond = rank_cond

    if mode == 'train':
        net.train()
    else:
        net.eval()

    io_time = []
    loader_iter = iter(loader)

    metrics = []
    dataset_id = []
    embeddings = []
    available_predicted_mask =False

    train_with_mask = args['loss_w_rend_pred_2_gt_edge'] or args['loss_w_rend_gt_2_pred'] \
                        or args['loss_w_rend_pred_2_gt']

    if (mode == 'test') and args['save_test_maps']:
        logger.write('Generating test object')
        test_results_obj = h5py.File(path_dict['results']+'/test_results.h5',
                                     'w', swmr=True)

    for bt in range(batches_per_ep):

        start_time = time.time()

        try:
            data_dict = next(loader_iter)
        except:
            print('Loader reset')
            loader_iter = iter(loader)
            data_dict = next(loader_iter)
            args['time_to_update'] = True

        if torch.any(data_dict['is_bad']):
            logger.write('Bad batch found!', do_warn=True)

            # DDP crashes if we skip over a batch and no gradients are matched
            # To avoid this, remove offending samples by replacing it with a
            # good sample randomly drawn from rest of the batch
            data_dict = fix_batch(data_dict)

        end_time = time.time()
        io_time.append(end_time - start_time)

        if args['do_distributed']:
            torch.distributed.barrier()

        if args['cur_obj'] !='Ours':
            assert torch.all(data_dict['pupil_ellipse'][:, :, -1] >= 0), \
                'pupil ellipse orientation >= 0'
            assert torch.all(data_dict['pupil_ellipse'][:, :, -1] <= 2*(np.pi)), \
                'pupil ellipse orientation <= 2*pi'
            assert torch.all(data_dict['iris_ellipse'][:, :, -1] >= 0), \
                'iris ellipse orientation >= 0'
            assert torch.all(data_dict['iris_ellipse'][:, :, -1] <= 2*(np.pi)), \
                'iris ellipse orientation <= 2*pi'

        with torch.autograd.set_detect_anomaly(bool(args['detect_anomaly'])):
            with torch.cuda.amp.autocast(enabled=bool(args['mixed_precision'])):

                batch_results_rend = {}
                batch_results_ellseg = {}

                # Change behavior in train and test mode
                if mode == 'train':
                    out_dict, out_dict_valid = net(data_dict, args)
                else:
                    with torch.no_grad():
                        out_dict, out_dict_valid = net(data_dict, args)

                if not out_dict_valid:
                    # Zero out gradients no matter what
                    optimizer.zero_grad()
                    net.zero_grad()
                    print('skip that batch')
                    continue

                #check if the predicted values for rendering are inside the range
                if  torch.all(data_dict['image']==0):
                    optimizer.zero_grad()
                    net.zero_grad()
                    print('invalid input image')
                    continue
                
                # Remove time to process each frame as it is not needed
                out_dict.pop('dT')

                #reshape tensor to merge the batch and frame to one dimension (B,F) 
                data_dict = reshape_gt(data_dict, args)
                out_dict = reshape_ellseg_out(out_dict, args)

                batch_size = args['batch_size']
                frames = args['frames']

                H = data_dict['image'].shape[1]
                W = data_dict['image'].shape[2]

                image_resolution_diagonal = math.sqrt(H**2 + W**2)

                if args['net_rend_head']:
                    #Add the rendering process to generate the point cloud for pupil and iris

                    if torch.any(torch.any(out_dict['T'] < -1) or torch.any(out_dict['T'] > 1)) or \
                        torch.any(torch.any(out_dict['R'] < -1) or torch.any(out_dict['R'] > 1)) or \
                        torch.any(torch.any(out_dict['L'] < -1) or torch.any(out_dict['L'] > 1)) or \
                        torch.any(torch.any(out_dict['focal'] < -1) or torch.any(out_dict['focal'] > 1)) or \
                        torch.any(torch.any(out_dict['r_pupil'] < -1) or torch.any(out_dict['r_pupil'] > 1)) or \
                        torch.any(torch.any(out_dict['r_iris'] < -1) or torch.any(out_dict['r_iris'] > 1)):

                        optimizer.zero_grad()
                        net.zero_grad()
                        print('invalid predicted values from rend head')
                        continue

                    if torch.isnan(out_dict['T']).any(): 
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('NaN problemT BEFORE FUNCTION')
                        continue
                    if torch.isinf(out_dict['T']).any(): 
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('inf problem T inf before function')
                        continue

                    if torch.isnan(out_dict['R']).any(): 
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('NaN problem R BEFORE FUNCTION')
                        continue
                    if torch.isinf(out_dict['R']).any(): 
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('inf problem R inf before function')
                        continue

                    out_dict, rend_dict = render_semantics(out_dict, H=H, W=W, args=args, data_dict=data_dict)

                    if (torch.isnan(rend_dict['pupil_UV']).any() or torch.isinf(rend_dict['pupil_UV']).any()):
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('invalid pupil from rendering points')
                        continue

                    if (torch.isnan(rend_dict['iris_UV']).any() or torch.isinf(rend_dict['iris_UV']).any()):
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('invalid iris from rendering points')
                        continue

                    if (torch.isnan(rend_dict['pupil_c_UV']).any() or torch.isinf(rend_dict['pupil_c_UV']).any()):
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('invalid pupil center')
                        continue

                    if (torch.isnan(rend_dict['eyeball_c_UV']).any() or torch.isinf(rend_dict['eyeball_c_UV']).any()):
                        optimizer.zero_grad()
                        net.zero_grad()
                        print('invalid eyeball center')
                        continue

                    data_dict = send_to_device(data_dict, device)

                    if train_with_mask:
                        if args['loss_rend_vectorized']:
                            loss_fn_rend = rendered_semantics_loss_vectorized
                        else:
                            loss_fn_rend = rendered_semantics_loss

                        total_loss_rend, loss_dict_rend = loss_fn_rend(data_dict['mask'], 
                                                                                    rend_dict,
                                                                                    sobel_filter,
                                                                                    faiss_gpu_res, 
                                                                                    args)

                        
                        # TODO : This shoul be vectorized and the for loop should be removed
                        # TODO What does this rendering_mask do?
                        #3 channels / channel 1 = iris / channel 2 = pupil
                        #for predicted mask based on the 3D eyeball params
                        # From 2D image to segmentation mask
                        iterations = args['batch_size'] * args['frames']
                        if (bt % args['produce_rend_mask_per_iter'] == 0 or last_epoch_valid \
                            or (mode == 'test')):

                            #print(out_dict['T'], out_dict['R'], out_dict['r_pupil'],
                            #        out_dict['r_iris'], out_dict['L'], out_dict['focal'])

                            available_predicted_mask = True

                            rend_dict['eyeball_circle'] = eyeball_center(out_dict, 
                                                                        H=H, 
                                                                        W=W, 
                                                                        args=args)
                            
                            rend_dict = generate_rend_masks(rend_dict,
                                                        H, W, iterations)

                            rend_dict['mask'] = torch.argmax(rend_dict['predict'], 
                                                            dim=1)

                            rend_dict['gaze_img'] = rend_dict['mask_gaze']

                            rend_dict['mask'] = torch.clamp(rend_dict['mask'], min=0, max=255)
                            rend_dict['gaze_img'] = torch.clamp(rend_dict['gaze_img'], min=0, max=255)
                            
                            rend_dict['mask'] = rend_dict['mask'].detach().cpu().numpy()
                            rend_dict['gaze_img'] = rend_dict['gaze_img'].detach().cpu().numpy()
                        else:
                            available_predicted_mask=False
                        
                        if torch.is_tensor(total_loss_rend):
                            total_loss_rend_value = total_loss_rend.item()  
                            is_spike = spiker.update(total_loss_rend_value) if spiker else False
                        else:
                            total_loss_rend_value = total_loss_rend

                        batch_results_rend = get_metrics_rend(detach_cpu_numpy(rend_dict),
                                                            detach_cpu_numpy(data_dict),
                                                            batch_results_rend,
                                                            image_resolution_diagonal,
                                                            args,
                                                            available_predicted_mask)
                        batch_results_rend['loss/rend_total'] = total_loss_rend_value
                        for k in loss_dict_rend:
                            batch_results_rend[f'loss/rend_{k}'] = loss_dict_rend[k].item()
                    else:
                        total_loss_rend = 0.0
                else: 
                    total_loss_rend = 0.0
                    total_loss_rend_value = 0.0
                    loss_dict_rend = {}
                    rend_dict = {}

                #add loss in case we want to supervise the 3D Eye model or directly the UV point
                if args['loss_w_supervise']:
                    if args['net_rend_head']:
                        total_supervised_loss, loss_dict_supervised = loss_fn_rend_sprvs(data_dict,
                                                                                        rend_dict,
                                                                                        args)
                        batch_results_rend[f'loss/eye_total'] = total_supervised_loss.item()
                        for k in loss_dict_supervised:
                            batch_results_rend[f'loss/eye_{k}'] = loss_dict_supervised[k].item()

                    elif args['net_simply_head']:
                        total_supervised_loss, loss_dict_supervised = loss_fn_rend_sprvs(data_dict,
                                                                                        out_dict,
                                                                                        args)
                        batch_results_rend[f'loss/eye_total'] = total_supervised_loss.item()
                        for k in loss_dict_supervised:
                            batch_results_rend[f'loss/eye_{k}'] = loss_dict_supervised[k].item()


                if args['net_ellseg_head'] and args['loss_w_ellseg']:
                    try:
                        total_loss_ellseg, disc_loss, loss_dict_ellseg = get_loss_ellseg(send_to_device(data_dict, device),
                                                                    out_dict,
                                                                    float(args['alpha']),
                                                                    float(args['beta']),
                                                                    adv_loss=args['adv_DG'],
                                                                    bias_removal=args['grad_rev'],
                                                                    make_aleatoric=args['make_aleatoric'],
                                                                    pseudo_labels=args['pseudo_labels'],
                                                                    regress_loss=args['regression_from_latent'])

                        total_loss_ellseg_value = total_loss_ellseg.item()
                    except Exception as e:

                        print(e)

                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

                        #  Something broke during loss computation, safe skip this
                        print('Something broke during loss computation')
                        print(data_dict['archName'])
                        print(data_dict['im_num'])

                        print('--------------- Skipping this batch from training ---------------')

                        # Zero out gradients
                        optimizer.zero_grad()
                        net.zero_grad()

                        # del out_dict

                        # Skip everything else in the current loop
                        # and proceed with the next batch cause this
                        # batch sucks
                        sys.exit('Exiting ...')
                    is_spike = spiker.update(total_loss_ellseg_value) if spiker else False

                    # Get metrics
                    batch_results_ellseg = get_metrics_ellseg(detach_cpu_numpy(out_dict),
                                                              detach_cpu_numpy(data_dict),
                                                              batch_results_ellseg,
                                                              args)
                    batch_results_ellseg['loss/ellseg_total'] = total_loss_ellseg_value
                    for k in loss_dict_ellseg:
                        batch_results_ellseg[f'loss/ellseg_{k}'] = loss_dict_ellseg[k].item()
                else:
                    total_loss_ellseg = 0.0
                    total_loss_ellseg_value = 0.0
                    disc_loss = 0.0
                    loss_dict_ellseg = {}
                    batch_results_ellseg = {}
                    is_spike = False

            #take metrics of the simply head
            if args['loss_w_supervise']:
                if args['net_simply_head']:
                    batch_results_ellseg = get_metrics_simple(out_dict, move_gpu(data_dict, out_dict['pupil_c_UV'].device),
                                                            batch_results_ellseg, 
                                                            image_resolution_diagonal,
                                                            args)
                elif args['net_rend_head'] and not train_with_mask:
                    batch_results_ellseg = get_metrics_simple(rend_dict, move_gpu(data_dict, rend_dict['pupil_c_UV'].device),
                                                            batch_results_ellseg, 
                                                            image_resolution_diagonal,
                                                            args)

            #define losses
            if args['net_rend_head'] and (args['net_ellseg_head'] == False \
                        or args['loss_w_ellseg'] == 0):
                loss = total_loss_rend
                if args['loss_w_supervise']:
                    loss += args['loss_w_supervise'] * total_supervised_loss
            elif  args['net_rend_head'] == False and args['net_ellseg_head']:
                loss = total_loss_ellseg
                if args['loss_w_supervise']:
                    loss += args['loss_w_supervise'] * total_supervised_loss
            elif args['net_rend_head'] and args['net_ellseg_head']:
                loss = total_loss_rend + args['loss_w_ellseg'] * total_loss_ellseg
                if args['loss_w_supervise']:
                    loss += args['loss_w_supervise'] * total_supervised_loss
            elif args['net_simply_head']:
                loss = args['loss_w_supervise'] * total_supervised_loss

            is_spike = spiker.update(loss.item()) if spiker else False

            if mode == 'train':
                if args['adv_DG']:
                    # Teach the disc to classify the domains based on predicted
                    # segmentation mask
                    disc_loss.backward(retain_graph=True)

                    # Remove gradients accumulated in the encoder and decoder
                    net.enc.zero_grad()
                    net.dec.zero_grad()
                    net.elReg.zero_grad()
                    net.renderingReg.zero_grad()

                loss.backward()

                #print('gradient {}'.format(compute_norm(net)))

                if not is_spike:

                    # Gradient clipping, if needed, goes here
                    if args['grad_clip_norm'] > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                                   max_norm=args['grad_clip_norm'],
                                                                   norm_type=2)

                    # Step the optimizer and update weights
                    # if args['adv_DG']:
                    #     optimizer_disc.step()

                    optimizer.step()

                else:
                    total_norm = np.inf
                    print('-------------')
                    print('Spike detected! Loss: {}'.format(loss.item()))
                    print('-------------')

            # Zero out gradients no matter what
            optimizer.zero_grad()
            net.zero_grad()

            if optimizer_disc:
                optimizer_disc.zero_grad()

        if args['do_distributed']:
            torch.cuda.synchronize()

        if bt < args['num_samples_for_embedding']:
            embeddings.append(out_dict['latent'])
            dataset_id.append(data_dict['ds_num'])
        
        # Merge metrics
        batch_results = merge_two_dicts(detach_cpu_numpy(batch_results_ellseg),
                                        detach_cpu_numpy(batch_results_rend))

        # Record network outputs
        if (mode == 'test') and args['save_test_maps']:

            test_dict = merge_two_dicts(detach_cpu_numpy(out_dict),
                                        detach_cpu_numpy(batch_results))
            test_dict_list = convert_to_list_entries(test_dict)

            for idx, entry in enumerate(test_dict_list):

                sample_id = data_dict['archName'][idx] + '/' + \
                                    str(data_dict['im_num'][idx].item())

                for key, value in entry.items():

                    sample_id_key = sample_id + '/' + key

                    try:
                        if 'predict' not in key:
                            if 'mask' in key:
                                # Save mask as integer objects to avoid
                                # overloading harddrive
                                test_results_obj.create_dataset(sample_id_key,
                                                                data=value,
                                                                dtype='uint8',
                                                                compression='lzf')
                            else:
                                # Save out remaining data points with float16 to
                                # avoid overloading harddrive
                                test_results_obj.create_dataset(sample_id_key,
                                                                data=np.array(value))
                    except Exception:
                        print('Repeated sample because of corrupt entry in H5')
                        print('Skipping sample {} ... '.format(sample_id))

        batch_results['loss'] = loss.item()
        batch_metrics = aggregate_metrics([batch_results])
        metrics.append(batch_results)

        if args['exp_name'] != 'DEBUG':
            log_wandb(batch_metrics, rend_dict, data_dict, out_dict, loss,
                        available_predicted_mask, mode, epoch, bt, H, W, args)

        # Use this if you want to save out spiky conditions
        # save all images for train validation and testing
        # if ((bt % 100 == 0) or is_spike) and rank_cond and (mode == 'train'):
        if (available_predicted_mask and args['net_rend_head'] and \
                     bt % args['produce_rend_mask_per_iter'] == 0):
            # Saving spiky conditions unnecessarily bloats the drive
            save_out(out_dict, rend_dict, data_dict['image'], path_dict, mode, 
                                        is_spike, args, epoch, bt)
        elif (bt % args['produce_rend_mask_per_iter'] ==0 and train_with_mask):
            save_out(out_dict, None, data_dict['image'], path_dict, mode, 
                                        is_spike, args, epoch, bt)        

        del out_dict  # Explicitly free up memory
        del rend_dict # Explicitly free up memory
        del batch_results_ellseg
        del batch_results_rend

    if csv_save_dir is not None:
        if os.path.isdir(csv_save_dir):
            csv_save_path = os.path.join(csv_save_dir, f'{mode}_raw_results.csv')
    else:
        csv_save_path = None
    results_dict = aggregate_metrics(metrics, csv_save_path)
    #results_dict = batch_metrics

    if (mode == 'test') and args['save_test_maps']:
        test_results_obj.close()

    # Clear out RAM accumulation if any
    del loader_iter

    # Clear out the cuda and RAM cache
    torch.cuda.empty_cache()
    gc.collect()

    return results_dict

def log_wandb(batch_metrics, rend_dict, data_dict, out_dict, loss,
              available_predicted_mask, mode, epoch, bt, H, W, args):
    #save result for training mode
    class_labels = {0: 'background',1: 'iris',2: 'pupil'}
    class_labels_gaze = {0: 'background',1: 'gaze'}

    #log all directory to wandb
    if (bt % 100 == 0):
        for key, item in batch_metrics.items():
            if ('mean' in key) and (not np.isnan(item).any()):
                if ('pupil_c_px_dst' in key) or ('eyeball_c_px_dist' in key) or \
                    ('gaze_ang_deg' in key) or ('loss' in key) or ('score' in key) or \
                    ('rendering_iou' in key) or ('masked_rendering_iou' in key) or \
                    ('norm' in key) or ('gaze' in key):
                    wandb.log({'{}/{}'.format(mode,key): item, 'epoch':epoch, 'batch':bt}, commit = False)

        if args['net_ellseg_head'] and args['loss_w_ellseg']:
            if bt % args['produce_rend_mask_per_iter'] == 0:
                mask_img = wandb.Image(data_dict['image'][0].detach().cpu().numpy(), 
                                        masks={
                                                "predictions": {
                                                    "mask_data": out_dict['mask'][0],
                                                    "class_labels": class_labels
                                                },
                                                "ground_truth": {
                                                    "mask_data": data_dict['mask'][0].detach().cpu().numpy(),
                                                    "class_labels": class_labels
                                                }
                                        })
                wandb.log({'{}/ellseg_mask'.format(mode): mask_img, 'epoch': epoch, 'batch': bt}, commit = False)
        
        if args['net_rend_head']:
            if (available_predicted_mask):
                mask_img = wandb.Image(data_dict['image'][0].detach().cpu().numpy(), 
                                        masks={
                                            "predictions": {
                                                "mask_data": np.clip(rend_dict['mask'][0], 0, 255),
                                                "class_labels": class_labels
                                            },
                                            "ground_truth": {
                                                "mask_data": np.clip(data_dict['mask'][0].detach().cpu().numpy(), 0, 255),
                                                "class_labels": class_labels
                                            }
                                        })
                wandb.log({'{}/rend_mask'.format(mode): mask_img, 'epoch': epoch, 'batch': bt}, commit = False)
                if 'TEyeD' in args['cur_obj']:
                    gaze_mask_gt = generate_gaze_gt(data_dict['eyeball'][0].detach().cpu().numpy(),
                                            data_dict['gaze_vector'][0].detach().cpu().numpy(), 
                                            H, W)
                    gaze_img = wandb.Image(data_dict['image'][0].detach().cpu().numpy(), 
                                            masks={
                                                "predictions": {
                                                    "mask_data": rend_dict['gaze_img'][0],
                                                    "class_labels": class_labels_gaze
                                                },
                                                "ground_truth": {
                                                    "mask_data": gaze_mask_gt,
                                                    "class_labels": class_labels_gaze
                                                }
                                            })
                else:
                    gaze_img = wandb.Image(data_dict['image'][0].detach().cpu().numpy(), 
                                            masks={
                                                "predictions": {
                                                    "mask_data": rend_dict['gaze_img'][0],
                                                    "class_labels": class_labels_gaze
                                                }
                                            })
                wandb.log({'{}/gaze_mask'.format(mode): gaze_img, 'epoch': epoch, 'batch': bt}, commit = False)

        wandb.log({'{}/loss'.format(mode): loss.item(), 'epoch': epoch, 'batch': bt})

def save_out(ellseg_dict, rend_dict, image, path_dict, mode,
             is_spike, args, epoch, bt):

    if args['net_ellseg_head'] and args['loss_w_ellseg']:
        ellseg_dict['image'] = image
        save_plot(ellseg_dict, path_dict, mode, 'ellseg', True,
                        is_spike, args, epoch, bt)

    if args['net_rend_head']:
        rend_dict['image'] = image
        save_plot(rend_dict, path_dict, mode, 'rend', True,
                                is_spike, args, epoch, bt)
        if 'TEyeD' in args['cur_obj']:
            save_plot(rend_dict, path_dict, mode, 'rend', False,
                                    is_spike, args, epoch, bt)  

def save_plot(data_dict, path_dict, mode, head, mask, is_spike, args, epoch, bt):

    if is_spike:
        im_name = '{}_spike/{}_ep_{}_bt_{}.jpg'.format(mode, head, epoch, bt)
    else:
        if mask:
            im_name = '{}_{}/{}_mask_bt_{}.jpg'.format(mode, epoch, head, bt)
        else:
            im_name = '{}_{}/{}_gaze_bt_{}.jpg'.format(mode, epoch, head, bt)

    path_im_image = os.path.join(path_dict['figures'], im_name)

    plot_images_with_annotations(detach_cpu_numpy(data_dict),# data_dict
                                         args,
                                         write=path_im_image,
                                         rendering= True,
                                         mask=mask,
                                         remove_saturated=False,
                                         is_list_of_entries=False,
                                         is_predict=True,
                                         show=False,
                                         mode=mode,
                                         epoch= epoch,
                                         batch=bt)

def reshape_gt(gt_dict, args):
    train_with_mask = args['loss_w_rend_pred_2_gt_edge'] or args['loss_w_rend_gt_2_pred'] \
                        or args['loss_w_rend_pred_2_gt']
    ##################### Prepare Tensors ############################
    gt_dict['image'] = rearrange(gt_dict['image'], 'b f h w-> (b f) h w')
    if train_with_mask:
        gt_dict['mask'] = rearrange(gt_dict['mask'], 'b f h w-> (b f) h w')
        gt_dict['mask_available'] = rearrange(gt_dict['mask_available'], 
                                                        'b f -> (b f)')
    if args['net_ellseg_head']:
        gt_dict['spatial_weights'] = rearrange(gt_dict['spatial_weights'], 
                                                        'b f h w-> (b f) h w')
        gt_dict['distance_map'] = rearrange(gt_dict['distance_map'], 
                                                        'b f c h w-> (b f) c h w')
    
    gt_dict['pupil_center'] = rearrange(gt_dict['pupil_center'], 
                                                'b f e -> (b f) e')
    gt_dict['pupil_ellipse'] = rearrange(gt_dict['pupil_ellipse'], 
                                                 'b f e -> (b f) e')
    gt_dict['pupil_center_norm'] = rearrange(gt_dict['pupil_center_norm'], 
                                                        'b f e -> (b f) e')
    gt_dict['pupil_center_available'] = rearrange(gt_dict['pupil_center_available'], 
                                                                'b f -> (b f)')
    gt_dict['pupil_ellipse_norm'] = rearrange(gt_dict['pupil_ellipse_norm'], 
                                                        'b f e -> (b f) e')
    gt_dict['pupil_ellipse_available'] = rearrange(gt_dict['pupil_ellipse_available'], 
                                                                'b f -> (b f)')

    gt_dict['iris_ellipse'] = rearrange(gt_dict['iris_ellipse'], 
                                                'b f e -> (b f) e')
    gt_dict['iris_ellipse_norm'] = rearrange(gt_dict['iris_ellipse_norm'], 
                                                            'b f e -> (b f) e')
    gt_dict['iris_ellipse_available'] = rearrange(gt_dict['iris_ellipse_available'], 
                                                                'b f -> (b f)')

    gt_dict['ds_num'] = rearrange(gt_dict['ds_num'], 'b f -> (b f)')
    gt_dict['im_num'] = rearrange(gt_dict['im_num'], 'b f -> (b f)')
    gt_dict['is_bad'] = rearrange(gt_dict['is_bad'], 'b f -> (b f)')

    gt_dict['eyeball'] = rearrange(gt_dict['eyeball'], 'b f e -> (b f) e')
    gt_dict['gaze_vector'] = rearrange(gt_dict['gaze_vector'], 'b f e -> (b f) e')
    gt_dict['pupil_lm_2D'] = rearrange(gt_dict['pupil_lm_2D'], 'b f e -> (b f) e')
    gt_dict['pupil_lm_23'] = rearrange(gt_dict['pupil_lm_3D'], 'b f e -> (b f) e')
    gt_dict['iris_lm_2D'] = rearrange(gt_dict['iris_lm_2D'], 'b f e -> (b f) e')
    gt_dict['iris_lm_3D'] = rearrange(gt_dict['iris_lm_3D'], 'b f e -> (b f) e')
    
    return gt_dict

def reshape_ellseg_out(out_dict, args):
    ##################### Prepare Tensors ############################
    if args['net_ellseg_head']:
        out_dict['predict'] = rearrange(out_dict['predict'], 'b f c h w-> (b f) c h w')

        out_dict['iris_ellipse'] = rearrange(out_dict['iris_ellipse'], 
                                                    'b f e -> (b f) e')

        out_dict['pupil_ellipse'] = rearrange(out_dict['pupil_ellipse'], 
                                                    'b f e -> (b f) e')
        out_dict['mask'] = rearrange(out_dict['mask'], 'b f h w-> (b f) h w')

        out_dict['pupil_conf'] = rearrange(out_dict['pupil_conf'], 'b f e -> (b f) e')
        out_dict['pupil_center'] = rearrange(out_dict['pupil_center'], 'b f e -> (b f) e')
        out_dict['pupil_ellipse_norm'] = rearrange(out_dict['pupil_ellipse_norm'], 
                                                            'b f e -> (b f) e')
        out_dict['pupil_ellipse_norm_regressed'] = rearrange(out_dict['pupil_ellipse_norm_regressed'], 
                                                            'b f e -> (b f) e')
        
        out_dict['iris_center'] = rearrange(out_dict['iris_center'], 'b f e -> (b f) e')
        out_dict['iris_conf'] = rearrange(out_dict['iris_conf'], 'b f e -> (b f) e')
        out_dict['iris_ellipse_norm'] = rearrange(out_dict['iris_ellipse_norm'], 
                                                                'b f e -> (b f) e')
        out_dict['iris_ellipse_norm_regressed'] = rearrange(out_dict['iris_ellipse_norm_regressed'], 
                                                                'b f e -> (b f) e')
        
    return out_dict

# %% Loss function
def get_loss_ellseg(gt_dict, pd_dict, alpha, beta,
             make_aleatoric=False, regress_loss=True, bias_removal=False,
             pseudo_labels=False, adv_loss=False, label_tracker=False):

    # Segmentation loss
    loss_seg = get_seg_loss(gt_dict, pd_dict, 0.5)

    # L1 w/o uncertainity loss for segmentation center
    loss_pupil_c = get_uncertain_l1_loss(gt_dict['pupil_center_norm'],
                                         pd_dict['pupil_ellipse_norm'][:, :2],
                                         None,
                                         uncertain=False,
                                         cond=gt_dict['pupil_center_available'],
                                         do_aleatoric=False)


    loss_iris_c  = get_uncertain_l1_loss(gt_dict['iris_ellipse_norm'][:, :2],
                                         pd_dict['iris_ellipse_norm'][:, :2],
                                         None,
                                         uncertain=False,
                                         cond=gt_dict['iris_ellipse_available'],
                                         do_aleatoric=False)


    # L1 with uncertainity loss for pupil center regression
    loss_pupil_c_reg = get_uncertain_l1_loss(gt_dict['pupil_center_norm'],
                                             pd_dict['pupil_ellipse_norm_regressed'][:, :2],
                                             None,
                                             uncertain=pd_dict['pupil_conf'][:, :2],
                                             cond=gt_dict['pupil_center_available'],
                                             do_aleatoric=make_aleatoric)

    # L1 with uncertainity loss for ellipse parameter regression from latent
    loss_pupil_el = get_uncertain_l1_loss(gt_dict['pupil_ellipse_norm'][:, 2:],
                                          pd_dict['pupil_ellipse_norm_regressed'][:, 2:],
                                          [4, 4, 3],
                                          uncertain=pd_dict['pupil_conf'][:, 2:],
                                          cond=gt_dict['pupil_ellipse_available'],
                                          do_aleatoric=make_aleatoric)


    loss_iris_el = get_uncertain_l1_loss(gt_dict['iris_ellipse_norm'],
                                         pd_dict['iris_ellipse_norm_regressed'],
                                         [1, 1, 4, 4, 3],
                                         uncertain=pd_dict['iris_conf'],
                                         cond=gt_dict['iris_ellipse_available'],
                                         do_aleatoric=make_aleatoric)

    # Gradient reversal
    if bias_removal:
        num_samples = gt_dict['ds_num'].shape[0]
        gt_ds_num = gt_dict['ds_num'].reshape(num_samples, 1, 1)
        gt_ds_num = gt_ds_num.repeat((1, ) + pd_dict['ds_onehot'].shape[-2:])
        loss_da = torch.nn.functional.cross_entropy(pd_dict['ds_onehot'],
                                                    gt_ds_num)
    else:
        loss_da = torch.tensor([0.0]).to(loss_seg.device)

    if not regress_loss:
        loss = 0.5*(20*loss_seg + loss_da) + (1-0.5)*(loss_pupil_c + loss_iris_c)
    else:
        loss = 0.5*(20*loss_seg + 2*alpha*loss_da) + \
               (1-0.5)*(loss_pupil_c + loss_iris_c +
                         loss_pupil_el + loss_iris_el + loss_pupil_c_reg)

    if adv_loss:
        da_loss_dec = torch.nn.functional.cross_entropy(pd_dict['disc_onehot'],
                                                        gt_dict['ds_num'].to(torch.long))

        # Inverse domain classification, we want to increase domain confusion
        # as training progresses. Ensure that the weight does not exceed
        # the weight of the main loss at any given epoch or else it could
        # lead to unexpected solutions in order to confused the discriminator
        loss = loss - 0.4*beta*da_loss_dec
    else:
        da_loss_dec = torch.tensor([0.0]).to(loss_seg.device)

    if pseudo_labels:

        # Generate pseudo labels and confidence based on entropy
        pseudo_labels, conf = generate_pseudo_labels(pd_dict['predict'])

        # Based on samples with groundtruth information, classify each
        # prediction as "good" or "bad" from entropy-based-confidence
        loc = remove_underconfident_psuedo_labels(conf.detach(),
                                                  label_tracker,
                                                  gt_dict=False) # gt_dict

        # Remove pseudo labels for samples with groundtruth
        loc = gt_dict['mask'] != -1  # Samples with groundtruth
        pseudo_labels[loc] = -1  # Disable pseudosamples
        conf[loc] = 0.0

        # Number of pixels and samples which are non-zero confidence
        num_valid_pxs = torch.sum(conf.flatten(start_dim=-2) > 0, dim=-1)+1
        num_valid_samples = torch.sum(num_valid_pxs > 0)

        pseudo_loss = torch.nn.functional.cross_entropy(pd_dict['predict'],
                                                        pseudo_labels,
                                                        ignore_index=-1,
                                                        reduction='none')
        pseudo_loss = (conf*pseudo_loss).flatten(start_dim=-2)
        pseudo_loss = torch.sum(pseudo_loss, dim=-1)/num_valid_pxs  # Average across pixels
        pseudo_loss = torch.sum(pseudo_loss, dim=0)/num_valid_samples  # Average across samples

        loss = loss + beta*pseudo_loss
    else:
        pseudo_loss = torch.tensor([0.0])

    loss_dict = {'da_loss': loss_da.item(),
                 'seg_loss': loss_seg.item(),
                 'pseudo_loss': pseudo_loss.item(),
                 'da_loss_dec': da_loss_dec.item(),
                 'iris_c_loss': loss_iris_c.item(),
                 'pupil_c_loss': loss_pupil_c.item(),
                 'iris_el_loss': loss_iris_el.item(),
                 'pupil_c_reg_loss': loss_pupil_c_reg.item(),
                 'pupil_params_loss': loss_pupil_el.item(),
                 }

    return loss, da_loss_dec, loss_dict

def get_metrics_simple(out_dict, gt_dict, metric_dict, diagonal, args):

    gt_pupil_c_UV = gt_dict['eyeball'][...,1:3] + \
                (gt_dict['eyeball'][...,0:1]  * gt_dict['gaze_vector'][...,:2])
    gt_pupil_c_UV = gt_pupil_c_UV.float()
    gt_eyeball_c_UV = gt_dict['eyeball'][...,1:3]

    pupil_c_UV = out_dict['pupil_c_UV']
    eyeball_c_UV = out_dict['eyeball_c_UV']

    #compute pupil center euclidean distance
    pupil_c_px_dist = torch.sqrt((pupil_c_UV[...,0] - gt_pupil_c_UV[...,0])**2
                                + (pupil_c_UV[...,1] - gt_pupil_c_UV[...,1])**2)
    
    eyeball_c_px_dist = torch.sqrt((eyeball_c_UV[...,0] - gt_eyeball_c_UV[...,0])**2
                                + (eyeball_c_UV[...,1] - gt_eyeball_c_UV[...,1])**2)
    
    metric_dict['pupil_c_px_dst'] = pupil_c_px_dist
    metric_dict['eyeball_c_px_dist'] = eyeball_c_px_dist

    metric_dict['norm_pupil_c_px_dst'] = (pupil_c_px_dist * 100) / diagonal
    metric_dict['norm_eyeball_c_px_dist'] = (eyeball_c_px_dist * 100) / diagonal 

    #compute the 3D gaze vector
    gt_gaze_3d = gt_dict['gaze_vector'].detach().cpu().numpy()
    pred_gaze_3d = out_dict['gaze_vector_3D'].detach().cpu().numpy()
    metric_dict['gaze_3D_ang_deg'] = angular_error(pred_gaze_3d, gt_gaze_3d)
    metric_dict['gaze_3D_xy_ang_deg'] = angular_error(pred_gaze_3d[...,:2]/np.linalg.norm(pred_gaze_3d[...,:2], axis=-1, keepdims=True),
                                                        gt_gaze_3d[...,:2]/np.linalg.norm(gt_gaze_3d[...,:2], axis=-1, keepdims=True))

    if args['loss_w_supervise_gaze_vector_3D_L2'] or args['loss_w_supervise_gaze_vector_3D_cos_sim']:
        metric_dict['score'] = metric_dict['gaze_3D_ang_deg']
    else:
        metric_dict['score'] = metric_dict['gaze_3D_ang_deg']

    #metric_dict = detach_cpu_numpy(metric_dict)
    
    return metric_dict


def angular_error(pred, gt):
    err = (pred * gt).sum(-1) #/ np.linalg.norm(pred, axis=-1)
    err = np.arccos(err)
    err = np.degrees(err)
    return err


# %% Get performance metrics for rendering
def get_metrics_rend(rendering_dict, gt_dict, metric_dict, diagonal, args, available_predicted_mask):

    B = args['batch_size']
    F = args['frames']

    train_with_mask = args['loss_w_rend_pred_2_gt_edge'] or args['loss_w_rend_gt_2_pred'] \
                        or args['loss_w_rend_pred_2_gt']

    if available_predicted_mask and train_with_mask:
        #keep the mask that is contained in the iris
        masked_predicted = np.where(gt_dict['mask']>0, 1, 0) * \
                                    rendering_dict['mask']

        # Segmentation IoU Rendering
        metric_dict['rendering_iou'] = get_seg_metrics(gt_dict['mask'],
                                                    rendering_dict['mask'],
                                                    gt_dict['mask_available'], 
                                                    B, F)

        if args['net_rend_head'] and (args['net_ellseg_head'] == False \
                    or args['loss_w_ellseg'] == 0):
            metric_dict['score'] = metric_dict['rendering_iou'].mean(axis=1)
        
        # Segmentation Masked IoU Rendering
        metric_dict['masked_rendering_iou'] = get_seg_metrics(gt_dict['mask'],
                                                    masked_predicted,
                                                    gt_dict['mask_available'], 
                                                    B, F)


    elif not train_with_mask:
        pass
    else:
        #save as nan to calculate the correct mean
        if args['net_rend_head'] and (args['net_ellseg_head'] == False \
                        or args['loss_w_ellseg'] == 0):
            metric_dict['score'] = np.zeros((B*F))
            metric_dict['score'][metric_dict['score']==0] = np.nan
        metric_dict['rendering_iou'] = np.zeros((B*F,3))
        metric_dict['rendering_iou'][metric_dict['rendering_iou']==0] = np.nan
        metric_dict['masked_rendering_iou'] = np.zeros((B*F,3))
        metric_dict['masked_rendering_iou'][metric_dict['masked_rendering_iou']==0] = np.nan


    if 'TEyeD' in args['cur_obj']:
        # gaze is available alwayss
        #compute the pupil center in UV coordinates using the gt eyeball and gaze vector
        gt_pupil_c_UV = gt_dict['eyeball'][...,1:3] + \
                (gt_dict['eyeball'][...,0:1]  * gt_dict['gaze_vector'][...,:2])
        gt_pupil_c_UV = gt_pupil_c_UV

        #compute pupil center euclidean distance
        pupil_c_px_dist = np.sqrt((rendering_dict['pupil_c_UV'][...,0] - gt_pupil_c_UV[...,0])**2
                                  + (rendering_dict['pupil_c_UV'][...,1] - gt_pupil_c_UV[...,1])**2)
        
        metric_dict['pupil_c_px_dst'] = pupil_c_px_dist
        metric_dict['norm_pupil_c_px_dst'] = (pupil_c_px_dist * 100) / diagonal
        #compute eyeball center euclidean distance
        gt_eyeball_c_UV = gt_dict['eyeball'][...,1:3]

        eyeball_c_px_dist = np.sqrt((rendering_dict['eyeball_c_UV'][...,0] - gt_eyeball_c_UV[...,0])**2
                                  + (rendering_dict['eyeball_c_UV'][...,1] - gt_eyeball_c_UV[...,1])**2)

        metric_dict['eyeball_c_px_dist'] = eyeball_c_px_dist
        metric_dict['norm_eyeball_c_px_dist'] = (eyeball_c_px_dist * 100) / diagonal

        gt_gaze_3d = gt_dict['gaze_vector']
        pred_gaze_3d = rendering_dict['gaze_vector_3D']
        metric_dict['gaze_3D_ang_deg'] = angular_error(pred_gaze_3d, gt_gaze_3d)
        metric_dict['gaze_3D_xy_ang_deg'] = angular_error(pred_gaze_3d[...,:2]/np.linalg.norm(pred_gaze_3d[...,:2], axis=-1, keepdims=True),
                                                          gt_gaze_3d[...,:2]/np.linalg.norm(gt_gaze_3d[...,:2], axis=-1, keepdims=True))

    elif 'nvgaze' in args['cur_obj']:
        #compute the gaze error
        metric_dict['gaze_ang_deg'] = np.zeros((B*F))
        metric_dict['eyeball_c_px_dist'] = np.zeros((B*F))
        metric_dict['pupil_c_px_dst'] = np.zeros((B*F))
    else:
        metric_dict['gaze_ang_deg'] = np.zeros((B*F))
        metric_dict['eyeball_c_px_dist'] = np.zeros((B*F))
        metric_dict['pupil_c_px_dst'] = np.zeros((B*F))

    
    return metric_dict
    
# %% Get performance metrics
def get_metrics_ellseg(pd_dict, gt_dict, metric_dict, args):
    B = args['batch_size']
    F = args['frames']
    # Results metrics of important per sample

    height, width = gt_dict['mask'].shape[-2:]
    scale = min([height, width])

    # Segmentation IoU
    metric_dict['iou'] = get_seg_metrics(gt_dict['mask'],
                                         pd_dict['mask'],
                                         gt_dict['mask_available'], B, F)

    metric_dict['iou_recon'] = get_seg_metrics(gt_dict['mask'],
                                               pd_dict['mask_recon'],
                                               gt_dict['mask_available'], B, F)

    # Pupil and Iris center metric
    metric_dict['pupil_c_dst'] = get_distance(gt_dict['pupil_center'],
                                              pd_dict['pupil_ellipse'][:, :2],
                                              gt_dict['pupil_center_available'])

    metric_dict['iris_c_dst'] = get_distance(gt_dict['iris_ellipse'][:, :2],
                                             pd_dict['iris_ellipse'][:, :2],
                                             gt_dict['iris_ellipse_available'])

    # Pupil and Iris axis metric
    metric_dict['pupil_axes_dst'] = get_distance(gt_dict['pupil_ellipse'][:, 2:-1],
                                                 pd_dict['pupil_ellipse'][:, 2:-1],
                                                 gt_dict['pupil_ellipse_available'])


    metric_dict['iris_axes_dst'] = get_distance(gt_dict['iris_ellipse'][:, 2:-1],
                                                pd_dict['iris_ellipse'][:, 2:-1],
                                                gt_dict['iris_ellipse_available'])

    # Pupil and Iris angle metric
    metric_dict['pupil_ang_dst'] = getAng_metric(gt_dict['pupil_ellipse'][:, -1],
                                                 pd_dict['pupil_ellipse'][:, -1],
                                                 gt_dict['pupil_ellipse_available'])

    metric_dict['iris_ang_dst'] = getAng_metric(gt_dict['iris_ellipse'][:, -1],
                                                pd_dict['iris_ellipse'][:, -1],
                                                gt_dict['iris_ellipse_available'])

    # Evaluation metric
    # max value will be 1, min value  will be 0. All individual metrics are
    # scaled approximately equally
    term_A = metric_dict['iou'][...,-2:].mean(axis=1) \
        if np.any(gt_dict['mask_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_A[~gt_dict['mask_available']] = np.nan

    term_B = metric_dict['iou_recon'][...,-2:].mean(axis=1) \
        if np.any(gt_dict['mask_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_B[~gt_dict['mask_available']] = np.nan


    term_C = 1 - (1/scale)*metric_dict['pupil_c_dst'] \
        if np.any(gt_dict['pupil_center_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_C[~gt_dict['pupil_center_available']] = np.nan

    term_D = 1 - (1/scale)*metric_dict['iris_c_dst'] \
        if np.any(gt_dict['iris_ellipse_available']) \
        else np.nan*np.zeros((gt_dict['mask_available'].shape[0], ))
    term_D[~gt_dict['iris_ellipse_available']] = np.nan

    term_mat = np.stack([term_A, term_B, term_C, term_D], axis=1)

    metric_dict['score'] = np.nanmean(term_mat, axis=1)

    metric_dict['iou_score']  = term_A

    return metric_dict

def aggregate_metrics(list_metric_dicts, csv_save_path=None):
    # Aggregate and compute global stats
    keys_list = list_metric_dicts[0].keys()
    agg_dict = {}
    raw_dict = {}
    for key_entry in keys_list:
        try:
            if 'loss' in key_entry:
                raw_dict[key_entry] = np.array([ele[key_entry] for ele in list_metric_dicts])
            else:
                raw_dict[key_entry] = np.concatenate([ele[key_entry] for ele in list_metric_dicts], axis=0)
            if  'iou' in key_entry:
                # If more than 1 dimension, then it corresponds to iou tag
                agg_dict[key_entry] = np.nanmean(raw_dict[key_entry], axis=0)

            agg_dict[key_entry+'_mean'] = np.nanmean(raw_dict[key_entry], axis=0)
        except:
            pass
    if csv_save_path:
        import pandas as pd
        values = []
        names = []
        for k in raw_dict:
            if 'loss' in k:
                continue
            if 'iou' in k:
                for j in range(raw_dict[k].shape[1]):
                    names.append(f'{k}_class{j}')
                    values.append(raw_dict[k][:, j])
            else:
                names.append(k)
                values.append(raw_dict[k])
        values = np.asarray(values).T
        df = pd.DataFrame(data=values, columns=names)
        df.to_csv(csv_save_path)

    return agg_dict
