#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:39:49 2021

@author: rakshit
"""
import os
import torch
import random
import warnings
import numpy as np
import wandb
import string

from datetime import datetime   
from distutils.dir_util import copy_tree

from main import train
from args_maker import make_args

# Suppress warnings
warnings.filterwarnings('ignore')

def create_experiment_folder_tree(repo_root,
                                  path_exp_records,
                                  exp_name,
                                  is_test=False,
                                  create_tree=True):

    if is_test:
        exp_name_str = exp_name
    else:
        now = datetime.now()
        date_time_str = now.strftime('%d_%m_%y_%H_%M_%S')
        rnd_str = ''.join(random.choice(string.ascii_letters) for i in range(5))
        exp_name_str = exp_name + '_' + rnd_str + '_' + date_time_str

    path_exp = os.path.join(path_exp_records, exp_name_str)

    path_dict = {}
    for ele in ['results', 'figures', 'logs', 'src']:
        path_dict[ele] = os.path.join(path_exp, ele)
        os.makedirs(path_dict[ele], exist_ok=True)

    path_dict['exp'] = path_exp

    # TODO Do not include hidden files and then turn back on
    # if (not is_test) and create_tree:
    #     # Do not copy data because in test only condition, this folder would
    #     # already be populated
    #     copy_tree(repo_root, os.path.join(path_exp, 'src'))

    return path_dict, exp_name_str


def cleanup():
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = vars(make_args())
    #args['exp_name'] = 'DEBU'

    path_dict, exp_name_str = create_experiment_folder_tree(args['repo_root'],
                                              args['path_exp_tree'],
                                              args['exp_name'],
                                              args['only_test'],
                                              create_tree=args['local_rank']==0 if args['do_distributed'] else True)
    if 'DEBUG' not in args['exp_name']:
        wandb.init(project="my-test-project", entity='orasi', config=args, name=exp_name_str)
    else:
        #wandb.init(project="my-test-project", config=args, name=exp_name_str)
        args['batches_per_ep'] = 10
        args['epochs'] = 2
        #args['path_model'] = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/T_N0_H2_F4_C_fKysY_27_04_23_23_58_41/results/2.pt'
        args['frames'] = 4 
        args['batch_size']= 4
        #args['use_GPU']=1

        #args['model'] = 'DenseEl3'
        args['model'] = 'res_50_3'
        args['early_stop_metric'] = '3D'

        args['random_dataloader'] = False
        args['temp_n_angles'] = 72
        args['temp_n_radius'] = 8

        args['net_rend_head'] = True
        args['loss_w_rend_pred_2_gt_edge'] = 0.0
        args['loss_w_rend_gt_2_pred'] = 0.0
        args['loss_w_rend_pred_2_gt'] = 0.0
        args['loss_w_rend_diameter'] = 0.0

        args['net_ellseg_head'] = False
        args['loss_w_ellseg'] = 0.0

        args['loss_rend_vectorized'] = True
        args['detect_anomaly'] = 0

        #supervised_loss
        args['net_simply_head'] = False
        args['loss_w_supervise'] = 1
        args['loss_w_supervise_eyeball_center'] = 0.0
        args['loss_w_supervise_pupil_center'] = 0.0
        args['loss_w_supervise_gaze_vector_UV'] = 0.0
        args['loss_w_supervise_gaze_vector_3D_L2'] = 5.0
        args['loss_w_supervise_gaze_vector_3D_cos_sim'] = 0.0

        args['scale_bound_eye'] = 'version_1'

        args['pretrained_resnet'] = False
        args['net_simply_head_tanh'] = 0
        
        args['grad_clip_norm'] = 0.1
        args['optimizer_type'] = 'adamw_cos'

        #args['only_test'] = 1

        # args['only_test']=0
        # args['only_valid']=0
        args['train_data_percentage'] = 1.0
        #args['use_pkl_for_dataload'] = True
        #args['path_data']='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets/All'
        #args['path_exp_tree']='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/'
        #args['weights_path'] = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/dimitrios_gaze/TG_trfT_2e-4_perc_0.005_res_50_3_BF_4_4_Nang72_Nrad_8_vpsEX_21_06_23_10_39_38/results/last.pt'
        #args['path_model']='/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/dimitrios_gaze/TG_trfT_2e-4_perc_0.005_res_50_3_BF_4_4_Nang72_Nrad_8_vpsEX_21_06_23_10_39_38/results/last.pt'


    path_dict['repo_root'] = args['repo_root']
    path_dict['path_data'] = args['path_data']

    #change the number of frames to predifine 10
    #to load the pkl file with 10 images
    if args['use_pkl_for_dataload']:
        args['frames']=4

    # %% DDP essentials

    if args['do_distributed']:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        world_size = torch.distributed.get_world_size()

    else:
        world_size = 1

    global batch_size_global
    batch_size_global = int(args['batch_size']*world_size)
 
    #torch.cuda.set_device(args['local_rank'])
    args['world_size'] = world_size
    args['batch_size'] = int(args['batch_size']/world_size)

    # %%
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Set seeds
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    # Train and save validated model
    if not args['only_test']:
        if not args['only_valid']:
            print('train mode')
            train(args, path_dict, validation_mode=False, test_mode=False)

            print('validation mode')
            train(args, path_dict, validation_mode=True, test_mode=False)

            # Test out best model and save results
            print("test mode")
            train(args, path_dict, validation_mode=False, test_mode=True)


    # Close process group
    if args['do_distributed']:
        torch.distributed.barrier()
        cleanup()
    elif args['only_valid']:
        print('validation mode')
        train(args, path_dict, validation_mode=True, test_mode=False)
    elif args['only_test']:

        print('validation mode')
        train(args, path_dict, validation_mode=True, test_mode=False)

        print("test mode")
        # Test out best model and save results
        train(args, path_dict, validation_mode=False, test_mode=True)

    wandb.finish()