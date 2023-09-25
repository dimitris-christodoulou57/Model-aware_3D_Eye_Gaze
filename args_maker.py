#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:30:38 2021
@author: rakshit
"""
from pprint import pprint
import argparse
import getpass
import socket
import os


username = getpass.getuser()
host = socket.gethostname()
if username == 'nipopovic' and host == 'archer':
    os.environ["WANDB_DIR"] = "/home/nipopovic/MountedDirs/aegis_cvl/aegis_cvl_root/data/nikola/Results/3d_gaze_seg"
    masterkey_root = '/home/nipopovic/MountedDirs/aegis_cvl/aegis_cvl_root/data/dchristodoul/Datasets/MasterKey'
    dataset_root = '/home/nipopovic/MountedDirs/aegis_cvl/aegis_cvl_root/data/dchristodoul/Datasets/All'
    results_root = '/home/nipopovic/MountedDirs/aegis_cvl/aegis_cvl_root/data/nikola/Results/3d_gaze_seg'
    default_repo = '/home/nipopovic/Code/3D-eye-model-seg'
    print('User nipopovic machine archer')
elif username == 'nipopovic' and os.path.expanduser("~") == '/cluster/home/nipopovic':
    # Euler
    os.environ["WANDB_DIR"] = "/cluster/work/cvl/specta/experiment_logs/dimitrios_gaze/"
    masterkey_root = '/cluster/work/cvl/specta/data/dimitrios_gaze/MasterKey'
    dataset_root = '/cluster/work/cvl/specta/data/dimitrios_gaze/All'
    results_root = '/cluster/work/cvl/specta/experiment_logs/dimitrios_gaze'
    default_repo = '/cluster/project/cvl/specta/code/3D-eye-model-seg'
    print('User nipopovic machine euler')
elif username == 'dchristodoul':
    os.environ["WANDB_DIR"] = "/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/wandb"
    masterkey_root = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets/MasterKey'
    dataset_root = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets/All'
    results_root = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results'
    default_repo = '/home/dchristodoul/3D-eye-model-seg'
    print('User dchristodoul')
else:
    raise NotImplementedError

def make_args():


    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='DEBUG',
                        help='experiment string or identifier')
    
    parser.add_argument('--use_pkl_for_dataload', type=bool, default=False,
                        help='pkl to load the data')
    parser.add_argument('--produce_rend_mask_per_iter', type=int ,default=2000,
                        help='set the num of iteration to generate the mask')
    parser.add_argument('--perform_valid', type=int ,default=25,
                        help='perform validation')

    # %% Hyperparams
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='base learning rate')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight_decay')
    parser.add_argument('--seed', type=int, default=108,
                        help='seed value for all packages')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size for training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank to set GPU')
    parser.add_argument('--lr_decay', type=int, default=0,
                        help='learning rate decay')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout? anything above 0 activates dropout')

    # %% Model specific parameters
    parser.add_argument('--base_channel_size', type=int, default=32,
                        help='base channel size around which model grows')
    parser.add_argument('--growth_rate', type=float, default=1.2,
                        help='growth rate of channels in network')
    parser.add_argument('--track_running_stats', type=int, default=0,
                        help='disable running stats for better transfer')
    parser.add_argument('--extra_depth', type=int, default=0,
                        help='extra convolutions to the encoder')
    parser.add_argument('--grad_rev', type=int, default=0,
                        help='gradient reversal for dataset identity')
    parser.add_argument('--adv_DG', type=int, default=0,
                        help='enable discriminator')
    parser.add_argument('--equi_var', type=int, default=0,
                        help='normalize data to respect image dimensions')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='number of encoder decoder blocks')
    parser.add_argument('--use_frn_tlu', type=int, default=0,
                        help='replace BN+L.RELU with FRN+TLU')
    parser.add_argument('--use_instance_norm', type=int, default=0,
                        help='replace BN with IN')
    parser.add_argument('--use_group_norm', type=int, default=0,
                        help='replace BN with GN, 8 channels per group')
    parser.add_argument('--use_ada_instance_norm', type=int, default=0,
                        help='use adaptive instance normalization')
    parser.add_argument('--use_ada_instance_norm_mixup', type=int, default=0,
                        help='use adaptive instance normalization with mixup')
    

    # %% Discriminator stats
    parser.add_argument('--disc_base_channel_size', type=int, default=8,
                        help='discriminator base channels?')
    '''
    Notes on discriminator channel size.
    Rakshit - from my experiments, I find that the discriminator easily learns
    the domains apart and adding more channels simply eats up the memory
    without contributing any more discriminative power. Hence, I leave this at
    a small value.
    '''

    # %% Experiment parameters
    parser.add_argument('--path_exp_tree', type=str,
                        default=results_root,
                        help='path to all experiments result folder')
    parser.add_argument('--path_data', type=str,
                        default=dataset_root,
                        help='path to all H5 file data')
    parser.add_argument('--path2MasterKey', type=str,
                        default=masterkey_root,
                        help='path to all H5 file data')
    parser.add_argument('--path_model', type=str, default=[],
                        help='path to model for test purposes')
    parser.add_argument('--repo_root', type=str,
                        default=default_repo,
                        help='path to repo root')
    parser.add_argument('--reduce_valid_samples', type=int, default=10,
                        help='reduce the number of\
                            validaton samples to speed up')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save weights every 1 iterations')

    # %% Train or test parameters
    parser.add_argument('--mode', type=str, default='one_vs_one',
                        help='training mode:\
                            one_vs_one, all_vs_one, all-one_vs_one')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--cur_obj', type=str, default='TEyeD',
                        help='which dataset to train on or remove?\
                            in all_vs_one, this flag does nothing')
    parser.add_argument('--aug_flag', type=int, default=1,
                        help='enable augmentations?')
    parser.add_argument('--one_by_one_ds', type=int, default=0,
                        help='train on a single dataset, one after the other')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='early stop epoch count')
    parser.add_argument('--mixed_precision', type=int, default=0,
                        help='enable mixed precision training and testing')
    parser.add_argument('--batches_per_ep', type=int, default=10,
                        help='number of batches per training epoch')
    parser.add_argument('--use_GPU', type=int, default=0,
                        help='train on GPU?')
    parser.add_argument('--remove_spikes', type=int, default=1,
                        help='remove noisy batches for smooth training')
    parser.add_argument('--pseudo_labels', type=int, default=0,
                        help='generate pseudo labels on datasets with missing\
                            labels')
    parser.add_argument('--frames', type=int, default=4,
                        help='number of frames that is used')

    # %% Model specific parameters
    parser.add_argument('--use_scSE', type=int, default=0,
                        help='use concurrent spatial and channel excitation \
                        at the end of every encoder or decoder block')
    parser.add_argument('--make_aleatoric', type=int, default=0,
                        help='add aleatoric formulation\
                            during latent regression')
    parser.add_argument('--scale_factor', type=float, default=0.0,
                        help='modify scaling factor')
    parser.add_argument('--make_uncertain', type=int, default=0,
                        help='activate aleatoric and epistemic')
    parser.add_argument('--continue_training', type=str, default='',
                        help='continue training from these weights')
    parser.add_argument('--regression_from_latent', type=int, default=1,
                        help='disable regression from the latent space?')
    parser.add_argument('--curr_learn_losses', type=int, default=1,
                        help='add two rampers and use them as you see fit')
    parser.add_argument('--regress_channel_grow', type=float, default=0,
                        help='grow channels in the regression module. Default\
                            0 means the channel size stays the same.')
    parser.add_argument('--maxpool_in_regress_mod', type=int, default=-1,
                        help='replace avg pool with max pooling in regression\
                            module, if -1, then pooling is disabled')
    parser.add_argument('--dilation_in_regress_mod', type=int, default=1,
                        help='enable dilation in the regression module')
    parser.add_argument('--groups', type=int, default=1,
                        help='group size? Default: all, i.e groups=1')

    # %% Model selection
    parser.add_argument('--net_simply_head', action='store_true', default=False,
                        help='direct predict the eyeball and pupil coordinates in 2D')
    parser.add_argument('--net_simply_head_tanh', type=str, default=1,
                        help='activate tanh for simple supervision')
    parser.add_argument('--net_ellseg_head', action='store_true', default=False,
                        help='compute segmentation head')
    parser.add_argument('--net_rend_head', action='store_true', default=False,
                        help='compute rendering head (3d eye model)')
    parser.add_argument('--model', type=str, default='DenseEl0',
                        help='DenseElNet, RITNet')
    
    parser.add_argument('--train_data_percentage', type=float, default=1.0,
                        help='percentage of training data')
    parser.add_argument('--loss_w_supervise', type=float, default=0.0,
                        help='loss component for supervising with ground truth')
    parser.add_argument('--loss_w_supervise_eyeball_center', type=float, default=0.0,
                        help='loss component for supervising eyeball center with ground truth')
    parser.add_argument('--loss_w_supervise_pupil_center', type=float, default=0.0,
                        help='loss component for supervising pupil center with ground truth')
    parser.add_argument('--loss_w_supervise_gaze_vector_3D_L2', type=float, default=0.0,
                        help='loss component for supervising gaze vector with ground truth')
    parser.add_argument('--loss_w_supervise_gaze_vector_3D_cos_sim', type=float, default=0.0,
                        help='loss component for supervising gaze vector with ground truth')
    parser.add_argument('--loss_w_supervise_gaze_vector_UV', type=float, default=0.0,
                        help='loss component for supervising gaze vector with ground truth')
    parser.add_argument('--loss_w_ellseg', type=float, default=0.0,
                        help='loss component weight')
    parser.add_argument('--loss_rend_vectorized', action='store_true', default=False,
                        help='compute segmentation head')
    parser.add_argument('--temp_n_angles', type=int, default=100,
                        help='number of discrete angles in template')
    parser.add_argument('--temp_n_radius', type=int, default=50,
                        help='number of discrete radiuses in template')
    parser.add_argument('--loss_w_rend_gt_2_pred', type=float, default=0.0,
                        help='loss component weight 0.15')
    parser.add_argument('--loss_w_rend_pred_2_gt', type=float, default=0.0,
                        help='loss component weight 0.15')
    parser.add_argument('--loss_w_rend_pred_2_gt_edge', type=float, default=0.0,
                        help='loss component weight 0.15')
    parser.add_argument('--loss_w_rend_diameter', type=float, default=0.0,
                        help='loss component weight 0.05')
    parser.add_argument('--random_dataloader', action='store_true', default=False,
                        help='shuffle the images')
    
    parser.add_argument('--scale_bound_eye', type=str, default='version_0',
                        help='load weights')


    # %% Pretrained conditions
    parser.add_argument('--weights_path', type=str, default=None,
                        help='load weights')
    parser.add_argument('--pretrained', type=int, default=0,
                        help='load weights from model\
                            pretrained on full datasets')
    parser.add_argument('--pretrained_resnet', action='store_true', default=False,
                        help='load weights from model\
                            pretrained resnet from timm')
    parser.add_argument('--optimizer_type', type=str, default='LAMB',
                        help='test only mode')
    parser.add_argument('--only_test', type=str, default=0,
                        help='test only mode')
    parser.add_argument('--only_valid', type=str, default=0,
                        help='validation only mode')
    parser.add_argument('--only_train', type=str, default=0,
                        help='train only mode')

    # %% General parameters
    parser.add_argument('--workers', type=int, default=8,
                        help='# of workers')
    parser.add_argument('--num_batches_to_plot', type=int, default=10,
                        help='number of batches to plot')
    parser.add_argument('--detect_anomaly', type=int, default=0,
                        help='enable anomaly detection?')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0,
                        help='to enable clipping, enter a norm value')
    parser.add_argument('--num_samples_for_embedding', type=int, default=200,
                        help='batches for t-SNE projection')
    parser.add_argument('--do_distributed', type=int, default=0,
                        help='move to distributed training?')
    parser.add_argument('--dry_run', action='store_true',
                        help="run a single epoch with entire train/valid sets")
    parser.add_argument('--save_test_maps', action='store_true',
                        help='save out test maps')
    
    #metric selection for early stopping : 3D or 2D gaze vector or IoU
    parser.add_argument('--early_stop_metric', type=str, default='3D',
                        help='metric selection for early stopping')

    # %% Test only conditions
    parser.add_argument('--save_results_here', type=str, default='',
                        help='if path is provided, it will override path to \
                        save the final test results')

    # %% Parse arguments
    args = parser.parse_args()
    
    if args.groups != 1:
        # We require even number of channels in all conv layers. This can be
        # achieved by setting growth factor to 1.5
        args.growth_rate = 1.5

    if args.mode == 'one_vs_one':
        print('One vs One mode')
        args.num_sets = 1

    if args.mode == 'all_vs_one':
        print('All vs one mode detected. Ignoring cur_obj flag.')
        args.cur_obj = 'allvsone'
        args.num_sets = 9

    if args.mode == 'pretrained':
        print('Pretrain mode detected.')
        args.cur_obj = 'pretrained'
        args.num_sets = 4

    if args.mode == 'all-one_vs_one':
        args.num_sets = 8

    if args.one_by_one_ds:
        print('Disabling spike removal')
        args.remove_spikes = 0

    if args.dry_run:
        args.epochs = 1

    if args.make_uncertain:
        args.make_aleatoric = True
        args.dropout = 0.2

    print('{} sets detected'.format(args.num_sets))

    opt = vars(args)
    print('---------')
    print('Parsed arguments')
    pprint(opt)
    return args