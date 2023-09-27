#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

import os
import torch
import pickle
import logging
import wandb

import timm
from timm.optim import Lamb as Lamb_timm
from timm.scheduler import CosineLRScheduler as CosineLRScheduler_timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models_mux import model_dict
from torch import autograd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts import forward
import helperfunctions.CurriculumLib as CurLib
from helperfunctions.CurriculumLib import DataLoader_riteyes
from helperfunctions.helperfunctions import mod_scalar
from helperfunctions.utils import EarlyStopping, make_logger
from helperfunctions.utils import SpikeDetection, get_nparams
from helperfunctions.utils import move_to_single, FRN_TLU, do_nothing


def train(args, path_dict, validation_mode=False, test_mode=False):

    rank_cond = (args['local_rank'] == 0) or not args['do_distributed']
    rank_cond_early_stop = rank_cond 
    #deactivate tensorboard log
    rank_cond = False


    net_dict = []

    # %% Load model
    if args['use_frn_tlu']:
        net = model_dict[args['model']](args,
                                        norm=FRN_TLU,
                                        act_func=do_nothing)
    elif args['use_instance_norm']:
        if args['model'] == 'DenseElNet':
            norm = nn.InstanceNorm3d
        else:
            norm = nn.InstanceNorm2d
        net = model_dict[args['model']](args,
                                        norm=norm,
                                        act_func=F.leaky_relu)
    elif args['use_group_norm']:
        norm = 'group_norm'
        net = model_dict[args['model']](args,
                                        norm=norm,
                                        act_func=F.leaky_relu)
    elif args['use_ada_instance_norm']:
        if args['model'] == 'DenseElNet':
            norm = nn.InstanceNorm3d
        else:
            norm = nn.InstanceNorm2d
        net = model_dict[args['model']](args,
                                        norm=norm,
                                        act_func=F.leaky_relu)
    elif args['use_ada_instance_norm_mixup']:
        if args['model'] == 'DenseElNet':
            norm = nn.InstanceNorm3d
        else:
            norm = nn.InstanceNorm2d
        net = model_dict[args['model']](args,
                                        norm=norm,
                                        act_func=F.leaky_relu)
    else:
        if args['model'] == 'DenseElNet':
            norm = nn.BatchNorm3d
        else:
            norm = nn.BatchNorm2d
        net = model_dict[args['model']](args,
                                        norm=norm,
                                        act_func=F.leaky_relu)

    # %% Weight loaders
    # if it is pretrained, then load pretrained weights
    if args['pretrained'] or args['continue_training'] or args['weights_path']:

        if args['weights_path']:
            path_pretrained = args['weights_path']
        elif args['pretrained']:
            path_pretrained = os.path.join(path_dict['repo_root'],
                                           '..',
                                           'pretrained',
                                           'pretrained.git_ok')
        elif args['continue_training']:
            path_pretrained = os.path.join(args['continue_training'])


        net_dict = torch.load(path_pretrained,
                              map_location=torch.device('cpu'))
        state_dict_single = move_to_single(net_dict['state_dict'])
        net.load_state_dict(state_dict_single, strict=False)
        print(f'Pretrained model loaded from: {path_pretrained}')

    if test_mode:
        print('Test mode detected. Loading best model.')
        
        if args['path_model']:
            net_dict = torch.load(args['path_model'],
                                  map_location=torch.device('cpu'))
        else:
            net_dict = torch.load(os.path.join(path_dict['results'],
                                               'last.pt'),
                                  map_location=torch.device('cpu'))

        # Ensure saved arguments match with parsed arguments
        state_dict_single = move_to_single(net_dict['state_dict'])
        net.load_state_dict(state_dict_single, strict=False)

        # Do not initialize a writer
        writer = []
    elif validation_mode:
        print('Validation mode detected. Loading model.')
        if args['path_model']:
            net_dict = torch.load(args['path_model'],
                                  map_location=torch.device('cpu'))
        else:
            net_dict = torch.load(os.path.join(path_dict['results'],
                                               'last.pt'),
                                  map_location=torch.device('cpu'))

        # Ensure saved arguments match with parsed arguments
        state_dict_single = move_to_single(net_dict['state_dict'])
        net.load_state_dict(state_dict_single, strict=False)

        # Do not initialize a writer
        writer = []
    else:
        # Initialize tensorboard if rank 0
        if rank_cond:
            writer = SummaryWriter(path_dict['logs'])
        else:
            writer = []


    if args['use_GPU']:
        net.cuda()

    # %% move network to DDP
    if args['do_distributed']:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net,
                  device_ids=[args['local_rank']],
                  find_unused_parameters=True)

    # %% Initialize logger
    logger = make_logger(path_dict['logs']+'/train_log.log',
                         rank=args['local_rank'] if args['do_distributed'] else 0)
    logger.write_summary(str(net.parameters))
    logger.write('# of parameters: {}'.format(get_nparams(net)))

    if test_mode==0 and validation_mode==0:
        logger.write('Training!')
        if args['exp_name'] != 'DEBUG':
            wandb.watch(net)
    elif validation_mode:
        logger.write('Validating!')
    else:
        logger.write('Testing!')

    # %% Training and validation loops or test only
    train_validation_loops(net,
                           net_dict,
                           logger,
                           args,
                           path_dict,
                           writer,
                           rank_cond,
                           rank_cond_early_stop,
                           validation_mode,
                           test_mode)

    # %% Closing functions and logging
    if writer:
        writer.close()


def train_validation_loops(net, net_dict, logger, args,
                           path_dict, writer, rank_cond, 
                           rank_cond_early_stop, 
                           validation_mode, test_mode):


    #specify to use a pkl or create the dataloader here
    if args['use_pkl_for_dataload']:
        # %% Load curriculum objects
        path_cur_obj = os.path.join(path_dict['repo_root'],
                                    'cur_objs',
                                    args['mode'],
                                    'cond_'+args['cur_obj']+'.pkl')
        with open(path_cur_obj, 'rb') as f:
            train_obj, valid_obj, test_obj = pickle.load(f)
    else:
        path_cur_obj = os.path.join(path_dict['repo_root'],
                                    'cur_objs',
                                    'dataset_selections'+'.pkl')
        path2h5 = args['path_data']
        #TODO check the open since can create memory leaks
        DS_sel = pickle.load(open(path_cur_obj, 'rb'))
        AllDS = CurLib.readArchives(args['path2MasterKey'])

        if (args['cur_obj']=='OpenEDS_S'):
            sel = 'S'
        else:
            sel = args['cur_obj']

        if (args['cur_obj']=='Ours'):
            #Train and Validation object
            AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train'][sel])
            dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=False)
            train_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640), 
                                            scale=0.5, num_frames=args['frames'], args=args)
            valid_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640), sort='nothing',
                                            scale=0.5, num_frames=args['frames'], args=args)
                
            # Test object
            AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][sel])
            dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=False)
            test_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'test', False, (480, 640), sort='nothing', 
                                            scale=0.5, num_frames=args['frames'], args=args)
            
        else:
            if 'DEBUG' in args['exp_name']:
                with open ('cur_objs/dataDiv_obj_train.pkl', 'rb') as f:
                    dataDiv_obj = pickle.load(f)
                    # with open('train_names.txt', 'w') as fhh:
                    #     for s in dataDiv_obj.arch:
                    #         fhh.write(s+'\n')
            else:
                #Train and Validation object
                AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train'][sel])
                dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='vanilla', notest=False)
            # file_name = 'dataDiv_obj_train.pkl'
            # with open(file_name, 'wb') as file:
            #     pickle.dump(dataDiv_obj, file)
            #     print(f'Pickle saved "{file_name}"')
            #     print(os.getcwd())
            train_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640), 
                                            scale=0.5, num_frames=args['frames'], args=args)
            valid_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640), sort='nothing', 
                                            scale=0.5, num_frames=args['frames'], args=args)
                

            if 'DEBUG' in args['exp_name']:
                with open ('cur_objs/dataDiv_obj_test.pkl', 'rb') as f:
                    dataDiv_obj = pickle.load(f)                    
                    # with open('test_names.txt', 'w') as fhh:
                    #     for s in dataDiv_obj.arch:
                    #         fhh.write(s+'\n')
            else:
                # Test object
                AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][sel])
                dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=False)           
            # file_name = 'dataDiv_obj_test.pkl'
            # with open(file_name, 'wb') as file:
            #     pickle.dump(dataDiv_obj, file)
            #     print(f'Pickle saved "{file_name}"')
            #     print(os.getcwd())
            test_obj = DataLoader_riteyes(dataDiv_obj, path2h5, 'test', False, (480, 640), sort='nothing', 
                                            scale=0.5, num_frames=args['frames'], args=args)
            #if 'DEBUG' not in args['exp_name']:
            #  args['batches_per_ep'] = train_obj.__len__()
    # FIXME Remove unwanted validation-train overlap
    print(f'Starting the procedure of removing unwanted train-val video overlap...')
    train_vid_ids = list(np.unique(train_obj.imList[:, :, 1]))
    for vid_id in np.unique(valid_obj.imList[:, :, 1]):
        #print(vid_id)
        if vid_id in train_vid_ids:
            print(f'Discarded valid overlap video_id:{vid_id}')
            bad_ids = ((valid_obj.imList[:, :, 1] == vid_id).sum(axis=-1) > 0)
            valid_obj.imList = valid_obj.imList[~bad_ids]
    # FIXME Subselect 100k test images 
    print('Sub-selecting first 100k test frames')
    test_cutoff = int(100000 / test_obj.imList.shape[1])
    test_obj.imList = test_obj.imList[:test_cutoff]

    print(f'')
    print(f'')
    print(f'Number of images:')
    print(f'Train images left: {train_obj.imList.shape[0]*train_obj.imList.shape[1]}')
    print(f'Valid images left: {valid_obj.imList.shape[0]*valid_obj.imList.shape[1]}')
    print(f'Test images left: {test_obj.imList.shape[0]*test_obj.imList.shape[1]}')
    print(f'')
    print(f'')

    # %% Specify flags of importance
    train_obj.augFlag = args['aug_flag']
    valid_obj.augFlag = False
    test_obj.augFlag = False

    train_obj.equi_var = args['equi_var']
    valid_obj.equi_var = args['equi_var']
    test_obj.equi_var = args['equi_var']

    # %% Modify path information
    train_obj.path2data = path_dict['path_data']
    valid_obj.path2data = path_dict['path_data']
    test_obj.path2data = path_dict['path_data']

    # %% Modify scale at which we are working
    train_obj.scale = args['scale_factor']
    valid_obj.scale = args['scale_factor']
    test_obj.scale = args['scale_factor']

    # %% Create distributed samplers
    train_sampler = DistributedSampler(train_obj,
                                       rank=args['local_rank'],
                                       shuffle=False,
                                       num_replicas=args['world_size'],
                                       )

    valid_sampler = DistributedSampler(valid_obj,
                                       rank=args['local_rank'],
                                       shuffle=False,
                                       num_replicas=args['world_size'],
                                       )

    test_sampler = DistributedSampler(test_obj,
                                      rank=args['local_rank'],
                                      shuffle=False,
                                      num_replicas=args['world_size'],
                                      )

    # %% Define dataloaders
    logger.write('Initializing loaders')
    if validation_mode:
        valid_loader = DataLoader(valid_obj,
                                  shuffle=False,
                                  num_workers=args['workers'],
                                  drop_last=True,
                                  pin_memory=True,
                                  batch_size=args['batch_size'],
                                  sampler=valid_sampler if args['do_distributed'] else None,
                                  )
    elif test_mode:
        test_loader = DataLoader(test_obj,
                                shuffle=False,
                                num_workers=0,
                                drop_last=True,
                                batch_size=args['batch_size'],
                                sampler=test_sampler if args['do_distributed'] else None,
                                )
    else:
        train_loader = DataLoader(train_obj,
                                  shuffle=args['random_dataloader'],
                                  num_workers=args['workers'],
                                  drop_last=True,
                                  pin_memory=True,
                                  batch_size=args['batch_size'],
                                  sampler=train_sampler if args['do_distributed'] else None,
                                  )

        valid_loader = DataLoader(valid_obj,
                                  shuffle= False,
                                  num_workers=args['workers'],
                                  drop_last=True,
                                  pin_memory=True,
                                  batch_size=args['batch_size'],
                                  sampler=valid_sampler if args['do_distributed'] else None,
                                  )

    # %% Early stopping criterion
    if '3D' in args['early_stop_metric'] or '2D' in args['early_stop_metric']:
        early_stop = EarlyStopping(metric=args['early_stop_metric'],
                                patience=args['early_stop'],
                                verbose=True,
                                delta=0.001,  # 0.1% improvement needed
                                rank_cond=rank_cond_early_stop,
                                mode='min',
                                fName='best_model.pt',
                                path_save=path_dict['results'],
                                )
    else:
        early_stop = EarlyStopping(metric=args['early_stop_metric'],
                                patience=args['early_stop'],
                                verbose=True,
                                delta=0.001,  # 0.1% improvement needed
                                rank_cond=rank_cond_early_stop,
                                mode='max',
                                fName='best_model.pt',
                                path_save=path_dict['results'],
                                )

    # %% Define alpha and beta scalars
    if args['curr_learn_losses']:
        alpha_scalar = mod_scalar([0, args['epochs']], [0, 1])
        beta_scalar = mod_scalar([10, 20], [0, 1])

    # %% Optimizer
    param_list = [param for name, param in net.named_parameters() if 'adv' not in name]
    if 'LAMB' in args['optimizer_type']:
        optimizer = Lamb_timm(param_list, lr=args['lr'], weight_decay=args['wd']) #wd=0.02
        warmup_epochs = 5
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=2, eta_min=args['lr']/100.0)
        scheduler = CosineLRScheduler_timm(optimizer, 
                                                     t_initial=args['epochs'], 
                                                     lr_min=args['lr']/10.0**2, 
                                                     warmup_t=4, 
                                                     warmup_lr_init=args['lr']/10.0**2)
        use_sched = True
    elif 'adamw_cos' in args['optimizer_type']:
        optimizer = torch.optim.AdamW(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        warmup_epochs = 5
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=2, eta_min=args['lr']/100.0)
        scheduler = CosineLRScheduler_timm(optimizer, 
                                                     t_initial=args['epochs'], 
                                                     lr_min=args['lr']/10.0**2, 
                                                     warmup_t=4, 
                                                     warmup_lr_init=args['lr']/10.0**2)
        use_sched = True
    elif 'adamw_step' in args['optimizer_type']:
        optimizer = torch.optim.AdamW(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        use_sched = True
    elif 'adam_cos' in args['optimizer_type']:
        optimizer = torch.optim.Adam(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        warmup_epochs = 5
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=2, eta_min=args['lr']/100.0)
        scheduler = CosineLRScheduler_timm(optimizer, 
                                                     t_initial=args['epochs'], 
                                                     lr_min=args['lr']/10.0**3, 
                                                     warmup_t=4, 
                                                     warmup_lr_init=args['lr']/10.0**2)
        use_sched = True
    elif 'adam_step' in args['optimizer_type']:
        optimizer = torch.optim.Adam(param_list, lr=args['lr'], betas=(0.9, 0.99), weight_decay=args['wd'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        use_sched = True
    else:
        optimizer = torch.optim.Adam(param_list,
                                 lr=args['lr'], amsgrad=False)
        use_sched = False

    if args['adv_DG']:
        param_list = [param for name, param in net.named_parameters() if 'adv' in name]
        optimizer_disc = torch.optim.Adam(param_list,
                                          lr=args['lr'],
                                          amsgrad=True)
    else:
        optimizer_disc = False

    # %% Loops and what not

    # Create a checkpoint based on current scores
    checkpoint = {}
    checkpoint['args'] = args  # Save arguments

    # Randomize the dataset again the next time you exit
    # to the main loop.
    args['time_to_update'] = True
    last_epoch_validation = False

    #speficy the mode for forward and save results
    if test_mode:
        logging.info('Entering test mode only ...')
        logger.write('Entering test mode only ...')

        args['alpha'] = 0.5
        args['beta'] = 0.5

        test_result = forward(net,
                                [],
                                logger,
                                test_loader,
                                optimizer,
                                args,
                                path_dict,
                                writer=writer,
                                rank_cond=rank_cond,
                                epoch=0,
                                mode='test',
                                batches_per_ep=len(test_loader) if 'DEBUG' not in args['exp_name'] else 10,
                                last_epoch_valid=True,
                                csv_save_dir=path_dict['exp'])

        checkpoint['test_result'] = test_result

        epoch = 0
        if args['exp_name'] != 'DEBUG':
            logger.write('Test results:')
            for key, item in checkpoint['test_result'].items():
                if 'mean' in key and 'loss' not in key:
                    wandb.log({'test_avg/{}'.format(key): item, 'epoch':epoch,})
                    logger.write(f'test_avg/{key}: {item:.3f}')
            logger.write(' ')
            logger.write(' ')
            logger.write(' ')


        if args['save_results_here']:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(args['save_results_here']),
                        exist_ok=True)

            # Save out test results here instead
            with open(args['save_results_here'], 'wb') as f:
                pickle.dump(checkpoint, f)
        else:
            # Ensure the directory exists
            os.makedirs(path_dict['results'], exist_ok=True)

            # Save out the test results
            with open(path_dict['results'] + '/test_results.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)

    elif validation_mode:
        logging.info('Entering validation mode  only...')
        args['alpha'] = 0.5
        args['beta'] = 0.5

        valid_result = forward(net,
                                [],
                                logger,
                                valid_loader,
                                optimizer,
                                args,
                                path_dict,
                                writer=writer,
                                rank_cond=rank_cond,
                                epoch=0,
                                mode='valid',
                                batches_per_ep=len(valid_loader) if 'DEBUG' not in args['exp_name'] else 10,
                                last_epoch_valid=True,
                                csv_save_dir=path_dict['exp'])

        checkpoint['valid_result'] = valid_result
        
        epoch = 0
        if args['exp_name'] != 'DEBUG':
            logger.write('Validation results:')
            for key, item in checkpoint['valid_result'].items():
                if 'mean' in key and 'loss' not in key:
                    wandb.log({'valid_avg/{}'.format(key): item, 'epoch':epoch,})
                    logger.write(f'valid_avg/{key}: {item:.3f}')
            logger.write(' ')
            logger.write(' ')
            logger.write(' ')
        
        if args['save_results_here']:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(args['save_results_here']),
                        exist_ok=True)

            # Save out test results here instead
            with open(args['save_results_here'], 'wb') as f:
                pickle.dump(checkpoint, f)
        else:
            # Ensure the directory exists
            os.makedirs(path_dict['results'], exist_ok=True)

            # Save out the test results
            with open(path_dict['results'] + '/valid_results.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
    
    else:
        spiker = SpikeDetection() if args['remove_spikes'] else False
        logging.info('Entering train mode ...')

        if args['continue_training']:
            optimizer.load_state_dict(net_dict['optimizer'])
            epoch = net_dict['epoch'] + 1
        else:
            epoch = 0

        # Disable early stop and keep training until it maxes out, this allows
        # us to test at the regular best model while saving intermediate result
        #while (epoch < args['epochs']) and not early_stop.early_stop:

        while (epoch < args['epochs']):
            if args['time_to_update']:

                # Toggle flag back to False
                args['time_to_update'] = False

                if args['one_by_one_ds']:
                    train_loader.dataset.sort('one_by_one_ds', args['batch_size'])
                    valid_loader.dataset.sort('one_by_one_ds', args['batch_size'])
                else:
                    #for sequence dataset:
                    train_loader.dataset.sort('ordered')
                    valid_loader.dataset.sort('ordered')

            # Set epochs for samplers
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

            # %%
            logging.info('Starting epoch: %d' % epoch)

            if args['curr_learn_losses']:
                args['alpha'] = alpha_scalar.get_scalar(epoch)
                args['beta'] = beta_scalar.get_scalar(epoch)
            else:
                args['alpha'] = 0.5
                args['beta'] = 0.5

            if args['dry_run']:
                train_batches_per_ep = len(train_loader)
                valid_batches_per_ep = len(valid_loader)
            else:
                train_batches_per_ep = args['batches_per_ep']
                if args['reduce_valid_samples']:
                    valid_batches_per_ep = 10
                else:
                    valid_batches_per_ep = int(20000 /(args['frames'] * args['batch_size']))

            train_result = forward(net,
                                    spiker,
                                    logger,
                                    train_loader,
                                    optimizer,
                                    args,
                                    path_dict,
                                    optimizer_disc=optimizer_disc,
                                    writer=writer,
                                    rank_cond=rank_cond,
                                    epoch=epoch,
                                    mode='train',
                                    batches_per_ep=train_batches_per_ep)
            
            if epoch == args['epochs'] - 1:
                last_epoch_validation = True
                valid_batches_per_ep = len(valid_loader)

            #incase you want to validate the whole validation set just Remove True
            if (args['reduce_valid_samples'] and (epoch%args['perform_valid'] != 0)) \
                    or ('DEBUG' in args['exp_name']) or True:
                valid_result = forward(net,
                                       spiker,
                                       logger,
                                       valid_loader,
                                       optimizer,
                                       args,
                                       path_dict,
                                       writer=writer,
                                       rank_cond=rank_cond,
                                       epoch=epoch,
                                       mode='valid',
                                       batches_per_ep=valid_batches_per_ep,
                                       last_epoch_valid=last_epoch_validation)
            else:
                valid_result = forward(net,
                                       spiker,
                                       logger,
                                       valid_loader,
                                       optimizer,
                                       args,
                                       path_dict,
                                       writer=writer,
                                       rank_cond=rank_cond,
                                       epoch=epoch,
                                       mode='valid',
                                       batches_per_ep=len(valid_loader),
                                       last_epoch_valid=last_epoch_validation)
                


            # Update the check point weights. VERY IMPORTANT!
            checkpoint['state_dict'] = move_to_single(net.state_dict())
            checkpoint['optimizer'] = optimizer.state_dict()

            checkpoint['epoch'] = epoch
            checkpoint['train_result'] = train_result
            checkpoint['valid_result'] = valid_result
                  
            if args['exp_name'] != 'DEBUG':
                logger.write('Validation results:')
                for key, item in checkpoint['valid_result'].items():
                    if 'mean' in key and 'loss' not in key:
                        wandb.log({'valid_avggg/{}'.format(key): item, 'epoch':epoch,})
                        logger.write(f'valid_avggg/{key}: {item:.3f}')
                logger.write(' ')
                logger.write(' ')
                logger.write(' ')

            # Save out the best validation result and model
            early_stop(checkpoint)

            if args['exp_name'] != 'DEBUG':
                wandb.log({'val_score': checkpoint['valid_result']['score_mean'], 'epoch': epoch})
                wandb.log({'gaze_3D_ang_deg_mean': checkpoint['valid_result']['gaze_3D_ang_deg_mean'], 'epoch': epoch})

            # If epoch is a multiple of args['save_every'], then write out
            if (epoch%args['save_every']) == 0:

                # Ensure that you do not update the validation score at this
                # point and simply save the model
                if '3D' in args['early_stop_metric']:
                    temp_score = checkpoint['valid_result']['gaze_3D_ang_deg_mean']
                elif '2D' in args['early_stop_metric']:
                    temp_score = checkpoint['valid_result']['gaze_ang_deg_mean']
                else:
                    temp_score = checkpoint['valid_result']['masked_rendering_iou_mean']
                early_stop.save_checkpoint(temp_score,
                                           checkpoint,
                                           update_val_score=False,
                                           use_this_name_instead='last.pt')

            if use_sched:
                scheduler.step(epoch=epoch)
                if args['exp_name'] != 'DEBUG':
                    wandb.log({'lr': optimizer.param_groups[0]['lr']})
            epoch += 1


if __name__ == '__main__':
    print('Entry script is run.py')
