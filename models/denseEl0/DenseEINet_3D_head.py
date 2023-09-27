#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from einops import rearrange

from helperfunctions.loss import get_com
from models.denseEl0.DenseEINet_3D_head_utils import driver_ada_instance
from models.denseEl0.DenseEINet_3D_head_utils import renderingregressionModule    
from models.denseEl0.DenseEINet_3D_head_utils import eye_param_same_regressionModule    
from models.denseEl0.DenseEINet_3D_head_utils import conv_layer
from models.denseEl0.DenseEINet_3D_head_utils import regressionModule, linStack, convBlock

from helperfunctions.utils import detach_cpu_np, get_nparams

from helperfunctions.helperfunctions import plot_images_with_annotations
from helperfunctions.helperfunctions import construct_mask_from_ellipse
from helperfunctions.helperfunctions import fix_ellipse_axis_angle
from helperfunctions.helperfunctions import my_ellipse

from extern.pytorch_revgrad.src.module import RevGrad
from extern.squeeze_and_excitation.squeeze_and_excitation.squeeze_and_excitation import ChannelSpatialSELayer


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


class Transition_down(nn.Module):
    '''
    Downsampling block which uses average pooling to reduce spatial dimensions
    '''

    def __init__(self, down_size):
        super(Transition_down, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=down_size) if down_size else False

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        return x


class DenseNet2D_down_block(nn.Module):
    '''
    An encoder block inspired from DenseNet.
    '''

    def __init__(self,
                 in_c,
                 inter_c,
                 op_c,
                 down_size,
                 norm,
                 act_func,
                 scSE=False,  # Concurrent spatial and channel-wise SE
                 groups=1,  # 'channels'
                 dropout=False,
                 track_running_stats=False,
                 args=None):

        super(DenseNet2D_down_block, self).__init__()

        self.num__of_frames = args['frames']  

        self.conv1 = conv_layer(in_c, inter_c, norm, act_func,
                                groups=groups,
                                dropout=dropout,
                                kernel_size=3, bias=False, padding=1,
                                track_running_stats=track_running_stats)

        self.conv21 = conv_layer(in_c+inter_c, inter_c, norm, act_func,
                                 groups=groups,
                                 dropout=dropout,
                                 kernel_size=1, bias=False, padding=0,
                                 track_running_stats=track_running_stats)

        self.conv22 = conv_layer(inter_c, inter_c, norm, act_func,
                                 groups=groups,
                                 dropout=dropout,
                                 kernel_size=3, bias=False, padding=1,
                                 track_running_stats=track_running_stats)

        self.conv31 = conv_layer(in_c+2*inter_c, inter_c, norm, act_func,
                                 groups=groups,
                                 kernel_size=1, bias=False, padding=0,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)

        self.conv32 = conv_layer(inter_c, inter_c, norm, act_func,
                                 groups=groups,
                                 kernel_size=3, bias=False, padding=1,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)

        self.conv4 = conv_layer(in_c+inter_c, op_c, norm, act_func,
                                groups=groups,
                                kernel_size=1, bias=False, padding=0,
                                dropout=dropout,
                                track_running_stats=track_running_stats)

        self.TD = Transition_down(down_size)

        if scSE:
            self.csE_layer = ChannelSpatialSELayer(in_c+inter_c,
                                                   reduction_ratio=8)
        else:
            self.csE_layer = False

    def forward(self, x):
        # NOTE: input x is assumed to be batchnorm'ed

        y = self.conv1(x)

        y = torch.cat([x, y], dim=1)
        out = self.conv22(self.conv21(y))

        out = torch.cat([y, out], dim=1)
        out = self.conv32(self.conv31(out))

        out = torch.cat([out, x], dim=1)

        if self.csE_layer:
            out = self.csE_layer(out)

        return out, self.TD(self.conv4(out))


class DenseNet2D_up_block(nn.Module):
    '''
    A lightweight decoder block which upsamples spatially using
    bilinear interpolation.
    '''

    def __init__(self, skip_c, in_c, out_c,
                 up_stride, act_func, norm,
                 scSE=False,
                 groups=1,
                 dropout=False,
                 track_running_stats=False,
                 args=None):

        super(DenseNet2D_up_block, self).__init__()

        self.num__of_frames = args['frames']

        self.conv11 = conv_layer(skip_c+in_c, out_c, norm, act_func,
                                 groups=groups,
                                 kernel_size=1, bias=False, padding=0,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.conv12 = conv_layer(out_c, out_c, norm, act_func,
                                 groups=groups,
                                 kernel_size=3, bias=False, padding=1,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.conv21 = conv_layer(skip_c+in_c+out_c, out_c, norm, act_func,
                                 groups=groups,
                                 kernel_size=1, bias=False, padding=0,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.conv22 = conv_layer(out_c, out_c, norm, act_func,
                                 groups=groups,
                                 kernel_size=3, bias=False, padding=1,
                                 dropout=dropout,
                                 track_running_stats=track_running_stats)
        self.up_stride = up_stride

        if scSE:
            self.csE_layer = ChannelSpatialSELayer(out_c)
        else:
            self.csE_layer = False

    def forward(self, prev_feature_map, x):
        # NOTE: inputs are assumed to be norm'ed

        x = F.interpolate(x,
                          mode='bilinear',
                          align_corners=False,
                          scale_factor=self.up_stride)

        x = torch.cat([x, prev_feature_map], dim=1)
        out = self.conv12(self.conv11(x))

        out = torch.cat([x, out], dim=1)
        out = self.conv22(self.conv21(out))

        if self.csE_layer:
            out = self.csE_layer(out)

        return out


class DenseNet_encoder(nn.Module):
    def __init__(self, args, in_c=1,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):

        super(DenseNet_encoder, self).__init__()

        self.num__of_frames = args['frames']

        chz = args['base_channel_size']
        growth = args['growth_rate']
        track_running_stats = args['track_running_stats']

        sizes = getSizes(chz, growth, blks=args['num_blocks'])

        opSize = sizes['enc']['op']
        ipSize = sizes['enc']['ip']
        interSize = sizes['enc']['inter']

        self.head = convBlock(in_c=in_c,
                              inter_c=chz,
                              out_c=chz,
                              act_func=act_func,
                              num_layers=1,
                              norm=nn.BatchNorm3d,
                              track_running_stats=track_running_stats,
                              args=args,
                              conv_3D=True)

        self.down_block_list = nn.ModuleList([])

        for block_num in range(args['num_blocks']):
            block = DenseNet2D_down_block(in_c=ipSize[block_num],
                                          inter_c=interSize[block_num],
                                          op_c=opSize[block_num],
                                          down_size=2,
                                          scSE=args['use_scSE'],
                                          norm=norm,
                                          groups=args['groups'],
                                          act_func=act_func,
                                          dropout=args['dropout'],
                                          track_running_stats=track_running_stats,
                                          args=args)
            self.down_block_list.append(block)

        # Recalibration using adaptive instance normalization
        if args['use_ada_instance_norm']:
            self.recalib = conv_layer(in_channels=opSize[-1],
                                      out_channels=opSize[-1],
                                      kernel_size=1,
                                      padding=0,
                                      norm_layer='ada_instance_norm')
        elif args['use_ada_instance_norm_mixup']:
            self.recalib = conv_layer(in_channels=opSize[-1],
                                      out_channels=opSize[-1],
                                      kernel_size=1,
                                      padding=0,
                                      norm_layer='ada_instance_norm_mixup')
        else:
            self.recalib = None

    def forward(self, x):
        x = self.head(x)

        x = x.squeeze(2)

        skip_list = []
        for block in self.down_block_list:
            skip, x = block(x)
            skip_list.append(skip)

        if self.recalib:
            x = self.recalib(x)

        return tuple(skip_list) + (x, )


class DenseNet_decoder(nn.Module):
    def __init__(self, args, out_c,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):

        chz = args['base_channel_size']
        growth = args['growth_rate']
        track_running_stats = args['track_running_stats']

        super(DenseNet_decoder, self).__init__()
        sizes = getSizes(chz, growth, blks=args['num_blocks'])
        skipSize = sizes['dec']['skip']
        opSize = sizes['dec']['op']
        ipSize = sizes['dec']['ip']

        self.up_block_list = nn.ModuleList([])
        for block_num in range(args['num_blocks']):
            block = DenseNet2D_up_block(skipSize[block_num],
                                        ipSize[block_num],
                                        opSize[block_num],
                                        2, act_func, norm,
                                        groups=args['groups'],
                                        scSE=args['use_scSE'],
                                        dropout=args['dropout'],
                                        track_running_stats=track_running_stats,
                                        args=args)
            self.up_block_list.append(block)

        self.final = nn.Conv2d(opSize[-1], out_c, kernel_size=1, bias=True)

    def forward(self, skip_list, x):

        for block_num, block in enumerate(self.up_block_list):
            x = block(skip_list[-block_num-1], x)

        return self.final(x)


class DenseNet3D_head(nn.Module):
    def __init__(self,
                 args,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):
        super(DenseNet3D_head, self).__init__()

        self.num_of_frames = args['frames']

        self.args = args

        self.alpha = 1.0

        self.extra_depth = args['extra_depth']

        self.sizes = getSizes(args['base_channel_size'],
                              args['growth_rate'],
                              blks=args['num_blocks'])

        self.equi_var = args['equi_var']

        self.enc = DenseNet_encoder(args,
                                    in_c=1,
                                    act_func=act_func, norm=norm)

        # Fix the decoder norm to IN unless adopting AdaIN. The original paper
        # of AdaIN recommends disabling normalization for the decoder.
        if args['use_ada_instance_norm'] or args['use_ada_instance_norm_mixup']:
            # Adaptive Instance norm suggests not using any normalization in
            # the decoder when paired with adaptive instance normalization. It
            # is a design choice which improves performance apparantly.
            self.dec = DenseNet_decoder(args,
                                        out_c=3*self.num_of_frames,
                                        act_func=act_func,
                                        norm='none')
        else:
            self.dec = DenseNet_decoder(args,
                                        out_c=3*self.num_of_frames,
                                        act_func=act_func,
                                        norm=nn.InstanceNorm2d)

        if args['maxpool_in_regress_mod'] > 0:
            regress_pool = nn.MaxPool2d
        elif args['maxpool_in_regress_mod'] == 0:
            regress_pool = nn.AvgPool2d
        else:
            regress_pool = False

        # Fix the Regression module to InstanceNorm
        self.elReg = regressionModule(self.sizes,
                                      sc=args['regress_channel_grow'],
                                      num_of_frames = self.num_of_frames,
                                      norm=nn.InstanceNorm2d,
                                      pool=regress_pool,
                                      dilate=args['dilation_in_regress_mod'],
                                      act_func=act_func,
                                      track_running_stats=args['track_running_stats'])

        # Fix the Regression module for the rendering process
        self.renderingReg = renderingregressionModule(self.sizes,
                                                      sc=args['regress_channel_grow'],
                                                      num_of_frames = self.num_of_frames,
                                                      norm=nn.InstanceNorm2d,
                                                      pool=regress_pool,
                                                      dilate=args['dilation_in_regress_mod'],
                                                      act_func=F.relu,
                                                      act_final=F.tanh,
                                                      track_running_stats=args['track_running_stats'])

        # Fix the Regression module for the rendering process
        self.eye_param_same = eye_param_same_regressionModule(self.sizes,
                                                      sc=args['regress_channel_grow'],
                                                      num_of_frames = self.num_of_frames,
                                                      norm=nn.InstanceNorm2d,
                                                      pool=regress_pool,
                                                      dilate=args['dilation_in_regress_mod'],
                                                      act_func=F.relu,
                                                      act_final=F.tanh,
                                                      track_running_stats=args['track_running_stats'])

        self.make_aleatoric = args['make_aleatoric']

        if args['grad_rev']:
            self.grad_rev = True
            self.setDatasetInfo(args['num_sets'])
        else:
            self.grad_rev = False

    def setDatasetInfo(self, numSets=2):
        # Produces a 1x1 conv2D layer which directly maps bottleneck to the
        # appropriate dataset ID. This mapping is done for each pixel, ensuring

        inChannels = self.sizes['enc']['op'][-1]

        self.numSets = numSets
        self.grad_rev_layer = RevGrad()
        self.dsIdentify_lin = nn.Conv2d(in_channels=inChannels,
                                        out_channels=numSets,
                                        kernel_size=(self.num_of_frames,1,1),
                                        bias=True)

    def forward(self, data_dict, args):

        # Cast to float32, move to device and add a dummy color channel
        if 'device' not in self.__dict__:
            self.device = next(self.parameters()).device

        out_dict_valid = True

        # Move image data to GPU
        x = data_dict['image'].to(torch.float32).to(self.device,
                                                    non_blocking=True)
        x = x.unsqueeze(1)

        B = args['batch_size']
        NUM_OF_FRAME = args['frames']
        _, _, _, H, W = x.shape

        start = time.time() 

        enc_op = self.enc(x)

        # %% Gradient reversal on latent space
        # Observation: Adaptive Instance Norm removes domain info quite nicely
        if self.grad_rev:
            self.grad_rev_layer._alpha = torch.tensor(self.alpha,
                                                      requires_grad=False,
                                                      device=self.device)
            # Gradient reversal to remove DS bias
            ds_predict = self.dsIdentify_lin(self.grad_rev_layer(enc_op[-1]))
        else:
            ds_predict = []

        # %% Convert predicted ellipse back to proper scale
        if self.equi_var:
            sc = max([W, H])
            Xform_to_norm = np.array([[2/sc, 0, -1],
                                      [0, 2/sc, -1],
                                      [0, 0,     1]])
        else:
            Xform_to_norm = np.array([[2/W, 0, -1],
                                      [0, 2/H, -1],
                                      [0, 0,    1]])

        Xform_from_norm = np.linalg.inv(Xform_to_norm)

        out_dict = {}

        latent = enc_op[-1].flatten(start_dim=-3).mean(dim=-1)

        if args['net_ellseg_head']:
            # Generate latent space bottleneck representation
            elOut, elConf = self.elReg(enc_op[-1])
            #reshape the elOut and elCOnf to have the following [B, FRAME, OUT]
            elOut = elOut.reshape(B, NUM_OF_FRAME, -1)
            elConf = elConf.reshape(B, NUM_OF_FRAME, -1)
            op = self.dec(enc_op[:-1], enc_op[-1])  
            #reshape the tensor to get the correct number of predicted masks
            op = op.view(B,NUM_OF_FRAME,-1,op.shape[2], op.shape[3])

            if torch.any(torch.isnan(op)) or torch.any(torch.isinf(op)):
                print(data_dict['archName'])
                print(data_dict['im_num'])
                print('WARNING! Convergence failed!')

                from scripts import detach_cpu_numpy

                plot_images_with_annotations(detach_cpu_numpy(data_dict),
                                            self.args,
                                            write='./FAILURE.jpg',
                                            remove_saturated=False,
                                            is_list_of_entries=False,
                                            is_predict=False,
                                            show=False)

                import sys
                sys.exit('Network predicted NaNs or Infs')
                #print('Network predicted NaNs or Infs')
                #out_dict_valid = False


            # %% Choose EllSeg proposed ellipse measures

            # Get center of ellipses from COM
            pred_pup_c = get_com(op[:, :, 2, ...], temperature=4)
            pred_iri_c = get_com(-op[:, :, 0, ...], temperature=4)

            # Ensure batch doesn't get squeezed if == 1
            pred_pup_c = pred_pup_c.reshape(B, NUM_OF_FRAME, -1)
            pred_iri_c = pred_iri_c.reshape(B, NUM_OF_FRAME, -1)

            if torch.any(torch.isnan(pred_pup_c)) or\
            torch.any(torch.isnan(pred_iri_c)):

                import sys
                print('WARNING! Convergence failed!')
                sys.exit('Pupil or Iris centers predicted as NaNs')
                #print('WARNING! Convergence failed!')
                #out_dict_valid = False


            # Append pupil and iris ellipse parameter predictions from latent space
            pupil_ellipse_norm = torch.cat([pred_pup_c, elOut[:, :, 7:10]], dim=2)
            iris_ellipse_norm = torch.cat([pred_iri_c, elOut[:, :, 2:5]], dim=2)

            out_dict['iris_ellipse'] = np.zeros((B, NUM_OF_FRAME, 5))
            out_dict['pupil_ellipse'] = np.zeros((B, NUM_OF_FRAME, 5))

            for b in range(B):
                # Read each normalized ellipse in a loop and unnormalize it
                try:
                    ellipse_list = []
                    for frame in range(NUM_OF_FRAME):
                        temp_var = detach_cpu_np(iris_ellipse_norm[b, frame, ])
                        temp_var = fix_ellipse_axis_angle(temp_var)
                        ellipse_list.append(my_ellipse(temp_var).transform(Xform_from_norm)[0][:5])
                    temp_var = np.stack(ellipse_list,axis=0)
                except Exception:
                    print(temp_var)
                    print('Incorrect norm iris: {}'.format(temp_var.tolist()))
                    temp_var = np.ones(5, )

                out_dict['iris_ellipse'][b, ...] = temp_var

                try:
                    ellipse_list = []
                    for frame in range(NUM_OF_FRAME):
                        temp_var = detach_cpu_np(pupil_ellipse_norm[b, frame, ])
                        temp_var = fix_ellipse_axis_angle(temp_var)
                        ellipse_list.append(my_ellipse(temp_var).transform(Xform_from_norm)[0][:5])
                    temp_var = np.stack(ellipse_list,axis=0)
                except Exception:
                    print(temp_var)
                    print('Incorrect norm pupil: {}'.format(temp_var.tolist()))
                    temp_var = np.ones(5, )

                out_dict['pupil_ellipse'][b, ...] = temp_var



            # %% Pupil and Iris mask construction from predicted ellipses
            pupil_mask_recon = construct_mask_from_ellipse(out_dict['pupil_ellipse'], (H,W))
            iris_mask_recon = construct_mask_from_ellipse(out_dict['iris_ellipse'], (H,W))

            pd_recon_mask = np.zeros(pupil_mask_recon.shape, dtype=int)
            pd_recon_mask[iris_mask_recon.astype(bool)] = 1
            pd_recon_mask[pupil_mask_recon.astype(bool)] = 2
            pd_recon_mask = rearrange(pd_recon_mask, 'b f h w -> (b f) h w')

            # %% Save out predicted data and return
            out_dict['mask'] = torch.argmax(op, dim=2).detach().cpu().numpy()

            out_dict['mask_recon'] = pd_recon_mask

            out_dict['pupil_ellipse_norm'] = pupil_ellipse_norm
            out_dict['iris_ellipse_norm'] = iris_ellipse_norm

            out_dict['pupil_ellipse_norm_regressed'] = elOut[:, :, 5:]
            out_dict['iris_ellipse_norm_regressed'] = elOut[:, :, :5]

            out_dict['pupil_center'] = out_dict['pupil_ellipse'][:, :, :2]
            out_dict['iris_center'] = out_dict['iris_ellipse'][:, :, :2]


            if self.make_aleatoric:
                out_dict['pupil_conf'] = elConf[:, :, 5:]
                out_dict['iris_conf'] = elConf[:, :, :5]
            else:
                out_dict['pupil_conf'] = torch.zeros_like(elConf)
                out_dict['iris_conf'] = torch.zeros_like(elConf)

            out_dict['ds_onehot'] = ds_predict
            out_dict['predict'] = op

        out_dict['latent'] = latent.detach().cpu()

        #rendOut_frames: values that are different fro each frames 
        #like radius, rotation and the distance 
        #rendOut_same: values that are same for all frames in the batch
        #like focal lenght and the eyeball center
        if args['net_rend_head']:
            rendOut_frames = self.renderingReg(enc_op[-1])
            rendOut_same = self.eye_param_same(enc_op[-1])
            rendOut_frames = rendOut_frames.reshape(B, NUM_OF_FRAME, -1) 
            rendOut_same = rendOut_same.reshape(B, -1) 

            rendOut_same = rendOut_same.unsqueeze(1)

            if torch.any(torch.isnan(rendOut_frames)):
                import sys
                print('WARNING!!! EYE MODEL FRAMEWORK PER FRAME PREDICT NANs')
                print('WARNING! Convergence failed!')
                sys.exit('Per frame predicted as NaNs')
                #out_dict_valid = False

            if torch.any(torch.isnan(rendOut_same)):
                import sys
                print('WARNING!!! EYE MODEL FRAMEWORK SAME PREDICT NANs')
                print('WARNING! Convergence failed!')
                sys.exit('same frame predicted as NaNs')
                #out_dict_valid = False

            out_dict['L'] = rendOut_same[..., 0]
            out_dict['r_iris'] = rendOut_same[..., 1]
            out_dict['T'] = rendOut_same[..., 2:5]
            out_dict['focal'] = rendOut_same[..., -2:]

            out_dict['R'] = rendOut_frames[..., :3]
            out_dict['r_pupil'] = rendOut_frames[..., 3]


        end = time.time()
        out_dict['dT'] = end - start
        
        return out_dict, out_dict_valid


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    from args_maker import make_args
    args = make_args()
    
    # args.growth_rate = 1.0
    # args.groups = 32
    
    model = DenseNet3D_head(vars(args))
    # model = model.cuda()
    
    model.train()
    n_params = get_nparams(model)
    model.eval()

    dT_list = []
    for fr in range(1000):
        
        with torch.no_grad():
            data_dict = {'image': torch.zeros(1, 240, 320)}
            out_dict = model(data_dict)
            dT_list.append(out_dict['dT'])
            
    dT_list = np.array(dT_list)
    print('FR: {}. # of params: {}'.format(np.mean(1/dT_list), n_params))