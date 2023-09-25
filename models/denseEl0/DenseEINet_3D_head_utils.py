#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from random import random
from typing import Optional
from sklearn import metrics

from helperfunctions.utils import do_nothing
from helperfunctions.helperfunctions import assert_torch_invalid

class linStack(torch.nn.Module):
    """A stack of linear layers followed by batch norm and hardTanh

    Attributes:
        num_layers: the number of linear layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """

    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 bias,
                 actBool,
                 dp):
        super().__init__()

        layers_lin = []
        for i in range(num_layers):
            m = torch.nn.Linear(hidden_dim if i > 0 else in_dim,
                                hidden_dim if i < num_layers - 1 else out_dim,
                                bias=bias)
            layers_lin.append(m)
        self.layersLin = torch.nn.ModuleList(layers_lin)
        self.act_func = torch.nn.ReLU()
        self.actBool = actBool
        self.dp = torch.nn.Dropout(p=dp)

    def forward(self, x):
        # Input shape (batch, features, *)
        for i, _ in enumerate(self.layersLin):
            x = self.act_func(x) if self.actBool else x
            x = self.layersLin[i](x)
            x = self.dp(x)
        return x

class driver_ada_instance(torch.nn.Module):
    def __init__(self, mixup=False):
        super(driver_ada_instance, self).__init__()
        self.mixup = mixup
        self.alpha = 0.1  # parameter for mixup, ignore otherwise
        pass

    def calc_mean_std(self, feat, eps=1e-5):
        # Code taken from https://github.com/naoto0804/pytorch-AdaIN/
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def do_mixup(self, content_mean, content_std, style_mean, style_std):
        mixup_mean = self.alpha*content_mean + (1 - self.alpha)*style_mean
        mixup_std = self.alpha*content_std + (1 - self.alpha)*style_std
        return mixup_mean, mixup_std

    def forward(self, content_feat, style_feat):
        # Code taken from https://github.com/naoto0804/pytorch-AdaIN/
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)

        if self.mixup:
            mixup_mean, mixup_std = self.do_mixup(content_mean,
                                                  content_std,
                                                  style_mean,
                                                  style_std)
            output = normalized_feat * mixup_std.expand(size) + \
                mixup_mean.expand(size)
        else:
            output = normalized_feat * style_std.expand(size) + \
                style_mean.expand(size)
        return output


class regressionModule(torch.nn.Module):
    def __init__(self, sizes, norm, act_func,
                 sc=0.25,
                 num_of_frames = 8,
                 pool=nn.AvgPool2d,
                 conf_pred=True,
                 ellipse_pred=True,
                 twice_ip_channels=False,
                 explicit_inChannels=False,
                 out_c=10, num_convs=3, dilate=False,
                 track_running_stats=False):
        super(regressionModule, self).__init__()

        out_c = out_c * num_of_frames

        self.ellipse_pred = ellipse_pred

        if explicit_inChannels:
            inChannels = explicit_inChannels
        else:
            inChannels = sizes['enc']['op'][-1] \
                if not twice_ip_channels else \
                2*sizes['enc']['op'][-1]
        

        conv_ops = []
        for ii in range(num_convs):
            if dilate:
                conv = conv_layer(int(inChannels*(sc*ii+1)),
                                  int(inChannels*(sc*(ii+1)+1)),
                                  norm, act_func,
                                  kernel_size=3, 
                                  dilation=ii+1, padding=1,
                                  track_running_stats=track_running_stats)
                conv_ops.append(conv)
            else:
                conv = conv_layer(int(inChannels*(sc*ii+1)),
                                  int(inChannels*(sc*(ii+1)+1)),
                                  norm, act_func,
                                  kernel_size=3,
                                  dilation=1, padding=1,
                                  track_running_stats=track_running_stats)
                conv_ops.append(conv)

            # Pool to eject pixels whose channels do not activate
            # maximally to pupil or iris ellipse features
            if pool:
                conv_ops.append(pool(kernel_size=2))

        self.conv_ops = nn.Sequential(*conv_ops)

        self.l1 = nn.Linear(int(inChannels*(sc*num_convs+1)), 256, bias=True)
        self.l2_out = nn.Linear(256, out_c, bias=True)
        self.l2_conf = nn.Linear(256, out_c, bias=True) if conf_pred else False

    def forward(self, x):
        # Linear layers to regress parameters
        # Note: cnf is the predicted variance and must always be positive
        x = self.conv_ops(x)
        x = x.flatten(start_dim=-2).mean(dim=-1)
        x = self.l1(x)

        torch.selu_(x)

        if self.l2_conf:
            cnf = self.l2_conf(x)
        else:
            cnf = []
        out = self.l2_out(x)

        if self.ellipse_pred:
            # Ellipse centers should be between -1 and 1
            torch.nn.functional.hardtanh_(out[:, [0, 1, 5, 6]])

            # Ellipse axes should be between 0 and 1
            torch.relu_(out[:, [2, 3, 7, 8]])

            # Ellipse orientation should be between 0 and pi
            out[:, [4, 9]] = np.pi*torch.clamp(out[:, [4, 9]], 0., 1.)
        else:
            pass
        return out, cnf

class renderingregressionModule(torch.nn.Module):
    def __init__(self, sizes, norm, act_func, act_final=F.tanh,
                 sc=0.25,
                 num_of_frames = 8,
                 pool=nn.AvgPool2d,
                 twice_ip_channels=False,
                 explicit_inChannels=False,
                 out_c_frame=4, out_c_same=7, num_convs=3, dilate=False,
                 track_running_stats=False):
        super(renderingregressionModule, self).__init__() 

        self.act_final = act_final     

        if explicit_inChannels:
            inChannels = explicit_inChannels
        else:
            inChannels = sizes['enc']['op'][-1] \
                if not twice_ip_channels else \
                2*sizes['enc']['op'][-1]
        

        conv_ops = []
        for ii in range(num_convs):
            if dilate:
                conv = conv_layer(int(inChannels*(sc*ii+1)),
                                  int(inChannels*(sc*(ii+1)+1)),
                                  norm, act_func,
                                  kernel_size=3, 
                                  dilation=ii+1, padding=1,
                                  track_running_stats=track_running_stats)
                conv_ops.append(conv)
            else:
                conv = conv_layer(int(inChannels*(sc*ii+1)),
                                  int(inChannels*(sc*(ii+1)+1)),
                                  norm, act_func,
                                  kernel_size=3, 
                                  dilation=1, padding=1,
                                  track_running_stats=track_running_stats)
                conv_ops.append(conv)

            # Pool to eject pixels whose channels do not activate
            # maximally to pupil or iris ellipse features
            if pool:
                conv_ops.append(pool(kernel_size=2))

        self.conv_ops = nn.Sequential(*conv_ops)

        self.l1 = nn.Linear(int(inChannels*(sc*num_convs+1)), 256, bias=True)
        self.l2 = nn.Linear(256, out_c_frame * num_of_frames, bias=True)

        self.l3 = nn.Linear(int(inChannels*(sc*num_convs+1)), 
                            out_c_frame * num_of_frames, bias=True)

    def forward(self, x):
        # Linear layers to regress parameters
        # Note: cnf is the predicted variance and must always be positive
        x = self.conv_ops(x)
        x = x.flatten(start_dim=-2).mean(dim=-1)
        x = self.l1(x)
        torch.tanh_(x)
        assert torch.all(torch.all(x >= -1) and \
                                 torch.all(x <= 1)), \
                'predicted frame x out of range 2'
        out_per_frame = self.l2(x)
        #torch.nn.functional.hardtanh_(out_per_frame)
        #out_per_frame = self.l3(x)

        #torch.tanh_(out_per_frame)
        out_per_frame = self.act_final(out_per_frame)
        # assert torch.all(torch.all(out_per_frame >= -1)& \
        #                          torch.all(out_per_frame <= 1)), \
        #         'predictedframe out x out of range 4'

        return out_per_frame

class eye_param_same_regressionModule(torch.nn.Module):
    def __init__(self, sizes, norm, act_func, act_final=F.tanh,
                 sc=0.25,
                 num_of_frames = 8,
                 pool=nn.AvgPool2d,
                 twice_ip_channels=False,
                 explicit_inChannels=False,
                 out_c_frame=4, out_c_same=7, num_convs=3, dilate=False,
                 track_running_stats=False):
        super(eye_param_same_regressionModule, self).__init__()   

        self.act_final = act_final   

        if explicit_inChannels:
            inChannels = explicit_inChannels
        else:
            inChannels = sizes['enc']['op'][-1] \
                if not twice_ip_channels else \
                2*sizes['enc']['op'][-1]
        

        conv_ops = []
        for ii in range(num_convs):
            if dilate:
                conv = conv_layer(int(inChannels*(sc*ii+1)),
                                  int(inChannels*(sc*(ii+1)+1)),
                                  norm, act_func,
                                  kernel_size=3, 
                                  dilation=ii+1, padding=1,
                                  track_running_stats=track_running_stats)
                conv_ops.append(conv)
            else:
                conv = conv_layer(int(inChannels*(sc*ii+1)),
                                  int(inChannels*(sc*(ii+1)+1)),
                                  norm, act_func,
                                  kernel_size=3, 
                                  dilation=1, padding=1,
                                  track_running_stats=track_running_stats)
                conv_ops.append(conv)

            # Pool to eject pixels whose channels do not activate
            # maximally to pupil or iris ellipse features
            if pool:
                conv_ops.append(pool(kernel_size=2))

        self.conv_ops = nn.Sequential(*conv_ops)

        self.l1 = nn.Linear(int(inChannels*(sc*num_convs+1)), 256, bias=True)
        self.l2 = nn.Linear(256, out_c_same, bias=True)

        self.l3 = nn.Linear(int(inChannels*(sc*num_convs+1)), 
                            out_c_same, bias=True)

    def forward(self, x):
        # Linear layers to regress parameters
        # Note: cnf is the predicted variance and must always be positive
        x = self.conv_ops(x)
        x = x.flatten(start_dim=-2).mean(dim=-1)
        x = self.l1(x)
        torch.tanh_(x)
        assert torch.all(torch.all(x >= -1)& \
                                 torch.all(x <= 1)), \
                'predicted same x out of range 1'
        out = self.l2(x)

        #torch.nn.functional.hardtanh_(out)
        #out = self.l3(x)
        #torch.tanh_(out) 
        out = self.act_final(out)
        # assert torch.all(torch.all(out >= -1)& \
        #                          torch.all(out <= 1)), \
        #         'predicted same x out of range 2'

        return out 


class convBlock(nn.Module):
    def __init__(self,
                 in_c,
                 inter_c,
                 out_c,
                 act_func,
                 norm,
                 groups=1,
                 num_layers=3,
                 track_running_stats=False,
                 args=None,
                 conv_3D=False):
        super(convBlock, self).__init__()

        self.num_of_frames = args['frames']

        list_of_conv_layers = []
        for i in range(num_layers):
            if conv_3D:
                conv = conv_layer(in_c if i == 0 else inter_c,
                                inter_c if i < num_layers-1 else out_c,
                                kernel_size=(self.num_of_frames,3,3),
                                norm_layer=norm,
                                act_func=act_func,
                                groups=groups,
                                padding=0,
                                track_running_stats=track_running_stats,
                                conv_3D=conv_3D,
                                )
            else:
                conv = conv_layer(in_c if i == 0 else inter_c,
                                inter_c if i < num_layers-1 else out_c,
                                kernel_size=3,
                                norm_layer=norm,
                                act_func=act_func,
                                groups=groups,
                                padding=1,
                                track_running_stats=track_running_stats,
                                conv_3D=conv_3D,
                                )
            list_of_conv_layers.append(conv)

        self.list_of_conv_layers = nn.Sequential(*list_of_conv_layers)

    def forward(self, x):
        return self.list_of_conv_layers(x)


class conv_layer(nn.Module):
    '''
    Standard convolutional layer followed by a normalization layer and
    an activation function. Justification for order:
    Using BN after convolutional layer allows us to ignore the bias
    parameters in the conv layer.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer,
                 act_func=F.leaky_relu,
                 track_running_stats=False,
                 kernel_size=3,
                 dilation=1,
                 dropout=False,
                 padding=1,
                 groups=1,  # If entry is a string, then groups==in_channels
                 bias=False,
                 conv_3D=False):

        super(conv_layer, self).__init__()

        self.random_style = False

        if type(norm_layer) == str:
            if norm_layer == 'group_norm':
                # Original paper suggests 8 or 16 channels per group, lets roll
                # with 8 because I like that number.
                self.norm_layer = nn.GroupNorm(num_groups=out_channels//2,
                                               num_channels=out_channels,
                                               affine=True)
            elif norm_layer == 'ada_instance_norm':
                # Use a style for instance norm
                self.norm_layer = driver_ada_instance()
                self.random_style = True
            elif norm_layer == 'ada_instance_norm_mixup':
                self.norm_layer = driver_ada_instance(mixup=True)
                self.random_style = True
            elif norm_layer == 'none':
                self.norm_layer = do_nothing
            else:
                import sys
                sys.exit('Incorrect norm entry')
        else:
            self.norm_layer = norm_layer(out_channels,
                                         affine=True,
                                         track_running_stats=track_running_stats)
        self.act_func = act_func

        if conv_3D:
            self.conv = nn.Conv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=(0,1,1),
                                groups=out_channels if groups == 0 else groups,
                                bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                groups=out_channels if groups == 0 else groups,
                                bias=bias)

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

    def forward(self, x):
        x = self.conv(x)
        if self.random_style:
            loc = torch.randperm(x.shape[0])
            if self.training:
                x = self.norm_layer(x, x[loc, ...])
            else:
                x = self.norm_layer(x, x)
        else:
            x = self.norm_layer(x)
        x = self.act_func(x)
        return self.dropout(x) if self.dropout else x
