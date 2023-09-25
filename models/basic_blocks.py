import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class convBlock(nn.Module):
    def __init__(self,
                 in_c,
                 inter_c,
                 out_c,
                 act_func,
                 norm,
                 groups=1,
                 num_layers=3,
                 track_running_stats=False):
        super(convBlock, self).__init__()

        list_of_conv_layers = []
        for i in range(num_layers):
            conv = conv_layer(in_c if i == 0 else inter_c,
                              inter_c if i < num_layers-1 else out_c,
                              kernel_size=3,
                              norm_layer=norm,
                              act_func=act_func,
                              groups=groups,
                              padding=1,
                              track_running_stats=track_running_stats,
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
                 bias=False):

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


def do_nothing(input):
    return input

