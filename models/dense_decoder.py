import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import getSizes
from models.basic_blocks import conv_layer

from extern.squeeze_and_excitation.squeeze_and_excitation.squeeze_and_excitation import ChannelSpatialSELayer


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
                                        track_running_stats=track_running_stats)
            self.up_block_list.append(block)

        self.final = nn.Conv2d(opSize[-1], out_c, kernel_size=1, bias=True)

    def forward(self, skip_list, x):

        for block_num, block in enumerate(self.up_block_list):
            x = block(skip_list[-block_num-1], x)

        return self.final(x)


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
                 track_running_stats=False):

        super(DenseNet2D_up_block, self).__init__()

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
