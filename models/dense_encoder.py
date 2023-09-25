import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import getSizes
from models.basic_blocks import convBlock, conv_layer

from extern.squeeze_and_excitation.squeeze_and_excitation.squeeze_and_excitation import ChannelSpatialSELayer


class DenseNet_encoder(nn.Module):
    def __init__(self, args, in_c=1,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):

        super(DenseNet_encoder, self).__init__()

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
                              norm=norm,
                              track_running_stats=track_running_stats)

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
                                          track_running_stats=track_running_stats)
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

        skip_list = []
        for block in self.down_block_list:
            skip, x = block(x)
            skip_list.append(skip)

        if self.recalib:
            x = self.recalib(x)

        return tuple(skip_list) + (x, )


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
                 track_running_stats=False):

        super(DenseNet2D_down_block, self).__init__()

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