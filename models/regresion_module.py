import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_blocks import  conv_layer


class regressionModuleEllseg(torch.nn.Module):
    def __init__(self, sizes, norm, act_func,
                 sc=0.25,
                 pool=nn.AvgPool2d,
                 conf_pred=True,
                 ellipse_pred=True,
                 twice_ip_channels=False,
                 explicit_inChannels=False,
                 out_c=10, num_convs=3, dilate=False,
                 track_running_stats=False):
        super(regressionModuleEllseg, self).__init__()

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
                                  kernel_size=3, dilation=ii+1, padding=1,
                                  track_running_stats=track_running_stats)
                conv_ops.append(conv)
            else:
                conv = conv_layer(int(inChannels*(sc*ii+1)),
                                  int(inChannels*(sc*(ii+1)+1)),
                                  norm, act_func,
                                  kernel_size=3, dilation=1, padding=1,
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
