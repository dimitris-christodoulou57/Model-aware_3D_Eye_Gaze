from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.

    NOTE: Function taken from Kornia libary
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)

    # Ensure that xs and ys do not require gradients
    xs.requires_grad = False
    ys.requires_grad = False

    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def get_com(seg_map, temperature=4):
    # Custom function to find the center of mass to get detected pupil or iris
    # center. The center of mass is in normalized space between -1 and 1.
    # seg_map: BXHXW - single channel corresponding to pupil or iris prediction

    device = seg_map.device

    #reduce the number of dimension combine the batch size and the number of frames
    seg_map = seg_map.view(-1, seg_map.shape[2], seg_map.shape[3])

    B, H, W = seg_map.shape
    wtMap = F.softmax(seg_map.view(B, -1)*temperature, dim=1)  # [B, HXW]

    XYgrid = create_meshgrid(H, W, normalized_coordinates=True)  # 1xHxWx2
    xloc = XYgrid[0, :, :, 0].reshape(-1).to(device)
    yloc = XYgrid[0, :, :, 1].reshape(-1).to(device)

    xpos = torch.sum(wtMap*xloc, -1, keepdim=True)
    ypos = torch.sum(wtMap*yloc, -1, keepdim=True)
    predPts = torch.stack([xpos, ypos], dim=1).squeeze()

    #the returning tensor has the following dimension [BXF, HXW]
    return predPts


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

        self.conv = nn.Conv3d(in_channels=in_channels,
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


class regressionModule(torch.nn.Module):
    def __init__(self, sizes, norm, act_func,
                 sc=0.25,
                 num_of_frames = 8,
                 pool=nn.AvgPool3d,
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
        x = x.flatten(start_dim=-3).mean(dim=-1)
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



class renderingregressionModule(torch.nn.Module):
    def __init__(self, sizes, norm, act_func,
                 sc=0.25,
                 num_of_frames = 8,
                 pool=nn.AvgPool3d,
                 twice_ip_channels=False,
                 explicit_inChannels=False,
                 out_c_frame=4, out_c_same=7, num_convs=3, dilate=False,
                 track_running_stats=False):
        super(renderingregressionModule, self).__init__()

        #the output of this part is 8 values for each frame:
        #radius, 2 for rotation, distance form eyeball center to center of pupil/iris
        #4 values same for all frames: eyeball center and focal lenght
        out_c = out_c_frame * num_of_frames + out_c_same        

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

        self.l1 = nn.Linear(int(inChannels*(sc*num_convs+1)), 256, bias=False)
        self.l2 = nn.Linear(256, out_c_frame * num_of_frames, bias=False)

    def forward(self, x):
        # Linear layers to regress parameters
        # Note: cnf is the predicted variance and must always be positive
        x = self.conv_ops(x)
        x = x.flatten(start_dim=-3).mean(dim=-1)
        x = self.l1(x)
        torch.tanh_(x)

        out_per_frame = self.l2(x)

        torch.nn.functional.hardtanh_(out_per_frame)

        return out_per_frame


class eye_param_same_regressionModule(torch.nn.Module):
    def __init__(self, sizes, norm, act_func,
                 sc=0.25,
                 num_of_frames = 8,
                 pool=nn.AvgPool3d,
                 twice_ip_channels=False,
                 explicit_inChannels=False,
                 out_c_frame=4, out_c_same=7, num_convs=3, dilate=False,
                 track_running_stats=False):
        super(eye_param_same_regressionModule, self).__init__()

        #the output of this part is 8 values for each frame:
        #radius, 2 for rotation, distance form eyeball center to center of pupil/iris
        #4 values same for all frames: eyeball center and focal lenght
        out_c = out_c_frame * num_of_frames + out_c_same        

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

        self.l1 = nn.Linear(int(inChannels*(sc*num_convs+1)), 256, bias=False)
        self.l2 = nn.Linear(256, out_c_same, bias=False)

    def forward(self, x):
        # Linear layers to regress parameters
        # Note: cnf is the predicted variance and must always be positive
        x = self.conv_ops(x)
        x = x.flatten(start_dim=-3).mean(dim=-1)
        x = self.l1(x)
        torch.tanh_(x)

        out = self.l2(x)

        torch.nn.functional.hardtanh_(out)

        return out 

def detach_cpu_np(ip):
    return ip.detach().cpu().numpy()

def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
