#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

from models.denseEl0.DenseEINet_3D_head import DenseNet3D_head
from models.denseEl0.DenseEINet_3D_gaze import DenseNet3D_gaze
from models.denseEl1.DenseElNet import DenseNet3D
from models.denseEl2.DenseElNet2 import DenseNet2
from models.denseEl2.DenseElNet2_gaze import DenseNet2_gaze
from models.denseEl3.DenseElNet3 import DenseNet3
from models.denseEl3.DenseElNet3_gaze import DenseNet3_gaze
from models.res_50_0.res_50_0 import res_50_0
from models.res_50_1.res_50_1 import res_50_1
from models.res_50_2.res_50_2 import res_50_2
from models.res_50_3.res_50_3 import res_50_3
# from models.DenseElNet_old import DenseNet2D

model_dict={'DenseEl0':DenseNet3D_head,
            'DenseEl0_gaze': DenseNet3D_gaze,
            'DenseEl1':DenseNet3D,
            'DenseEl2':DenseNet2,
            'DenseEl2_gaze':DenseNet2_gaze,
            'DenseEl3':DenseNet3,
            'DenseEl3_gaze':DenseNet3_gaze,
            'res_50_0': res_50_0,
            'res_50_1': res_50_1,
            'res_50_2': res_50_2,
            'res_50_3': res_50_3,
            }