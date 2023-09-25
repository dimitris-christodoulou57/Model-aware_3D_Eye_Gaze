#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 23:28:55 2021

@author: rakshit
"""

import os
import sys
import h5py
import torch
import numpy as np

import matplotlib.pyplot as plt

sys.path.append('/home/rakshit/Documents/Python_Scripts/multiset_gaze/src/helperfunctions')

from helperfunctions import my_ellipse

# def get_reduced(cone_param):
#     eta =

def get_cone_quad(ellipse_quad, focal=-1):
    a_ = ellipse_quad[0]
    h_ = ellipse_quad[1]
    b_ = ellipse_quad[2]
    g_ = ellipse_quad[3]
    f_ = ellipse_quad[4]
    d_ = ellipse_quad[5]

    a = a_*focal**2
    b = b_*focal**2
    c = d_
    d = d_*focal**2
    f = -focal*(f_)
    g = -focal*(g_)
    h = h_*focal**2
    u = g_*focal**2
    v = f_*focal**2
    w = -focal*d_

    M = torch.tensor([[a, h, g],
                      [h, b, f],
                      [g, f, c]])

    eig_vals = torch.symeig(M)[0]

    cone_params = [a,b,c,d,f,g,h,u,v,w]
    return M, cone_params, eig_vals

H5_file = 'riteyes_s-natural_15_4.h5'
path_data = '/media/rakshit/Monster/Datasets/All'

f = h5py.File(os.path.join(path_data, H5_file), 'r')

im_num = 143

mask = f['Masks_noSkin'][im_num, ...]
image = f['Images'][im_num, ...]
iris_ellipse = f['Fits']['iris'][im_num, ...]
f.close()

iris_quad = my_ellipse(iris_ellipse.tolist()).quad.tolist()
a, hh, b, gg, ff, d = iris_quad

h, g, f = hh/2, gg/2, ff/2

ellipse_quad = torch.tensor([a, h, b, g, f, d])

#%% Step 1 - Get conic equation
M, cone_params, eig_vals = get_cone_quad(ellipse_quad)

#%% Step 2 - Get reduce conic equation
# alphas = get_reduced(cone_param)





