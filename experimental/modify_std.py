#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:41:07 2021

@author: rakshit
"""

import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

def norm(image, std):
    out = copy.deepcopy(image)
    out = out - out.mean()
    out = out/std
    return out

def scale_viz(image):
    # image = image - image.min()
    # image = 255*image/image.max()
    return image

image_0 = cv2.imread('/media/rakshit/Monster/Datasets/Swirski/p2-left/frames/195-eye.png', 0)
image_1 = cv2.imread('/media/rakshit/Monster/Datasets/Fuhl/data set VIII/0000007755.png', 0)[:-10, :-10]

image_0 = cv2.resize(image_0, dsize=(640, 480))
image_1 = cv2.resize(image_1, dsize=(640, 480))

std_0 = image_0.std()
std_1 = image_1.std()

image_0 = norm(image_0, std_1)
image_1 = norm(image_1, std_0)

out = scale_viz(np.concatenate([image_0, image_1], axis=1))


fig, axs = plt.subplots()
axs.imshow(out, cmap='gray')
