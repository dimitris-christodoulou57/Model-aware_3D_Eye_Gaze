#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:40:51 2021

@author: rakshit
"""

import os
import cv2
import sys
import argparse
import matplotlib
import numpy as np
import pandas as pd
import deepdish as dd
import scipy.io as scio
import matplotlib.pyplot as plt

from skimage import draw

import warnings
warnings.filterwarnings("error")

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds',
                    help='Path to dataset',
                    type=str,
                    default='/media/rakshit/Monster/Datasets')
args = parser.parse_args()

sys.path.append(os.path.join(os.path.abspath('..')))
from helperfunctions.helperfunctions import generateEmptyStorage, mypause

if args.noDisp:
    noDisp = True
    print('No graphics')
else:
    noDisp = False
    print('Showing figures')

print('Extracting Swirski')

gui_env = ['Qt5Agg','WXAgg','TKAgg','GTKAgg']
for gui in gui_env:
    try:
        print("testing: {}".format(gui))
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue

print("Using: {}".format(matplotlib.get_backend()))

plt.ion()
PATH_DIR = os.path.join(args.path2ds, 'Swirski')
PATH_DS  = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')
list_ds = os.listdir(PATH_DIR)

Image_counter = 0.0
ds_num = 0

for fName in list_ds:
    warnings.filterwarnings("error")
    ds_name = 'Swirski'+'_'+fName+'_'+str(ds_num)

    # Parse subject ID from the name
    subject_id = str(fName.split('-')[0])

    # Ignore the first row and column.
    # Columns: [index, p_x, p_y]
    imList = os.listdir(os.path.join(PATH_DIR, fName, 'frames'))
    imList.sort()

    if not noDisp:
        fig, plts = plt.subplots(1,1)

    Data, keydict = generateEmptyStorage(name='Swirski', subset='Swirski_'+fName)

    # Read annotations
    info = pd.read_csv(os.path.join(PATH_DIR, fName, 'pupil-ellipses.txt'), delimiter=' ')
    ellipses = info.iloc[:, 2:].to_numpy()
    frame_no = info.iloc[:, 0].to_numpy()

    fr_num = 0

    for idx, entry in np.ndenumerate(frame_no):

        path_im = os.path.join(PATH_DIR, fName, 'frames', '{}-eye.png'.format(entry))
        I = cv2.imread(path_im, 0)

        # Pad 10 replicated pixels to bring to 640x480
        I = np.pad(I, ((10, 10), (10, 10)), mode='edge')

        iris_ellipse = -np.ones(5, )
        pupil_ellipse = ellipses[idx, ...].squeeze()
        pupil_ellipse[:2] += 10

        [rr_i, cc_i] = draw.ellipse(int(pupil_ellipse[1]),
                                    int(pupil_ellipse[0]),
                                    int(pupil_ellipse[3]),
                                    int(pupil_ellipse[2]),
                                    rotation=-pupil_ellipse[4],
                                    shape=I.shape)

        mask_noSkin = np.zeros_like(I)
        mask_noSkin[rr_i, cc_i] = 3  # Pupil label
        # mask = -np.ones_like(I) # We donot have PartSeg map

        Data['Info'].append(os.path.join(fName, 'frames', '{}-eye.png'.format(entry)))
        Data['Images'].append(I)
        Data['subject_id'].append(subject_id)
        Data['pupil_loc'].append(pupil_ellipse[:2])

        keydict['subject_id'].append(subject_id)
        keydict['resolution'].append(I.shape)
        keydict['pupil_loc'].append(pupil_ellipse[:2])
        keydict['archive'].append(ds_name)

        Data['Fits']['pupil'].append(pupil_ellipse)
        Data['Fits']['iris'].append(iris_ellipse)

        keydict['Fits']['pupil'].append(pupil_ellipse)
        keydict['Fits']['iris'].append(iris_ellipse)

        fr_num += 1
        if not noDisp:
            if fr_num == 1:
                cI = plts.imshow(I)
                cM = plts.imshow(mask_noSkin, alpha=0.5)
                plt.show()
                plt.pause(.01)
            else:
                cI.set_data(I)
                cM.set_data(mask_noSkin)
                mypause(0.01)

    keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
    keydict['archive'] = np.stack(keydict['archive'], axis=0)
    keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
    keydict['subject_id'] = np.stack(keydict['subject_id'], axis=0)
    keydict['Fits']['pupil'] = np.stack(keydict['Fits']['pupil'], axis=0)
    keydict['Fits']['iris'] = np.stack(keydict['Fits']['iris'], axis=0)

    Data['subject_id'] = np.stack(Data['subject_id'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    Data['Images'] = np.stack(Data['Images'], axis=0)
    Data['Masks'] = np.stack(Data['Masks'], axis=0)
    Data['Masks_noSkin'] = np.stack(Data['Masks_noSkin'], axis=0)
    Data['Fits']['pupil'] = np.stack(Data['Fits']['pupil'], axis=0)
    Data['Fits']['iris'] = np.stack(Data['Fits']['iris'], axis=0)

    # Save data
    warnings.filterwarnings("ignore")
    dd.io.save(os.path.join(PATH_DS, ds_name+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, ds_name+'.mat'), keydict, appendmat=True)
    ds_num=ds_num+1