#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

import os
import cv2
import sys
import argparse

import numpy as np
import pandas as pd
import multiprocessing as mp

import deepdish as dd
import scipy.io as scio

from skimage import draw

sys.path.append('..')
from helperfunctions.helperfunctions import plot_segmap_ellpreds
from helperfunctions.helperfunctions import generateEmptyStorage

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2ds', type=str, default='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets',
                        help='path to datasets')
    parser.add_argument('--path_data', type=str, default='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets/Ours',
                        help='path to TEyeD Dikablis eye videos')
    args = parser.parse_args()
    return args


def process_entry(args, ):

    vid_name_ext = args['vid_name']
    PATH_DS = os.path.join(args['path2ds'], 'All')
    PATH_MASTER = os.path.join(args['path2ds'], 'MasterKey')
    ds_name = '{}'.format(vid_name_ext)
    
    #generate empty storage for the data to the folders
    #TODO use subset= variable
    Data, keydict = generateEmptyStorage(name='Ours',subset='{}'.format(vid_name_ext))

    # Read video frame by frame
    vid_obj = cv2.VideoCapture(args['path_video'])
    width = vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    ret = True  # Start the loop
    fr_idx = 0

    while ret:
        ret, frame = vid_obj.read()
        if not ret or (fr_idx+1)%500==0:
            break

        #if not ret:
        #   break

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LANCZOS4)
        imName_Full = 'Ours-{}-frame-{}'.format(vid_name_ext,fr_idx) 

        out_dict = {}

        #increase the count before go to the next frame in case of continue
        fr_idx += 1

        
        # Save frame and masks
        LabelMat = np.zeros((int(height), int(width)), dtype=np.int32)
        LabelMat[0,0] = 1
        LabelMat[0,1] = 2
        LabelMat[0,2] = 3

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LANCZOS4)
        LabelMat = cv2.resize(LabelMat, (640, 480), interpolation=cv2.INTER_NEAREST)

        #Add information to keydict (model information)
        keydict['archive'].append(ds_name)
        keydict['resolution'].append(frame.shape)
        keydict['pupil_loc'].append(np.ones(2)*150 + (0.1)*fr_idx)
        keydict['subject_id'].append('0')

        #Append images and label
        Data['Info'].append(imName_Full)
        #the predicted mask is without skin 
        #save the same mask in both cases
        Data['Masks'].append(LabelMat)
        Data['Images'].append(frame)
        Data['pupil_loc'].append(np.ones(2)*150 + (0.1)*fr_idx)
        Data['subject_id'].append('0')
        Data['Eyeball'].append(np.ones(4))
        Data['Gaze_vector'].append(np.ones(3))
        Data['pupil_lm_2D'].append(np.ones(17))
        Data['pupil_lm_3D'].append(np.ones(25))
        Data['iris_lm_2D'].append(np.ones(17))
        Data['iris_lm_3D'].append(np.ones(25))

        #Append fits
        Data['Fits']['pupil'].append(np.ones(5))
        Data['Fits']['iris'].append(np.ones(5))

        keydict['Fits']['pupil'].append(np.ones(5))
        keydict['Fits']['iris'].append(np.ones(5))

    vid_obj.release()

    # Stack data
    Data['Masks'] = np.stack(Data['Masks'], axis=0)
    Data['Images'] = np.stack(Data['Images'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    Data['subject_id'] = np.stack(Data['subject_id'], axis=0)
    Data['Masks_noSkin'] = Data['Masks']
    Data['Fits']['pupil'] = np.stack(Data['Fits']['pupil'], axis=0)
    Data['Fits']['iris'] = np.stack(Data['Fits']['iris'], axis=0)

    #New data from the TEyeD dataset
    Data['Eyeball'] = np.stack(Data['Eyeball'], axis=0)
    Data['Gaze_vector'] = np.stack(Data['Gaze_vector'], axis=0)
    Data['pupil_lm_2D'] = np.stack(Data['pupil_lm_2D'], axis=0)
    Data['pupil_lm_3D'] = np.stack(Data['pupil_lm_3D'], axis=0)
    Data['iris_lm_2D'] = np.stack(Data['iris_lm_2D'], axis=0)
    Data['iris_lm_3D'] = np.stack(Data['iris_lm_3D'], axis=0)

    # Save keydict
    keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
    keydict['subject_id'] = np.stack(keydict['subject_id'], axis=0)
    keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
    keydict['archive'] = np.stack(keydict['archive'], axis=0)
    keydict['Fits']['pupil'] = np.stack(keydict['Fits']['pupil'], axis=0)
    keydict['Fits']['iris'] = np.stack(keydict['Fits']['iris'], axis=0)

    dd.io.save(os.path.join(PATH_DS, ds_name+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)

if __name__ == '__main__':

    args = vars(make_args())
    path_videos = args['path_data']
    list_videos = os.listdir(path_videos)

    # pool = mp.Pool(mp.cpu_count())

    num_of_videos = 0

    for vid_name_ext in list_videos:

        args['vid_name'] = os.path.splitext(vid_name_ext)[0]
        args['path_video'] = os.path.join(path_videos, vid_name_ext)
        print(vid_name_ext)
        print('num of videos {}'.format(num_of_videos))
                           

        # pool.apply_async()
        process_entry(args, )

        num_of_videos += 1
    
    print('num of videos {}'.format(num_of_videos))





