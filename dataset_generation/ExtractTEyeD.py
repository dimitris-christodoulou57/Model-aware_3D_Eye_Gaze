#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 03:41:08 2021

@author: rsk3900
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

from Visualitation_TEyeD.gaze_estimation import draw_gaze, draw_landmark

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2ds', type=str, default='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets',
                        help='path to datasets')
    parser.add_argument('--path_data', type=str, default='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets/TEyeD/TEyeDSSingleFiles/Dikablis',
                        help='path to TEyeD Dikablis eye videos')
    args = parser.parse_args()
    return args


def process_entry(args, ):

    vid_name_ext = args['vid_name']
    PATH_DS = os.path.join(args['path2ds'], 'All')
    PATH_MASTER = os.path.join(args['path2ds'], 'MasterKey')
    ds_name = '{}'.format(vid_name_ext)
    

    # Read text file
    iris_ellipses = pd.read_csv(args['path_annot']+'iris_eli.txt',
                                error_bad_lines=False,
                                delimiter=';').to_numpy()
    
    iris_validity = pd.read_csv(args['path_annot']+'validity_iris.txt',
                                error_bad_lines=False,
                                delimiter=';').to_numpy()
    
    iris_landmark_2D = pd.read_csv(args['path_annot']+'iris_lm_2D.txt',
                                    error_bad_lines=False,
                                    delimiter=';').to_numpy()
    
    iris_landmark_3D = pd.read_csv(args['path_annot']+'iris_lm_3D.txt',
                                    error_bad_lines=False,
                                    delimiter=';').to_numpy()
    
    pupil_in_iris_ellipses = pd.read_csv(args['path_annot']+'pupil_in_iris_eli.txt',
                                 error_bad_lines=False,
                                 delimiter=';').to_numpy()
    
    pupil_ellipses = pd.read_csv(args['path_annot']+'pupil_eli.txt',
                                 error_bad_lines=False,
                                 delimiter=';').to_numpy()

    pupil_validity = pd.read_csv(args['path_annot']+'validity_pupil.txt',
                                 error_bad_lines=False,
                                 delimiter=';').to_numpy()
    
    pupil_landmark_2D = pd.read_csv(args['path_annot']+'pupil_lm_2D.txt',
                                    error_bad_lines=False,
                                    delimiter=';').to_numpy()
    
    pupil_landmark_3D = pd.read_csv(args['path_annot']+'pupil_lm_3D.txt',
                                    error_bad_lines=False,
                                    delimiter=';').to_numpy()

    eye_ball = pd.read_csv(args['path_annot']+'eye_ball.txt',
                           error_bad_lines=False,
                           delimiter=';').to_numpy()
    
    gaze_vector = pd.read_csv(args['path_annot']+'gaze_vec.txt',
                              error_bad_lines=False,
                              delimiter=';').to_numpy()
    
    #clean data from NaN and frame number
    iris_ellipses = iris_ellipses[...,1:-1]
    iris_validity = iris_validity[...,1:-1]
    iris_landmark_2D = iris_landmark_2D[...,1:-1]
    iris_landmark_3D = iris_landmark_3D[...,1:-1]
    pupil_ellipses = pupil_ellipses[...,1:-1]
    pupil_in_iris_ellipses = pupil_in_iris_ellipses[...,1:-1]
    pupil_validity = pupil_validity[...,1:-1]
    pupil_landmark_2D = pupil_landmark_2D[...,1:-1]
    pupil_landmark_3D = pupil_landmark_3D[...,1:-1]
    eye_ball = eye_ball[...,1:-1]
    gaze_vector = gaze_vector[...,1:-1]

    #generate empty storage for the data to the folders
    #TODO use subset= variable
    Data, keydict = generateEmptyStorage(name='TEyeD',subset='{}'.format(vid_name_ext))

    # Read video frame by frame
    vid_obj = cv2.VideoCapture(args['path_video'])
    width = vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    if (width!=384):
        print('width')

    if (height!=288):
        print('height')

    ret = True  # Start the loop
    fr_idx = 0

    while ret:

        # Skip frames
        for _ in range(3):
            vid_obj.grab()
            fr_idx += 1

        ret, frame = vid_obj.read()
        if not ret or (fr_idx+1)%50000==0:
            break

        #if not ret:
        #   break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #resize all the ground truth
        scale_factor = 320/frame.shape[1]
        eye_ball[fr_idx] = eye_ball[fr_idx] * scale_factor
        iris_landmark_2D[fr_idx] = iris_landmark_2D[fr_idx] * scale_factor
        pupil_landmark_2D[fr_idx] = pupil_landmark_2D[fr_idx] * scale_factor

        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LANCZOS4)
        imName_Full = 'TEyeD-{}-frame-{}'.format(vid_name_ext,fr_idx) 

        out_dict = {}

        #define the pupil and iris model using the gt_label
        model_pupil = pupil_ellipses[fr_idx]
        model_pupil_in_iris = pupil_in_iris_ellipses[fr_idx]
        model_iris = iris_ellipses[fr_idx]

        #Convert model to EllSeg style
        model_pupil = np.roll(model_pupil, shift=-1)
        model_pupil[2:4] = model_pupil[2:4]/2
        model_pupil[-1] = np.deg2rad(model_pupil[-1]-90)
        model_pupil[[2, 3]] = model_pupil[[3, 2]]

        pupil_loc = model_pupil[:2]

        #Convert model to EllSeg style
        model_pupil_in_iris = np.roll(model_pupil_in_iris, shift=-1)
        model_pupil_in_iris[2:4] = model_pupil_in_iris[2:4]/2
        model_pupil_in_iris[-1] = np.deg2rad(model_pupil_in_iris[-1]-90)
        model_pupil_in_iris[[2, 3]] = model_pupil_in_iris[[3, 2]]

        pupil_in_iris_loc = model_pupil_in_iris[:2]

        model_iris = np.roll(model_iris, shift=-1)
        model_iris[2:4] = model_iris[2:4]/2
        model_iris[-1] = np.deg2rad(model_iris[-1]-90)
        model_iris[[2, 3]] = model_iris[[3, 2]]

        #compute timestamp
        timestamp = (1/25) * fr_idx

        #increase the count before go to the next frame in case of continue
        fr_idx += 1

        if (np.any(model_iris == -1) or np.any(iris_validity[fr_idx-1]==-1)) or\
           (np.any(model_pupil == -1) or np.any(pupil_validity[fr_idx-1]==-1)) or \
            np.any(model_pupil_in_iris == -1) or np.any(gaze_vector[fr_idx-1] == -1):
            continue

        [rr_i, cc_i] = draw.ellipse(round(model_iris[1]),
                                    round(model_iris[0]),
                                    round(model_iris[3]),
                                    round(model_iris[2]),
                                    shape=(int(height), int(width)),
                                    rotation=-model_iris[4])

        [rr_p, cc_p] = draw.ellipse(round(model_pupil[1]),
                                    round(model_pupil[0]),
                                    round(model_pupil[3]),
                                    round(model_pupil[2]),
                                    shape=(int(height), int(width)),
                                    rotation=-model_pupil[4])
        
        [rr_pi, cc_pi] = draw.ellipse(round(model_pupil_in_iris[1]),
                                    round(model_pupil_in_iris[0]),
                                    round(model_pupil_in_iris[3]),
                                    round(model_pupil_in_iris[2]),
                                    shape=(int(height), int(width)),
                                    rotation=-model_pupil_in_iris[4])
        
        # Save frame and masks
        LabelMat = np.zeros((int(height), int(width)), dtype=np.int32)
        LabelMat[rr_i, cc_i] = 2
        LabelMat[rr_p, cc_p] = 3

        LabelMat = cv2.resize(LabelMat, (320, 240), interpolation=cv2.INTER_NEAREST)

        # Save frame and masks pupil in iris
        LabelMat_pi = np.zeros((int(height), int(width)), dtype=np.int32)
        LabelMat_pi[rr_i, cc_i] = 2
        LabelMat_pi[rr_pi, cc_pi] = 3

        LabelMat_pi = cv2.resize(LabelMat, (320, 240), interpolation=cv2.INTER_NEAREST)

        #Add information to keydict (model information)
        keydict['archive'].append(ds_name)
        keydict['resolution'].append(frame.shape)
        keydict['pupil_loc'].append(pupil_loc)
        keydict['pupil_in_iris_loc'].append(pupil_in_iris_loc)
        keydict['subject_id'].append('0')

        #Append images and label
        Data['Info'].append(imName_Full)
        #the predicted mask is without skin 
        #save the same mask in both cases
        Data['Masks'].append(LabelMat)
        Data['Masks_pupil_in_iris'].append(LabelMat_pi)
        Data['Images'].append(frame)
        Data['pupil_loc'].append(pupil_loc)
        Data['pupil_in_iris_loc'].append(pupil_in_iris_loc)
        Data['subject_id'].append('0')
        Data['Eyeball'].append(eye_ball[fr_idx-1])
        Data['Gaze_vector'].append(gaze_vector[fr_idx-1])
        Data['pupil_lm_2D'].append(pupil_landmark_2D[fr_idx-1])
        Data['pupil_lm_3D'].append(pupil_landmark_3D[fr_idx-1])
        Data['iris_lm_2D'].append(iris_landmark_2D[fr_idx-1])
        Data['iris_lm_3D'].append(iris_landmark_3D[fr_idx-1])
        Data['timestamp'].append(timestamp)

        #Append fits
        Data['Fits']['pupil'].append(model_pupil)
        Data['Fits']['iris'].append(model_iris)

        keydict['Fits']['pupil'].append(model_pupil)
        keydict['Fits']['iris'].append(model_iris)

    vid_obj.release()

    # Stack data
    Data['Masks'] = np.stack(Data['Masks'], axis=0)
    Data['Masks_pupil_in_iris'] = np.stack(Data['Masks_pupil_in_iris'], axis=0)
    Data['Images'] = np.stack(Data['Images'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    Data['pupil_in_iris_loc'] = np.stack(Data['pupil_in_iris_loc'], axis=0)
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
    Data['timestamp'] = np.stack(Data['timestamp'], axis=0)

    # Save keydict
    keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
    keydict['subject_id'] = np.stack(keydict['subject_id'], axis=0)
    keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
    keydict['pupil_in_iris_loc'] = np.stack(keydict['pupil_in_iris_loc'], axis=0)
    keydict['archive'] = np.stack(keydict['archive'], axis=0)
    keydict['Fits']['pupil'] = np.stack(keydict['Fits']['pupil'], axis=0)
    keydict['Fits']['iris'] = np.stack(keydict['Fits']['iris'], axis=0)
    print(fr_idx)

    dd.io.save(os.path.join(PATH_DS, ds_name+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)

if __name__ == '__main__':

    args = vars(make_args())
    path_videos = os.path.join(args['path_data'], 'VIDEOS')
    path_annots = os.path.join(args['path_data'], 'ANNOTATIONS')
    list_videos = os.listdir(path_videos)

    ### some files from the dataset are coppurted thus skip these

    list_videos = [x for x in list_videos if "DikablisSS_10_1" not in x]
    list_videos = [x for x in list_videos if "DikablisT_6_12" not in x]
    list_videos = [x for x in list_videos if "DikablisT_20_12" not in x]
    list_videos = [x for x in list_videos if "DikablisT_24_8" not in x]
    list_videos = [x for x in list_videos if "DikablisT_23_5" not in x]
    list_videos = [x for x in list_videos if "DikablisT_1_9" not in x]
    list_videos = [x for x in list_videos if "DikablisT_23_6" not in x]
    list_videos = [x for x in list_videos if "DikablisT_23_3" not in x]
    list_videos = [x for x in list_videos if "DikablisT_23_8" not in x]
    list_videos = [x for x in list_videos if "DikablisT_20_6" not in x]
    list_videos = [x for x in list_videos if "DikablisT_22_7" not in x]
    list_videos = [x for x in list_videos if "DikablisT_24_5" not in x]
    list_videos = [x for x in list_videos if "DikablisT_23_7" not in x]
    list_videos = [x for x in list_videos if "DikablisT_22_9" not in x]
    list_videos = [x for x in list_videos if "DikablisT_20_3" not in x]
    list_videos = [x for x in list_videos if "DikablisT_24_11" not in x]
    list_videos = [x for x in list_videos if "DikablisT_15_3" not in x]
    list_videos = [x for x in list_videos if "DikablisT_22_8" not in x]
    list_videos = [x for x in list_videos if "DikablisT_11_3" not in x]
    list_videos = [x for x in list_videos if "DikablisT_24_7" not in x]
    list_videos = [x for x in list_videos if "DikablisT_21_7" not in x]
    list_videos = [x for x in list_videos if "DikablisT_20_5" not in x]

    list_videos = [x for x in list_videos if "DikablisT" in x]

    # pool = mp.Pool(mp.cpu_count())

    num_of_videos = 0

    for vid_name_ext in list_videos:

        if 'Dikablis' in vid_name_ext:
            args['vid_name'] = os.path.splitext(vid_name_ext)[0]
            args['path_video'] = os.path.join(path_videos, vid_name_ext)
            args['path_annot'] = os.path.join(path_annots, vid_name_ext)
            print(vid_name_ext)
            print('num of videos {}'.format(num_of_videos))
                           

            # pool.apply_async()
            process_entry(args, )

            num_of_videos += 1
    
    print('num of videos {}'.format(num_of_videos))





