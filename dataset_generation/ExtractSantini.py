#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:16:57 2019

@author: rakshit
"""
import os
import sys
import cv2
import h5py
import argparse
import matplotlib
import numpy as np
import deepdish as dd
import scipy.io as scio
import matplotlib.pyplot as plt

sys.path.append('..')
from helperfunctions.helperfunctions import generateEmptyStorage, mypause

print('Extracting Santini')

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds', help='Path to dataset', type=str, default='/media/rakshit/Monster/Datasets/')
args = parser.parse_args()
if args.noDisp:
    noDisp = True
    print('No graphics')
else:
    noDisp = False
    print('Showing figures')

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

PATH_DIR = os.path.join(args.path2ds, 'Santini')
PATH_DS = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')
list_ds = ['1', '2', '3', '4', '5', '6']

sc = (640.0/384.0)
Image_counter = 0.0
ds_num = 0

def fix_pupil_loc(p, res):
    # res: [H, W]
    p[0] = 0.5*p[0]
    p[1] = res[0] - 0.5*p[1]
    return p

def readFormattedText(path2file, ignoreLines):
    data = []
    count = 0
    f = open(path2file, 'r')
    for line in f:
        d = [int(d) for d in line.split() if d.isdigit()]
        count = count + 1
        if d and count > ignoreLines:
            data.append(d)
    f.close()
    return data

for name in list_ds:
    # Ignore the first row and column.
    # Columns: [index, p_x, p_y]
    opts = os.listdir(os.path.join(PATH_DIR, name))

    for subdir in opts:
        PATH_DATA = os.path.join(PATH_DIR, name, subdir)

        # Read pupil data
        Path2text = os.path.join(PATH_DATA, 'journal-{:04d}.txt'.format(int(subdir)-1))
        Path2vid = os.path.join(PATH_DATA, 'eye-{:04d}-0000.avi'.format(int(subdir)-1))
        PupilData = np.array(readFormattedText(Path2text, 2))
        VidObj = cv2.VideoCapture(Path2vid)

        # Create an empty dictionary as per agreed structure
        Data, keydict = generateEmptyStorage(name='Santini', subset='Santini_{}'.format(name))

        ds_name = keydict['dataset'] + '_' + keydict['subset'] + '_' + str(ds_num)

        if not noDisp:
            fig, plts = plt.subplots(1,1)

        fr_num = 0
        while(VidObj.isOpened()):
            ret, I = VidObj.read()

            if ret == True:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                I = cv2.resize(I, (640, 480), cv2.INTER_LANCZOS4)

                Data['Images'].append(I)
                keydict['resolution'].append(I.shape)
                keydict['archive'].append(ds_name)

                pupil_loc = fix_pupil_loc(PupilData[fr_num, 10:12]*sc, I.shape)

                keydict['subject_id'].append(str(name))
                Data['subject_id'].append(str(name))

                keydict['pupil_loc'].append(pupil_loc)
                keydict['Info'].append(str(fr_num))

                Data['pupil_loc'].append(pupil_loc)
                Data['Info'].append(str(fr_num))

                fr_num+=1
                Image_counter+=1
                if not noDisp:
                    if fr_num == 1:
                        cI = plts.imshow(I)
                        cX = plts.scatter(pupil_loc[0], pupil_loc[1])
                        plt.show()
                        plt.pause(.01)
                    else:
                        newLoc = np.array([pupil_loc[0], pupil_loc[1]])
                        cI.set_data(I)
                        cX.set_offsets(newLoc)
                        mypause(0.01)
            else: # No more frames to load
                break

        Data['Images'] = np.stack(Data['Images'], axis=0)
        Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
        Data['subject_id'] = np.stack(Data['subject_id'], axis=0)

        keydict['archive'] = np.stack(keydict['archive'], axis=0)
        keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
        keydict['subject_id'] = np.stack(keydict['subject_id'], axis=0)
        keydict['resolution'] = np.stack(keydict['resolution'], axis=0)

        # Save out data
        dd.io.save(os.path.join(PATH_DS, str(ds_name)+'.h5'), Data)
        scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)
        ds_num=ds_num+1