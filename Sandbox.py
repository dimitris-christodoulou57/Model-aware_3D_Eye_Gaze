#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz

The purpose of this script is to verify if train/test objects are working as
intended. This function will iterate over H5 files and display groundtruth
annotations (whichever are present)
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Deactive file locking

sys.path.append('./helperfunctions')

from torch.utils.data import DataLoader

from utils import points_to_heatmap
from helperfunctions.data_augment import augment
from helperfunctions.helperfunctions import plot_images_with_annotations
from helperfunctions.CurriculumLib import readArchives, listDatasets, generate_fileList
from helperfunctions.CurriculumLib import selDataset, selSubset, DataLoader_riteyes


if __name__=='__main__':
    path2data = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets'
    path2h5 = os.path.join(path2data, 'All')
    path2arc_keys = os.path.join(path2data, 'MasterKey')

    # Create a curriculum object
    AllDS = readArchives(path2arc_keys)
    AllDS_cond = selDataset(AllDS, ['OpenEDS']) # , 'Fuhl' 'UnityEyes', 'NVGaze'
    dataDiv_obj = generate_fileList(AllDS_cond, mode='vanilla', notest=True)

    trainObj = DataLoader_riteyes(dataDiv_obj,
                                  path2h5,
                                  'train',
                                  augFlag=False,
                                  size=(480, 640),
                                  sort='mutliset_random',
                                  scale=0.5)

    validObj = DataLoader_riteyes(dataDiv_obj,
                                  path2h5,
                                  'valid',
                                  augFlag=False,
                                  size=(480, 640),
                                  sort='mutliset_random',
                                  scale=0.5)

    # Train loader
    trainLoader = DataLoader(trainObj,
                             batch_size=18,
                             shuffle=False,
                             num_workers=8,
                             drop_last=True)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    totTime = []
    startTime = time.time()
    for bt, data_dict in enumerate(trainLoader):

        # Display annotated image
        plot_images_with_annotations(data_dict, show=False,
                                     write='{}.png'.format(bt),
                                     is_list_of_entries=False)

        dT = time.time() - startTime
        totTime.append(dT)
        print('Batch: {}. Time: {}'.format(bt, dT))
        startTime = time.time()

    print('Time for 1 epoch: {}'.format(np.sum(totTime)))
