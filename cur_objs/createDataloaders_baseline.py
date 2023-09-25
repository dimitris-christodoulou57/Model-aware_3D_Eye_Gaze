#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: rakshit

This file generates objects with train and testing split information for each
dataset. Each dataset has a predefined train and test partition. For more info
on the partitions, please see the file <datasetSelections.py>
'''

import os
import sys
import pickle
import numpy as np

sys.path.append('..')
import helperfunctions.CurriculumLib as CurLib
from helperfunctions.CurriculumLib import DataLoader_riteyes

path2data = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets'
path2h5 = os.path.join(path2data, 'All')
keepOld = False

DS_sel = pickle.load(open('dataset_selections.pkl', 'rb'))
AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
#list_ds = ['OpenEDS','sequence', 'S']
list_ds = ['TEyeD']

args={}
args['train_data_percentage'] = 1.0
args['net_ellseg_head'] =False
args['loss_w_rend_pred_2_gt_edge'] = 0.1
args['loss_w_rend_gt_2_pred'] = 0.1
args['loss_w_rend_pred_2_gt'] = 0.0
args['net_ellseg_head'] = 0.0

# Generate objects per dataset
for setSel in list_ds:
    # Train object
    AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train'][setSel])
    dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=True)
    trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640), 
                                  scale=0.5, num_frames=4, args=args)
    validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640), 
                                  scale=0.5, num_frames=4,args=args)
    # Test object
    AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][setSel])
    dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=True)
    testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'test', False, (480, 640), 
                                 scale=0.5, num_frames=4, args=args)

    if setSel == 'S':
        path2save = os.path.join(os.getcwd(), 'one_vs_one', 'cond_'+'OpenEDS_S'+'.pkl')
    else:
        path2save = os.path.join(os.getcwd(), 'one_vs_one', 'cond_'+setSel+'.pkl')
    if os.path.exists(path2save) and keepOld:
        print('Preserving old selections ...')

        # This ensure that the original selection remains the same
        trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
        trainObj.imList = trainObj_orig.imList
        validObj.imList = validObj_orig.imList
        testObj.imList = testObj_orig.imList
        pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
    else:
        print('Save data')
        pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
