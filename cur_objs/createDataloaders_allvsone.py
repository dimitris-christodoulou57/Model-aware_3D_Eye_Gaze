#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

@author: rakshit
'''
import os
import sys
import pickle

sys.path.append('..')
import helperfunctions.CurriculumLib as CurLib
from helperfunctions.CurriculumLib import DataLoader_riteyes

DS_sel = pickle.load(open('dataset_selections.pkl', 'rb'))
path2data = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets'
path2h5 = os.path.join(path2data, 'All')
keepOld = True

list_ds = list(DS_sel['train'].keys())

subsets_train = []
subsets_test  = []
for setSel in list_ds:
    subsets_train += DS_sel['train'][setSel]
    subsets_test  += DS_sel['test'][setSel]

AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))

# Train & test object
AllDS_train = CurLib.selSubset(AllDS, subsets_train)
AllDS_test  = CurLib.selSubset(AllDS, subsets_test)

dataDiv_obj = CurLib.generate_fileList(AllDS_train, mode='vanilla', notest=True)
dataDiv_obj_test = CurLib.generate_fileList(AllDS_test, mode='none', notest=True)

trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'train', True, (480, 640), scale=0.5)
validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 'valid', False, (480, 640), scale=0.5)
testObj  = DataLoader_riteyes(dataDiv_obj_test, path2h5, 'test', False, (480, 640), scale=0.5)

path2save = os.path.join(os.getcwd(), 'all_vs_one', 'cond_'+'allvsone'+'.pkl')
if os.path.exists(path2save) and keepOld:
    print('Preserving old selections ...')
    # This ensure that the original selection remains the same
    trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
    trainObj.imList = trainObj_orig.imList
    validObj.imList = validObj_orig.imList
    pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
else:
    pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
