# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

import pickle as pkl

# OpenEDS
openeds_train = ['train']
openeds_test = ['validation']

# LPW
lpw_subs_train = ['LPW_{}'.format(i) for i in [1,3,4,6,9,10,12,13,16,18,20,21]]
lpw_subs_test = ['LPW_{}'.format(i) for i in [2,5,7,8,11,14,15,17,19,22]]

# S-General
riteyes_subs_train_gen = ['riteyes-s-general_{}'.format(i+1) for i in range(0, 18)]
riteyes_subs_test_gen = ['riteyes-s-general_{}'.format(i+1) for i in range(18, 24)]

# S-Natural
riteyes_subs_train_nat = ['riteyes-s-natural_{}'.format(i+1) for i in range(0, 18)]
riteyes_subs_test_nat = ['riteyes-s-natural_{}'.format(i+1) for i in range(18, 24)]

# OpenEDS sequence 
openEDS_sequence_train = ['sequence_train']
openEDS_sequence_test = ['sequence_validation']

#Ours Datasets 
Ours_train = ['PI_left_v1_ps1'] + ['PI_right_v1_ps1'] 
Ours_test = ['PI_left_v1_ps1'] + ['PI_right_v1_ps1']

Sequence_whole_train = ['S_{}'.format(i) for i in range(0,150)]
Sequence_whole_test = ['S_{}'.format(i) for i in range(150,199)]

#TEyeD datasets
TEyeD_train = ['DikablisR_1_1'] + \
              ['DikablisSS_{}'.format(i+1) for i in range(0,30)] + \
              ['DikablisSA_{}'.format(i+1) for i in range(0,30)] + \
              ['DikablisT_{}'.format(i) for i in range(0,17)]

TEyeD_test = ['DikablisR_1_1'] + \
             ['DikablisSS_{}'.format(i) for i in range(30,37)] + \
             ['DikablisSA_{}'.format(i) for i in range(30,39)] + \
             ['DikablisT_{}'.format(i) for i in range(17,32)]

# %% Generate split dictionaries
DS_train = {
            'OpenEDS': openeds_train,
            'riteyes-s-general': riteyes_subs_train_gen,
            'riteyes-s-natural': riteyes_subs_train_nat,
            'sequence' : openEDS_sequence_train,
            'S': Sequence_whole_train,
            'TEyeD': TEyeD_train,
            'Ours' : Ours_train
            }

DS_test = {
           'OpenEDS': openeds_test,
           'riteyes-s-general': riteyes_subs_test_gen,
           'riteyes-s-natural': riteyes_subs_test_nat,
           'sequence' : openEDS_sequence_test,
           'S': Sequence_whole_test,
           'TEyeD': TEyeD_test,
           'Ours': Ours_test
           } 

DS_selections = {'train': DS_train,
                 'test': DS_test}

pkl.dump(DS_selections, open('dataset_selections.pkl', 'wb'))
