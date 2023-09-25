# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 07:21:10 2021

@author: Rudra
"""

import os
import sys
import pickle
import pprint
import argparse


def join_str(*args):
    out_str = ''
    for idx, ele in enumerate(args):
        str_add = ele
        if idx != (len(args)-1):
            str_add += '_'
        out_str += str_add
    return out_str


sys.path.append('..')

default_repo = '/home/rsk3900/Documents/Python_Scripts/multiset_gaze/src'

parser = argparse.ArgumentParser()

parser.add_argument('--path_exp_tree',
                    default='/results/test_results_blank_folders',
                    help='path to create blank test results folder',
                    type=str)
parser.add_argument('--path_results',
                    default='/results',
                    help='path to all experiments results')
parser.add_argument('--mode', type=str, default='all-one_vs_one',
                    help='mode you want to test out')
parser.add_argument('--path_acc_results',
                    default='/results/multiset_accumulated_results',
                    help='path to accumulate all results',
                    type=str)
parser.add_argument('--path_data',
                    default='/data/datasets/All',
                    help='path to all H5 file data',
                    type=str)
parser.add_argument('--exp_cond',
                    default='GR-1.2_AUG-1_ADV_DG-1_ADA_IN_NORM-1_PSEUDO_LABELS-1',
                    help='exact exp condition you want results for',
                    type=str)
parser.add_argument('--local_rank', type=int, default=0,
                    help='rank to set GPU')
parser.add_argument('--batch_size', type=int, default=32,
                    help='testing batchsize')
parser.add_argument('--repo_root', type=str,
                    default=default_repo,
                    help='path to repo root')
parser.add_argument('--save_test_maps',
                    action='store_true',
                    help='save out test maps')

args = parser.parse_args()
args = vars(args)
pprint.pprint(args)

DS_selections = pickle.load(open('./cur_objs/dataset_selections.pkl', 'rb'))
DS_present = list(DS_selections['train'].keys())

train_itr_list = ['all_vs_one'] if args['mode'] == 'all_vs_one' else DS_present

for train_ds in train_itr_list:
    for test_ds in DS_present:

        print('----------------------')
        print('Mode: {}'.format(args['mode']))
        print('Trained on: {}'.format(train_ds))
        print('Test on: {}'.format(test_ds))

        exp_name = join_str('RESULT',
                            args['mode'],
                            'TRAIN',
                            train_ds,
                            'TEST',
                            test_ds,
                            args['exp_cond'])

        path_to_find_model = os.path.join(args['path_results'],
                                          args['mode'],
                                          args['exp_cond'],
                                          )

        possible_paths = []
        for path in os.listdir(path_to_find_model):
            if (train_ds in path):
                possible_paths.append(path)

        assert len(possible_paths) <= 1, 'only one such model must exist'

        if (possible_paths):
            path_model = os.path.join(path_to_find_model,
                                      possible_paths[0],
                                      'results',
                                      'best_model.pt')

            path_acc_results = os.path.join(args['path_acc_results'],
                                            args['mode'],
                                            args['exp_cond'])

            run_cmd = 'python run.py '
            run_cmd += '--repo_root=%s ' % args['repo_root']
            run_cmd += '--path_data=%s ' % args['path_data']
            run_cmd += '--path_model=%s ' % path_model
            run_cmd += '--cur_obj=%s ' % test_ds  # Set test cur obj
            run_cmd += '--path_exp_tree=%s ' % args['path_exp_tree']
            run_cmd += '--save_results_here=%s ' % (path_acc_results+'/'+exp_name+'.pkl')
            run_cmd += '--exp_name=%s ' % exp_name
            run_cmd += '--local_rank=%d ' % args['local_rank']
            run_cmd += '--batch_size=%d ' % args['batch_size']
            run_cmd += '--workers=32 '
            run_cmd += '--only_test=1 '
            run_cmd += '--adv_DG=1 '
            run_cmd += '--maxpool_in_regress_mod=0 '
            run_cmd += '--use_instance_norm=0 '
            run_cmd += '--use_ada_instance_norm=1 '

            if not os.path.exists(path_acc_results+'/'+exp_name+'.pkl'):
                os.system(run_cmd)
            else:
                print('DONE!')
                print(run_cmd)
