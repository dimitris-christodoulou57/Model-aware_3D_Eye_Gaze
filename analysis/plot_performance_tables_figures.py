#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:49:25 2021

@author: rakshit
"""

import os
import sys
import copy
import pickle
import operator
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerBase
from functools import reduce


def normalize_to_zscore(input, median, mad):
    return (input - median)/mad


def traverse_nested_dict(nested_dict, key_list):
    out_list = []
    for key, item in nested_dict.items():
        if type(item) is dict:
            out_list += traverse_nested_dict(item, key_list+[key])
        else:
            out_list.append(key_list + [key])
    return out_list


def get_metrics(array, get_blank):

    # Outlier removal in array
    # array = reject_outliers(array)

    if not get_blank:
        result = {'data': array.squeeze() if np.any(array) else [],
                  'std': np.std(array) if np.any(array) else 0,
                  'mad': stats.median_abs_deviation(array) if np.any(array) else 0,
                  'mean': np.mean(array) if np.any(array) else 0,
                  'median': np.median(array) if np.any(array) else 0}
    else:
        result = {'data': [],
                  'mad': 0,
                  'std': 0,
                  'mean': 0,
                  'median': 0}
    return result


def get_scores(results, get_blank=False):
    result = {}
    if not get_blank:
        result['pupil_c'] = get_metrics(results['pupil_c_dst'], get_blank)
        result['iris_c'] = get_metrics(results['iris_c_dst'], get_blank)
        result['iou'] = get_metrics(results['iou'], get_blank)
    else:
        result['pupil_c'] = get_metrics([], get_blank)
        result['iris_c'] = get_metrics([], get_blank)
        result['iou'] = get_metrics([], get_blank)
    return result


def get_results_mode(path_results, list_ds, mode, exp_cond):

    cond_results = {}

    if mode == 'all_vs_one':
        train_set_list = ['all_vs_one']
    else:
        train_set_list = list_ds

    list_files = os.listdir(path_results)

    for train_ds in train_set_list:
        cond_train_results = {}

        for test_ds in list_ds:

            name_str = 'RESULT_%s_TRAIN_%s_TEST_%s' % (mode,
                                                       train_ds,
                                                       test_ds)

            name_str = [ele for ele in list_files if name_str in ele]

            if len(name_str) != 1:
                import pdb
                pdb.set_trace()

            name_str = name_str[0]

            # Load pickle fille
            full_path = os.path.join(path_results, name_str)

            if os.path.exists(full_path):
                with open(full_path, 'rb') as fid:
                    results = pickle.load(fid)

                # Get required data from results
                result_dict = get_scores(results['test_result'])
                cond_train_results[test_ds] = result_dict
            else:
                print(full_path)
                cond_train_results[test_ds] = get_scores([], get_blank=True)

        cond_results[train_ds] = cond_train_results
    return cond_results


def get_from_dict(data_dict, map_list):
    return reduce(operator.getitem, map_list, data_dict)


def format_results(df, list_ds, metric_str, meas='data'):

    template = {'one_vs_one': {},
                'best_cross': {},
                'all_vs_one': {},
                'all-one_vs_one': {}}

    result_dict = copy.deepcopy(template)

    # Remove data and push to CSV accordingly
    for ii, ds_str in enumerate(list_ds):

        # For a given set, find which dataset performs best for it
        temp = df.xs((ds_str, metric_str, 'median'),
                     level=[1, 2, 3],
                     drop_level=True)['one_vs_one'].to_numpy()[:9]

        all_but_this_one = np.arange(len(temp)) != ii
        loc = np.argmax(temp[all_but_this_one]) \
            if metric_str == 'iou' else \
            np.argmin(temp[all_but_this_one])

        # Because the current position got moved by one space, we must push it
        # back to ensure proper DS assignment
        loc = loc+1 if loc >= ii else loc

        temp = df.xs((list_ds[loc], ds_str, metric_str, meas),
                     level=[0, 1, 2, 3],
                     drop_level=True)

        if type(temp['one_vs_one'][0]) == list:
            result_dict['best_cross'][ds_str] = np.nan
        else:
            result_dict['best_cross'][ds_str] = temp['one_vs_one'][0].tolist()

        temp = df.xs((ds_str, ds_str, metric_str, meas),
                     level=[0, 1, 2, 3],
                     drop_level=True)

        if type(temp['one_vs_one'][0]) == list:
            result_dict['one_vs_one'][ds_str] = np.nan
        else:
            result_dict['one_vs_one'][ds_str] = temp['one_vs_one'][0].tolist()

        if type(temp['all-one_vs_one'][0]) == list:
            result_dict['all-one_vs_one'][ds_str] = np.nan
        else:
            result_dict['all-one_vs_one'][ds_str] = temp['all-one_vs_one'][0].tolist()

        temp = df.xs(('all_vs_one', ds_str, metric_str, meas),
                     level=[0, 1, 2, 3],
                     drop_level=True)

        if type(temp['all_vs_one'][0]) == list:
            result_dict['all_vs_one'][ds_str] = np.nan
        else:
            result_dict['all_vs_one'][ds_str] = temp['all_vs_one'][0].tolist()
    return result_dict


# %% Compile within dataset

if __name__ == '__main__':

    # path_all_results = '/home/rsk3900/sporc/Results/multiset_accumulated_results'
    path_all_results = '/results/multiset_accumulated_results/'

    metric_str = 'iris_c'

    list_ds = ['OpenEDS', 'NVGaze', 'UnityEyes', 'riteyes-s-general',
                'riteyes-s-natural', 'LPW', 'Santini', 'Fuhl', 'Swirski']
    # list_ds = ['LPW', 'Fuhl', 'OpenEDS', 'Santini', 'Swirski', 'NVGaze',
    #            'UnityEyes', 'riteyes-s-general', 'riteyes-s-natural']
    # list_ds = ['LPW', 'Fuhl', 'OpenEDS', 'Santini',  'Swirski']

    # The base condition against which everything should be compared
    # exp_cond_base = 'GR-1.2_AUG-0_ADV_DG-0_IN_NORM-1'
    exp_cond_base = 'GR-1.2_AUG-0_ADV_DG-0_IN_NORM-1_WFLIP'
    # exp_cond_base = 'BN_GR-1.2_AUG-0'
    # exp_cond_base = 'BN_BS-1_GR-1.2_AUG-0'

    # Other conditions
    exp_cond = 'AUG-1_IN_NORM-1_GROWTH-1.0_GRPS-32'
    # exp_cond = 'AUG-1_IN_NORM-1_origimg'
    # exp_cond = 'AUG-0_GRADREV-0_UNCERTAIN-0_ADA_IN_NORM-0_IN_NORM-0'
    # exp_cond = 'GR-1.2_AUG-0_ADV_DG-0_IN_NORM-1'
    # exp_cond = 'GR-1.2_AUG-0_ADV_DG-0_IN_NORM-1_WFLIP'
    # exp_cond = 'AUG-1_ADA_IN_NORM-1'
    # exp_cond = 'GR-1.2_AUG-1_ADV_DG-0_IN_NORM-1_WFLIP'
    # exp_cond = 'AUG-1_IN_NORM-1_GRAD_REV-1'
    # exp_cond = 'AUG-0_IN_NORM-1_SC-0.2'
    # exp_cond = 'BN_GR-1.2_AUG-0'
    # exp_cond = 'BN_BS-1_GR-1.2_AUG-0'
    # exp_cond = 'GR-1.2_AUG-1_ADV_DG-0_ADA_IN_NORM-1_PSEUDO_LABELS-0'
    # exp_cond = 'GR-1.2_AUG-1_ADV_DG-0_IN_NORM-1_PSEUDO_LABELS-1'

    results = {}
    results_base = {}

    for mode in ['one_vs_one', 'all-one_vs_one', 'all_vs_one']:

        path_results = os.path.join(path_all_results, mode, exp_cond)

        temp_dict = get_results_mode(path_results, list_ds, mode, exp_cond)
        key_list = traverse_nested_dict(temp_dict, [])

        results[mode] = {tuple(entry): get_from_dict(temp_dict, entry)
                         for entry in key_list}

        path_results = os.path.join(path_all_results, mode, exp_cond_base)

        temp_dict = get_results_mode(path_results, list_ds, mode, exp_cond_base)
        key_list = traverse_nested_dict(temp_dict, [])

        results_base[mode] = {tuple(entry): get_from_dict(temp_dict, entry)
                              for entry in key_list}

    # Create a pandas data frame
    df = pd.DataFrame(results)
    df_base = pd.DataFrame(results_base)

    # %% For Google sheets

    meas = 'mad'
    one_vs_one_tab = np.zeros((9, 9))
    all_one_vs_one_tab = np.zeros((9, 9))
    all_vs_one_tab = np.zeros((1, 9))

    # Remove data and push to CSV accordingly
    for ii, train_ds in enumerate(list_ds):
        temp = df.xs((train_ds, metric_str, meas),
                      level=[0, 2, 3],
                      drop_level=True).to_numpy()
        one_vs_one_tab[ii, :] = temp[:, 0]
        all_one_vs_one_tab[ii, :] = temp[:, 1]

    temp = df.xs(('all_vs_one', metric_str, meas),
                  level=[0, 2, 3],
                  drop_level=True).to_numpy()
    all_vs_one_tab[0, :] = temp[:, 2]

    one_vs_one_tab_diag = np.diag(one_vs_one_tab).reshape(1, -1)
    all_one_vs_one_tab_diag = np.diag(all_one_vs_one_tab).reshape(1, -1)

    best_one_vs_one = np.zeros((1, 9))
    for ii in range(len(list_ds)):
        temp = one_vs_one_tab[:, ii]
        temp = np.delete(temp, ii)  # Remove within dataset result
        best_one_vs_one[0, ii] = np.max(temp) \
            if metric_str == 'iou' else \
            np.min(temp)


    # %% For plotting

    result_dict = format_results(df, list_ds, metric_str, meas='data')
    result_base_dict = format_results(df_base, list_ds, metric_str, meas='data')
    
    # for key in result_dict.keys():
    #     result_dict[key].pop('NVGaze', None)
    #     result_dict[key].pop('UnityEyes', None)
    #     result_dict[key].pop('riteyes-s-general', None)
    #     result_dict[key].pop('riteyes-s-natural', None)

    num = 9
    step = 5

    start = 0

    fig, axs = plt.subplots()

    bp0 = axs.boxplot([ele - np.median(result_base_dict['one_vs_one'][key]) for key, ele in result_dict['one_vs_one'].items()],
                      notch=True, sym='', patch_artist=True,
                      positions=np.arange(0, num)*step + start,
                      widths=0.75)

    start = 1

    bp1 = axs.boxplot([ele - np.median(result_base_dict['one_vs_one'][key]) for key, ele in result_dict['all_vs_one'].items()],
                      notch=True, sym='', patch_artist=True,
                      positions=np.arange(0, num)*step + start,
                      widths=0.75)

    start = 2

    bp2 = axs.boxplot([ele - np.median(result_base_dict['one_vs_one'][key]) for key, ele in result_dict['all-one_vs_one'].items()],
                      notch=True, sym='', patch_artist=True,
                      positions=np.arange(0, num)*step + start,
                      widths=0.75)

    start = 3

    bp3 = axs.boxplot([ele - np.median(result_base_dict['one_vs_one'][key]) for key, ele in result_dict['best_cross'].items()],
                      notch=True, sym='', patch_artist=True,
                      positions=np.arange(0, num)*step + start,
                      widths=0.75)

    ds_names = result_dict['one_vs_one'].keys()
    # ds_names = [ele.replace('riteyes-', '') for ele in ds_names]
    ds_names = [ele.replace('Santini', 'BAT') for ele in ds_names]
    ds_names = [ele.replace('riteyes-s-general', 'RITEyes-gen') for ele in ds_names]
    ds_names = [ele.replace('riteyes-s-natural', 'RITEyes-nat') for ele in ds_names]

    if 'iou' in metric_str:
        axs.set_ylabel('Relative mIoU', fontsize=14)
    else:
        axs.set_ylabel('Relative pixel error', fontsize=14)

    axs.set_xticks(np.arange(0, num)*step + 0)
    axs.set_xticklabels(ds_names, rotation=45)

    colors = ['pink', 'lightblue', 'darkgreen', 'lightgreen']
    for bplot, color in zip((bp0, bp1, bp2, bp3), colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    axs.legend([bp0["boxes"][0],
                bp1["boxes"][0],
                bp2["boxes"][0],
                bp3["boxes"][0]],
               ['within dataset', 'all-vs-one', 'leave-one-out', 'best cross'])

    axs.grid(axis='y')

    axs.hlines(0, 0, num*5, colors='gray', )

    # axs.set_title(metric_str)
    if metric_str == 'iou':
        axs.set_ylim(-0.4, 0.1)
        # axs.set_ylabel('IOU')
    else:
        axs.set_ylim(-4, 8)
        # axs.set_ylabel('Error (px)')
    plt.show(block=True)
    plt.tight_layout()

    fig.savefig(exp_cond+'_'+metric_str+'.png',
                dpi=600, transparent=True,
                bbox_inches='tight')
