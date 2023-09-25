#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:35:57 2021

@author: rsk3900
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_performance_tables_figures import get_results_mode, get_from_dict
from plot_performance_tables_figures import traverse_nested_dict
from plot_performance_tables_figures import format_results


path_all_results = '/home/rsk3900/sporc/Results/multiset_accumulated_results_by_epoch'

metric_str = 'iou'
exp_cond_base = 'GR-1.2_AUG-1_ADV_DG-0_ADA_IN_NORM-1_PSEUDO_LABELS-0'

list_ds = ['OpenEDS', 'NVGaze', 'UnityEyes', 'riteyes-s-general',
           'riteyes-s-natural', 'LPW', 'Santini', 'Fuhl', 'Swirski']


def plot_trend_per_set(ax, results, ds, epochs):

    baseline_acc = []
    best_cross_acc = []
    all_one_vs_one_acc = []

    for epoch in epochs:
        results_dict = format_results(results_base[epoch],
                                      list_ds, metric_str)

        baseline = np.median(results_dict['one_vs_one'][ds])
        best_cross = np.median(results_dict['best_cross'][ds])
        all_one_vs_one = np.median(results_dict['all-one_vs_one'][ds])

        baseline_acc.append(baseline)
        best_cross_acc.append(best_cross ) # - baseline
        all_one_vs_one_acc.append(all_one_vs_one ) # baseline

    ax.plot(epochs, best_cross_acc, label='best cross')
    ax.plot(epochs, all_one_vs_one_acc, label='all-one_vs_one')
    ax.plot(epochs, baseline_acc, label='all-one_vs_one')
    # ax.legend()



if __name__ == '__main__':

    results_base = {}

    epochs = list(range(5, 80, 5))

    for epoch in epochs:

        results_base[epoch] = {}

        for mode in ['one_vs_one', 'all-one_vs_one', 'all_vs_one']:

            global path_results
            path_results = os.path.join(path_all_results, mode,
                                        str(epoch), exp_cond_base)

            temp_dict = get_results_mode(path_results, list_ds,
                                         mode, exp_cond_base)
            key_list = traverse_nested_dict(temp_dict, [])

            results_base[epoch][mode] = {tuple(entry): get_from_dict(temp_dict, entry)
                                         for entry in key_list}

        # Create a pandas data frame
        results_base[epoch] = pd.DataFrame(results_base[epoch])

    fig, axs = plt.subplots(3, 3)

    count = 0
    for ii in range(3):
        for jj in range(3):
            axs[ii, jj].set_title(list_ds[count])
            plot_trend_per_set(axs[ii, jj],
                               results_base, list_ds[count], epochs)
            count += 1





