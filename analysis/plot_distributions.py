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
import h5py
import tqdm
import scipy
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import numpy.fft as fft
import multiprocessing as mp
import matplotlib.pyplot as plt

sys.path.append('..')

from models_mux import model_dict
from skimage.transform import rescale
from helperfunctions.utils import normPts
from helperfunctions.helperfunctions import plot_2D_hist
from helperfunctions.helperfunctions import image_contrast

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str,
                        default='/data/datasets/All')
    parser.add_argument('--sel_ds', type=str,
                        default='all')  # default='riteyes-s-natural'
    parser.add_argument('--path_model', type=str,
                        default='')
    args = parser.parse_args()
    return args


def find_fname(path, name):
    options = [ele for ele in os.listdir(path) if name in ele]
    return options


def accumulate_stats_per_entry(im_num, h5_obj):

    global net

    scale_ratio = 0.5

    # Histogram of image
    image = h5_obj['Images'][im_num, ...]

    dsize = (round(scale_ratio*image.shape[1]),
             round(scale_ratio*image.shape[0]))

    image = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)

    image_norm = (image - np.mean(image))/np.std(image)

    height_width = image.shape[:2]

    ''' Justification
    We must normalize by the sum to preserve original statistics. Normalizing
    by the maximum value will skew statistics and not represent the true
    distribution.
    '''

    # ML forward operation
    data_dict = {}
    data_dict['image'] = torch.from_numpy(image_norm).to(torch.float32)
    data_dict['image'] = data_dict['image'].reshape((1,) + height_width).cuda()

    with torch.no_grad():
        out_dict = net(data_dict)

    # Compute contrast at scales
    contrast_scales = [1, 1/2, 1/4, 1/8, 1/16]
    contrast, contrast_by_class = image_contrast(image, contrast_scales,
                                                 by_category=out_dict['mask'].squeeze())

    # FFT
    # image_fft = np.abs(fft.fftshift(fft.fft2(image)))

    # Ratio of samples
    ratio = [np.sum(out_dict['mask'] == 0),
             np.sum(out_dict['mask'] == 1),
             np.sum(out_dict['mask'] == 2)]

    try:
        assert np.sum(ratio) == np.prod(height_width), 'Something wrong'
    except:
        import pdb; pdb.set_trace()

    ratio = np.array(ratio)/np.prod(height_width)

    # Pupil center location
    counts = scipy.ndimage.histogram(image.flatten(), 0, 255, 128,)
    counts = counts/np.sum(counts)  # Actual probability score

    try:
        counts_pupil = scipy.ndimage.histogram(image.flatten(), 0, 255, 128,
                                               labels=out_dict['mask'].flatten(),
                                               index=2)
        counts_pupil = counts_pupil / np.sum(counts_pupil) if ratio[2] > 0 else np.zeros(128, )

        counts_iris = scipy.ndimage.histogram(image.flatten(), 0, 255, 128,
                                              labels=out_dict['mask'].flatten(),
                                              index=1)
        counts_iris = counts_iris / np.sum(counts_iris) if ratio[1] > 0 else np.zeros(128, )

        counts_bg = scipy.ndimage.histogram(image.flatten(), 0, 255, 128,
                                            labels=out_dict['mask'].flatten(),
                                            index=0)
        counts_bg = counts_bg / np.sum(counts_bg) if ratio[0] > 0 else np.zeros(128, )
    except:
        import pdb; pdb.set_trace()

    if h5_obj['pupil_loc'].__len__() != 0:
        pupil_center = h5_obj['pupil_loc'][im_num, ...]
    else:
        pupil_center = -np.ones(2, )

    # Iris center location
    if h5_obj['Fits']['iris'].__len__() != 0:
        iris_center = h5_obj['Fits']['iris'][im_num, :2]
    else:
        iris_center = -np.ones(2, )

    datum = {}
    # datum['fft'] = image_fft.astype(np.float32)
    datum['hist'] = counts
    datum['ratio'] = ratio
    datum['iris_c'] = iris_center
    datum['pupil_c'] = pupil_center
    datum['contrast'] = contrast
    datum['hist_pupil'] = counts_pupil
    datum['hist_iris'] = counts_iris
    datum['hist_bg'] = counts_bg
    datum['contrast_by_class'] = contrast_by_class
    return datum


def accumulate_stats_per_subset(path_H5):

    h5_obj = h5py.File(path_H5, 'r', swmr=True)
    num_images = h5_obj['Images'].shape[0]
    stats = []

    for idx in range(num_images):
        stats += [accumulate_stats_per_entry(idx, h5_obj)]

    h5_obj.close()

    return stats


def accumulate_stats(args, subsets):

    # Load ML model
    model_data = torch.load(args['path_model'], map_location='cpu')

    global net
    net = model_dict[model_data['args']['model']](model_data['args'],
                                                  norm=nn.InstanceNorm2d,
                                                  act_func=nn.functional.leaky_relu)

    net.load_state_dict(model_data['state_dict'], strict=True)
    net.cuda()

    stats = []

    def log_result(result):
        stats.append(result)

    for subset in tqdm.tqdm(subsets):
        try:
            subset_fname = find_fname(args['path_data'], subset)[0]
        except RuntimeError:
            import pdb
            pdb.set_trace()
        path_H5 = os.path.join(args['path_data'], subset_fname)

        log_result(accumulate_stats_per_subset(path_H5))

    return stats


def collapse_stats(stats):
    data = []
    for value in stats:
        data += value
    return data


def print_stats(stats, mode):

    # Number of images
    print('%s. # of images: %d' % (mode, len(stats)))


def plot_stats(stats, mode, axs=None):

    num_samples = len(stats)

    pupil_c = [ele['pupil_c'] for ele in stats]
    pupil_c = np.stack(pupil_c, axis=0).squeeze()

    iris_c = [ele['iris_c'] for ele in stats]
    iris_c = np.stack(iris_c, axis=0).squeeze()

    # l_fft = [ele['fft'] for ele in stats]
    # l_fft = np.stack(l_fft, axis=0).squeeze()

    contrast = [ele['contrast_by_class'] for ele in stats]
    contrast = np.stack(contrast, axis=0).squeeze()

    l_hist = [ele['hist'] for ele in stats]
    l_hist = np.stack(l_hist, axis=0).squeeze()

    p_hist = [ele['hist_pupil'] for ele in stats]
    p_hist = np.stack(p_hist, axis=0).squeeze()

    i_hist = [ele['hist_iris'] for ele in stats]
    i_hist = np.stack(i_hist, axis=0).squeeze()

    bg_hist = [ele['hist_bg'] for ele in stats]
    bg_hist = np.stack(bg_hist, axis=0).squeeze()
    
    class_ratio = [ele['ratio'] for ele in stats]
    class_ratio = np.stack(class_ratio, axis=0).squeeze()

    x_range = np.linspace(0, 255, 128)

    loc = p_hist.sum(axis=1) != 0

    x_range_all = np.stack([x_range for i in range(num_samples)], axis=0)
    dc_pupil_lum_all = np.average(x_range_all[loc, ...],
                                  weights=p_hist[loc, ...],
                                  axis=1)
    rms_pupil_lum_all = np.average((x_range_all[loc, ...] - dc_pupil_lum_all[..., np.newaxis]) ** 2,
                                   weights=p_hist[loc, ...], axis=1)**0.5
    dc_pupil_lum = np.mean(dc_pupil_lum_all)
    # std_pupil_lum = np.sqrt(np.mean(np.power(rms_pupil_lum_all, 2)))
    std_pupil_lum = np.mean(rms_pupil_lum_all)

    # l_fft = l_fft.mean(axis=0)
    l_mean = l_hist.mean(axis=0)
    p_hist = p_hist.mean(axis=0)
    i_hist = i_hist.mean(axis=0)
    bg_hist = bg_hist.mean(axis=0)
    contrast = contrast.mean(axis=0)[:-2, 2, :]

    # p_mad = scipy.stats.median_abs_deviation(p_hist, axis=0)
    # i_mad = scipy.stats.median_abs_deviation(i_hist, axis=0)
    # bg_mad = scipy.stats.median_abs_deviation(bg_hist, axis=0)

    # l_mean = np.median(l_hist, axis=0)
    # p_hist = np.median(p_hist, axis=0)
    # i_hist = np.median(i_hist, axis=0)
    # bg_hist = np.median(bg_hist, axis=0)

    if axs is None:
        plot_2D_hist(pupil_c[:, 0], pupil_c[:, 1],
                     [0, 640], [0, 480],
                     axs=None,
                     str_save=mode+'_pupil_hist.jpg')

        plot_2D_hist(iris_c[:, 0], iris_c[:, 1],
                     [0, 640], [0, 480],
                     axs=None,
                     str_save=mode+'_iris_hist.jpg')

        fig, axs = plt.subplots()
        axs.plot(x_range, l_mean, '-', label='overall')
        axs.plot(x_range, p_hist, '-', label='pupil')
        axs.plot(x_range, i_hist, '-', label='iris')
        axs.plot(x_range, bg_hist, '-', label='bg')
        axs.legend()
        fig.savefig(mode+'_L_hist.jpg', dpi=600)
    else:
        if np.all(pupil_c == -1):
            axs[0].axis('off')
        else:
            plot_2D_hist(pupil_c[:, 0], pupil_c[:, 1],
                         [0, 640], [0, 480],
                         axs=axs[0],
                         str_save=mode+'_pupil_hist.jpg')

        # Do not bother with iris distributions
        # if np.all(iris_c == -1):
        #     axs[1].axis('off')
        # else:
        #     plot_2D_hist(iris_c[:, 0], iris_c[:, 1],
        #                  [0, 640], [0, 480],
        #                  axs=axs[1],
        #                  str_save=mode+'_iris_hist.jpg')

        def norm_minmax(data, template=[]):
            if len(template):
                data - template.min()
                data = data/(template - template.min()).max()
            else:
                data = data - data.min()
                data = data/data.max()
            return data

        # axs[1].plot(x_range, norm_minmax(l_mean), '-', label='overall')
        axs[1].plot(x_range, norm_minmax(p_hist), '-', label='pupil', color='r')
        axs[1].plot(x_range, norm_minmax(i_hist), '-', label='iris', color='g')
        # axs[1].plot(x_range, norm_minmax(bg_hist), '-', label='bg', color='b')
        # axs[1].fill_between(x_range,
        #                     norm_minmax(p_hist-p_mad, p_hist),
        #                     norm_minmax(p_hist+p_mad, p_hist),
        #                     color='r', alpha=0.2)
        # axs[1].fill_between(x_range,
        #                     norm_minmax(i_hist-i_mad, i_hist),
        #                     norm_minmax(i_hist+i_mad, i_hist),
        #                     color='g', alpha=0.2)
        # axs[1].fill_between(x_range,
        #                     norm_minmax(bg_hist-bg_mad, bg_hist),
        #                     norm_minmax(bg_hist+bg_mad, bg_hist),
        #                     color='b', alpha=0.2)
        axs[1].set_ylim(-0.2, 1.2)
        
        # Ratio plot
        bp = axs[2].boxplot(class_ratio[:, 1:],
                            patch_artist=True,
                            showfliers=False,
                            labels=['Iris', 'Pupil']) # Bg', 
        
        # bp['boxes'][0].set_facecolor('b')
        bp['boxes'][0].set_facecolor('g')
        bp['boxes'][1].set_facecolor('r')
        
        axs[2].set_ylim(0, 0.4)

        # Stats plot

        # Contrast plot
        # lines = axs[2].plot(contrast.T)
        # axs[2].legend(lines, ['1', '1/2', '1/4', '1/8', '1/16'])

        # FFT plots
        # l_fft = l_fft - l_fft.min()
        # l_fft = l_fft / l_fft.max()
        # l_fft = np.log10(l_fft+1)

        # axs[2].imshow(l_fft)

        return dc_pupil_lum, std_pupil_lum


def analyze_dataset(args, sel_ds, axs=None):
    stats_dict = {}

    if os.path.exists('./{}_stats.pkl'.format(sel_ds)):
        with open('./{}_stats.pkl'.format(sel_ds), 'rb') as fid:
            print('Loading stats from disk')
            stats_dict = pickle.load(fid)
    else:
        stats_dict['train'] = accumulate_stats(args,
                                               DS_selections['train'][sel_ds])
        stats_dict['test'] = accumulate_stats(args,
                                              DS_selections['test'][sel_ds])

        with open('./{}_stats.pkl'.format(sel_ds), 'wb') as fid:
            print('Saving stats to disk')
            pickle.dump(stats_dict, fid)

    stats_train = collapse_stats(stats_dict['train'])
    stats_test = collapse_stats(stats_dict['test'])

    # Save memory
    del stats_dict

    print_stats(stats_train, sel_ds+'_train')
    print_stats(stats_test, sel_ds+'_test')

    dc_pupil_lum, rms_pupil_lum = plot_stats(stats_train, sel_ds+'_train',
                                             axs=None if axs is None else axs[:3])

    plot_stats(stats_test, sel_ds+'_test',
               axs=None if axs is None else axs[3:])

    return dc_pupil_lum, rms_pupil_lum


if __name__ == '__main__':
    args = vars(make_args())

    args['path_model'] = '/home/rsk3900/sporc/Results/all_vs_one/GR-1.2_AUG-1_ADV_DG-0_IN_NORM-1_WFLIP/WFLIP_all_vs_one_AUG-1_ADA_IN_NORM-0_PSEUDO_LABELS-0_08_07_21_20_03_32/results/best_model.pt'

    with open('../cur_objs/dataset_selections.pkl', 'rb') as f:
        DS_selections = pickle.load(f)

    fig, axs = plt.subplots(9, 6, figsize=(10, 15))

    luma_dict = {}
    if args['sel_ds'] == 'all':
        list_ds = list(DS_selections['train'].keys())
        list_ds_names_fixed = []
        for idx, sel_ds in enumerate(list_ds):
            for ax in axs[idx, ...]:
                if idx != 8:
                    ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

            ds_name = sel_ds
            ds_name = ds_name.replace('riteyes-s-natural', 'RITEyes-nat')
            ds_name = ds_name.replace('riteyes-s-general', 'RITEyes-gen')
            ds_name = ds_name.replace('Santini', 'BAT')

            list_ds_names_fixed.append(ds_name)

            # Add title to central subplot
            axs[idx, 0].set_title(ds_name, fontdict={'fontsize': 18})
            axs[idx, 3].set_title(ds_name, fontdict={'fontsize': 18})

            luma_dict[sel_ds] = analyze_dataset(args, sel_ds,
                                                axs=axs[idx, ...])
        fig.tight_layout()
        fig.savefig('all_dists.png', dpi=600, transparent=True)

        # fig, axs = plt.subplots()
        # for idx, (key, item) in enumerate(luma_dict.items()):
        #     axs.errorbar(idx, item[0], yerr=item[1])
        # axs.set_xticks(list(range(9)))
        # axs.set_xticklabels(list_ds_names_fixed, rotation=45)

    else:
        analyze_dataset(args, args['sel_ds'],
                        axs=None)
