#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:51:33 2021

@author: rakshit
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import multiprocessing as mp

# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from MulticoreTSNE import MulticoreTSNE as TSNE

# path_exp = '/home/rsk3900/sporc/Results/all_vs_one/GR-1.2_AUG-1_ADV_DG-0_IN_NORM-1/GR-1.2_all_vs_one_AUG-1_IN_NORM-1_ADV_DG-0_PSEUDO_LABELS-0_20_06_21_15_53_51'

path_exp = '/home/rsk3900/sporc/Results/all_vs_one/GR-1.2_AUG-0_ADV_DG-0_IN_NORM-1/GR-1.2_all_vs_one_AUG-0_IN_NORM-1_ADV_DG-0_13_06_21_15_40_17'

# path_exp = '/home/rsk3900/sporc/Results/all_vs_one/GR-1.2_AUG-0_ADV_DG-1_IN_NORM-1_PSEUDO_LABELS-1/GR-1.2_all_vs_one_AUG-0_IN_NORM-1_ADV_DG-1_PSEUDO_LABELS-1_16_06_21_14_27_54'

# path_exp = '/home/rsk3900/PL3090/results/all_vs_one/GR-1.2_AUG-1_ADV_DG-1_ADA_IN_NORM-1_PSEUDO_LABELS-1/all_vs_one_GR-1.2_AUG-1_ADV_DG-1_ADA_IN_NORM-1_PSEUDO_LABELS-1_25_06_21_22_36_15'

# path_exp = '/home/rsk3900/sporc/Results/all_vs_one/GR-1.2_AUG-1_ADV_DG-0_ADA_IN_NORM-1_PSEUDO_LABELS-0/GR-1.2_all_vs_one_AUG-1_ADA_IN_NORM-1_ADV_DG-0_PSEUDO_LABELS-0_01_07_21_09_33_34'

# path_exp = '/home/rsk3900/sporc/Results/all_vs_one/GR-1.2_AUG-1_ADV_DG-0_IN_NORM-1_PSEUDO_LABELS-1/GR-1.2_all_vs_one_AUG-1_ADA_IN_NORM-0_ADV_DG-0_PSEUDO_LABELS-1_30_06_21_09_51_21'

epoch = 60

path_embed = os.path.join(path_exp, 'logs', '{:05}'.format(epoch), 'train')
path_tensors = os.path.join(path_embed, 'tensors.tsv')
path_meta = os.path.join(path_embed, 'metadata.tsv')

embeds = pd.read_table(path_tensors).to_numpy()
labels = pd.read_table(path_meta).to_numpy()

# Normalize embeds by scale
# embeds = embeds/np.linalg.norm(embeds, axis=1, keepdims=True)
# embeds_dst = cdist(embeds, embeds, metric='cosine')

pca = PCA(n_components=64)
pca_embeds = pca.fit_transform(embeds)
# print(pca.explained_variance_ratio_)

def get_int(str_ele):
    return int(re.search(r'\d+', str_ele).group())

# Get each item from element
labels = np.array([get_int(ele.item()) for ele in labels])

params = {'init':'random',
          'n_jobs': 64,
          'n_iter': 500,
          'perplexity': 40,
          'random_state': 0,
          'n_components': 2,
          'learning_rate': 10}

tsne_embeds = TSNE(**params).fit_transform(pca_embeds)

ds_names = ['LPW', 'Fuhl', 'NVGaze', 'Swirski', 'OpenEDS', 'BAT',
            'UnityEyes', 'RITEyes-gen', 'RITEyes-nat']

viridis = cm.get_cmap('brg', 9)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

for ds_idx in range(9):

    loc = labels == ds_idx

    color = np.array(viridis(ds_idx/9))
    color[-1] = 0.5  # Scale the Alpha component

    color = [color]*np.sum(loc)

    ax.scatter(tsne_embeds[loc, 0],
               tsne_embeds[loc, 1],
               c=np.stack(color, axis=0))

ax.legend(ds_names)
ax.grid()
fig.savefig(os.path.split(path_exp)[-1] + '.png',
            dpi=600,
            transparent=True,
            bbox_inches='tight')