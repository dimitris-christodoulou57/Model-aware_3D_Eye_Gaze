################################################################################
#                           FORMATTING                    
################################################################################
import os

import pandas as pd 
import numpy as np

from matplotlib import pyplot as plt 

pgf_with_latex = {                      # setup matplotlib to use latex for output
    #"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        r"\usepackage[gensymb]"
        ])
    }
plt.rcParams.update(pgf_with_latex)

fig_fontsize = 22
axis_fontsize = fig_fontsize*0.8
legend_fontsize = fig_fontsize*0.6
marker_size=7

dpi = 600

plt.rcParams["figure.figsize"] = (9,3)

################################################################################
#                           LOADING DATA                    
################################################################################
data_perc = [
    0.0005,
    #0.00075,
    0.001,
    0.0025,
    0.004,
    0.005,
    0.007,
    0.009,
    0.011,
    0.013
]
    
data_N = [347112*x for x in data_perc]

###############################################################################################

scratch_G_3D = [
    [8.134, 5.305],
    #[7.933, 0.0],    # TODO <---------------------------------
    [7.628, 4.858],
    [6.063, 4.24],
    [5.966, 3.848], # G_perc_0.004_res_50_3_BF_4_4_Nang72_Nrad_8_osKbk_21_06_23_03_53_55
    [5.663, 3.713],
    [4.717, 3.457],
    [4.568, 3.181], 
    [3.624, 2.928],
    [2.618, 2.941],
]
scratch_G_3D = np.asarray(scratch_G_3D)
scratch_G_2D = [
    [64.016, 35.224], 
    #[63.984, 0.0],     # TODO <---------------------------------
    [62.115, 32.368],  
    [50.637, 26.561],
    [51.515, 23.589],
    [47.64, 21.802],
    [39.729, 19.199],
    [35.16, 17.313],
    [26.151, 15.359],
    [16.588, 15.908],
]
scratch_G_2D = np.asarray(scratch_G_2D)

###############################################################################################
    
trf_G_3D_1e3 = [
    [4.168, 5.184],
    #[4.199, 4.618],
    [4.129, 4.705],
    [2.758, 3.933],
    [2.597, 4.223], # G_trfT_perc_0.004_res_50_3_BF_4_4_Nang72_Nrad_8_ocXkD_21_06_23_04_09_19
    [2.747, 4.209],
    [2.577, 4.159],
    [2.468, 4.535],
    [2.129, 4.088],
    [1.528, 4.499],
]
trf_G_3D_1e3 = np.asarray(trf_G_3D_1e3)
trf_G_2D_1e3 = [
    [29.079, 33.638],
    #[28.438, 28.688],
    [27.632, 29.135],
    [19.461, 24.197],
    [18.027, 26.62],
    [18.05, 26.535],
    [17.569, 25.931],
    [16.451, 27.545],
    [14.206, 24.682],
    [9.505, 27.243],
]
trf_G_2D_1e3 = np.asarray(trf_G_2D_1e3)

###############################################################################################

trf_G_3D_2e4 = [
    [2.962, -1.0],
    #[3.164, -1.0],
    [2.995, -1.0],
    [2.257, -1.0],
    [2.075, -1.0], #G_trfT_2e-4_perc_0.004_res_50_3_BF_4_4_Nang72_Nrad_8_pzwwQ_21_06_23_10_45_20
    [2.306, -1.0],
    [2.198, -1.0],
    [2.206, -1.0],
    [2.129, -1.0], # TODO <----------------------- from wrong learning rate exp
    [1.626, -1.0],
]
trf_G_3D_2e4 = np.asarray(trf_G_3D_2e4)
trf_G_2D_2e4 = [
    [21.024, -1.0],
    #[23.145, -1.0],
    [21.879, -1.0],
    [15.361, -1.0],
    [13.763, -1.0],
    [15.28, -1.0],
    [14.63, -1.0],
    [14.763, -1.0],
    [14.206, -1.0], # TODO <----------------------- from wrong learning rate exp
    [9.879, -1.0],
]
trf_G_2D_2e4 = np.asarray(trf_G_2D_2e4)

###############################################################################################

trf_TG_3D_2e4 = [
    [3.19, -1.0],
    #[3.328, -1.0],
    [3.122, -1.0],
    [2.172, -1.0],
    [2.372, -1.0], #TG_trfT_2e-4_perc_0.004_res_50_3_BF_4_4_Nang72_Nrad_8_uRusH_21_06_23_10_39_38 # <------ TODO Crashed
    [2.475, -1.0],
    [2.11, -1.0],
    [2.497, -1.0],
    [3.209, -1.0],
    [2.944, -1.0],
]
trf_TG_3D_2e4 = np.asarray(trf_TG_3D_2e4)
trf_TG_2D_2e4 = [
    [22.723, -1.0],
    #[23.834, -1.0],
    [22.265, -1.0],
    [14.94, -1.0],
    [18.374, -1.0],
    [16.979, -1.0],
    [14.362, -1.0],
    [17.542, -1.0],
    [22.229, -1.0],
    [20.503, -1.0],
]
trf_TG_2D_2e4 = np.asarray(trf_TG_2D_2e4)

###############################################################################################

gaze_full_data_3D = [0.9617, 2.996]
gaze_full_data_2D = [6.926, 16.344]


################################################################################
#                           DRAWING   3D                 
################################################################################
x_min = -0.1
x_max = data_N[-1]*1.1
y_min = 0.0
y_max = 11.0
# variance plot 
# https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans

i_split = 0
split = 'val' if i_split==0 else 'test'

plt.plot(
    data_N,
    scratch_G_3D[:, i_split],
    color='darkred',
    marker='s',
    label='gaze loss',
    markersize=marker_size
)

plt.plot(
    data_N,
    trf_G_3D_2e4[:, i_split],
    color='blue',
    marker='o',
    label='+ sem. loss (pre-training)',
    markersize=marker_size
)

plt.plot(
    data_N,
    trf_TG_3D_2e4[:, i_split],
    color='green',
    marker='^',
    label='+ sem. loss (pre \& during)',
    markersize=marker_size
)

plt.plot(
    [x_min, x_max],
    [gaze_full_data_3D[i_split]]*2,
    color='red',
    linestyle='dashed',
    #marker='o',
    label='gaze loss only (350k labels)',
    markersize=marker_size
)


plt.xlabel('Number of 3D gaze labels', fontsize=fig_fontsize)
plt.ylabel('Angular error [$^{\circ}$] $\\downarrow$', fontsize=fig_fontsize)
plt.title(f'3D gaze error after few-shot learning', fontsize=fig_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
#plt.show()


plt.savefig('gaze_low_shot_3D.jpg', bbox_inches='tight', dpi=dpi)
plt.savefig('gaze_low_shot_3D.pdf', bbox_inches='tight', dpi=dpi)
plt.close()

################################################################################
#                           DRAWING  2D                  
################################################################################
x_min = -0.1
x_max = data_N[-1]*1.1
y_min = 0.0
y_max = 83.0
# variance plot 
# https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans

split = 'val' if i_split==0 else 'test'


plt.plot(
    data_N,
    scratch_G_2D[:, i_split],
    color='darkred',
    marker='s',
    label='gaze loss',
    markersize=marker_size
)

plt.plot(
    data_N,
    trf_G_2D_2e4[:, i_split],
    color='blue',
    marker='o',
    label='+ sem. loss (pre-training)',
    markersize=marker_size
)

plt.plot(
    data_N,
    trf_TG_2D_2e4[:, i_split],
    color='green',
    marker='^',
    label='+ sem. loss (pre \& during)',
    markersize=marker_size
)

plt.plot(
    [x_min, x_max],
    [gaze_full_data_2D[i_split]]*2,
    color='red',
    linestyle='dashed',
    #marker='o',
    label='gaze loss only (350k labels)',
    markersize=marker_size
)


plt.xlabel('Number of 3D gaze labels', fontsize=fig_fontsize)
plt.ylabel('Angular error [$^{\circ}$] $\\downarrow$', fontsize=fig_fontsize)
plt.title(f'2D gaze error after few-shot learning', fontsize=fig_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
#plt.show()


plt.savefig('gaze_low_shot_2D.jpg', bbox_inches='tight', dpi=dpi)
plt.savefig('gaze_low_shot_2D.pdf', bbox_inches='tight', dpi=dpi)
plt.close()