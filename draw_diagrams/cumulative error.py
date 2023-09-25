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
tresholds = np.linspace(0.1, 14.0, num=20)
csv_files = [
    '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/dimitrios_gaze/EVALUATION_G_perc_0.004_res_50_3_BF_4_4_Nang72_Nrad_8_osKbk_21_06_23_03_53_55/valid_raw_results.csv',
    '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/dimitrios_gaze/EVALUATION_G_trfT_2e-4_perc_0.004_res_50_3_BF_4_4_Nang72_Nrad_8_pzwwQ_21_06_23_10_45_20/valid_raw_results.csv',
    '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/dimitrios_gaze/EVALUATION_TG_trfT_2e-4_perc_0.004_res_50_3_BF_4_4_Nang72_Nrad_8_uRusH_21_06_23_10_39_38/valid_raw_results.csv',
]
labels = [
    'gaze loss',
    '+ sem. loss (pre-training)',
    '+ sem. loss (pre \& during)'
]
colors = [
    'darkred',
    'blue',
    'green',
    #'red'
]
markers = [
    's',
    'o',
    '^'
]

dataframes = [pd.read_csv(x) for x in csv_files]
gaze_errors = [x['gaze_3D_ang_deg'].values for x in dataframes]
num_points = [x.values.shape[0] for x in dataframes]

assert len(set(num_points)) == 1

perc_below_tresh = []
for g_err_i, n in zip(gaze_errors, num_points):
    perc_below_tresh.append([(g_err_i<=t).sum()/n*100.0 for t in tresholds])


a = 1
################################################################################
#                           DRAWING   3D                 
################################################################################
x_min = tresholds[0] - 0.1
x_max = tresholds[-1] + 0.1
y_min =-1.0
y_max = 101.0
# variance plot 
# https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans


for i, _ in enumerate(perc_below_tresh):

    plt.plot(
        tresholds,
        perc_below_tresh[i],
        color=colors[i],
        marker=markers[i],
        label=labels[i],
        markersize=marker_size
    )


plt.xlabel('Angular error treshold [$^{\circ}$]', fontsize=fig_fontsize)
plt.ylabel('Points below \\ treshold [\%]', fontsize=fig_fontsize)
plt.title(f'3D gaze error distribution (few-shot learning w. 0.4\% labels)', fontsize=fig_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
#plt.show()


plt.savefig('cum_err.jpg', bbox_inches='tight', dpi=dpi)
plt.savefig('cum_err.pdf', bbox_inches='tight', dpi=dpi)
plt.close()

