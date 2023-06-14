import sys, os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Times New Roman'], 'size':12})
rc('text', usetex=True)

def sigmoid(x):
	return 1/(1+np.exp(-x))

## refer to https://stackoverflow.com/questions/52908925/add-a-standard-normal-pdf-over-a-seaborn-histogram
show_legend = False
x_range = [0,1]
y_loc = 50.0
pos_db = 0.5
neg_db = 0.5
reversOrder = True

# input_npz = 'results/crosssectional_rnflt_race/fullysup_efficientnet_rnflt_Taskcls_lr5e-5_bz6_beta-1_auc0.8438/pred_gt_ep004.npz'
# output_svg = 'data4paper/hist_rnflt_race.svg'
# show_legend = True
# y_loc = 50.0

input_npz = 'results/crosssectional_rnflt_bn_race/fullysup_efficientnet_rnflt_Taskcls_lr5e-5_bz6_tw100_agnmom_auc0.8401/pred_gt_ep004.npz'
output_svg = 'data4paper/hist_rnflt_race_bn.svg'
y_loc = 30.0

# input_npz = 'results/crosssectional_rnflt_lbn_race/fullysup_efficientnet_rnflt_Taskcls_lr5e-5_bz6_tw100_agnmom_auc0.8433/pred_gt_ep002.npz'
# output_svg = 'data4paper/hist_rnflt_race_lbn.svg'
# y_loc = 50.0

# input_npz = 'results/crosssectional_rnflt_fin_race/fullysup_efficientnet_rnflt_Taskcls_lr5e-5_bz6_4702_auc0.8646/pred_gt_ep008.npz'
# output_svg = 'data4paper/hist_rnflt_race_fin.svg'
# # y_loc = 50.0


npzfile = np.load(input_npz)
print(npzfile.files)
# ['oracle_pred', 'oracle_pred_label', 'oracle_gt', 'stats_dist']

# feat = npzfile['oracle_pred']
# pred_label = npzfile['oracle_pred_label']
# gt = npzfile['oracle_gt']

gt = npzfile['val_gt']
x_pos = npzfile['val_pred']
x_neg = x_pos[gt==False]

# x_pos = sigmoid(feat[gt])
# x_neg = sigmoid(feat[gt == False])

print(f'pos num: {x_pos.shape[0]:.4f}, neg num: {x_neg.shape[0]:.4f}')

x_tp = x_pos[x_pos>pos_db]
x_fn = x_pos[x_pos<=pos_db]
x_tn = x_neg[x_neg<neg_db]
x_fp = x_neg[x_neg>=neg_db]

val = 0. # this is the value where you want the data to appear on the y-axis.

plt.figure(figsize=(12,7))
ax = plt.subplot()

# plt.plot(x_pos, np.zeros_like(x_pos) + val, 'o')
# plt.plot(x_neg, np.zeros_like(x_neg) + val, 'x')

# sns.set(font_scale=2)
if x_range is not None:
	plt.xlim(x_range)

bins = np.linspace(x_range[0], x_range[1], 100)

if reversOrder:
	plt.hist(x_tp, bins, alpha=1, label='TP', color='red')
	plt.hist(x_fp, bins, alpha=1, label='FP', color='m')
	plt.hist(x_fn, bins, alpha=1, label='FN', color='c')
	plt.hist(x_tn, bins, alpha=1, label='TN', color='blue')
else:
	plt.hist(x_tn, bins, alpha=1, label='TN', color='blue')
	plt.hist(x_fn, bins, alpha=1, label='FN', color='c')
	plt.hist(x_fp, bins, alpha=1, label='FP', color='m')
	plt.hist(x_tp, bins, alpha=1, label='TP', color='red')
plt.axvline(neg_db, label='Neg/Pos Threshold', color='blue', linestyle='dashed', linewidth=3)
# plt.axvline(pos_db, label='Positive threshold', color='red', linestyle='dashed', linewidth=3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(axis='y', linestyle='--')

plt.ylabel('count', fontsize=45)
plt.xlabel(r'$p(y=1|x,f_{\theta})$', fontsize=45)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

loc = plticker.MultipleLocator(base=y_loc) # this locator puts ticks at regular intervals
ax.yaxis.set_major_locator(loc)
if show_legend:
	plt.legend(loc=0,fontsize=28)

plt.savefig(output_svg, bbox_inches='tight', dpi=300)
plt.show()