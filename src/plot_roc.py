import os
import sys
import numpy as np
import random
from sklearn.metrics import *
# import wandb

# import matplotx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

from math import floor


sys.path.append('.')

sns.set_theme()
sns.set_style("darkgrid")

# if you have installed Times New Roman on your machine, uncomment the following line
rc('font', **{'family':'serif','serif':['Times New Roman'], 'size':12})
rc('text', usetex=True)

def compute_auc(glau_gt, glau_pred):
	fpr, tpr, thresholds = roc_curve(glau_gt, glau_pred)
	auc_score = auc(fpr, tpr)
	return auc_score, fpr, tpr

def round_down_float(a_float):
	return floor(a_float*100)*0.01

input_f1 = 'results/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.md_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f2 = 'results/Longitudinal_ilm/fullysup_efficientnet_ilm_progression.outcome.md_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f3 = 'results/Longitudinal_rnflt+ilm/fullysup_efficientnet_rnflt+ilm_progression.outcome.md_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
output_f = 'roc_md'
title_name = '(a) MD Progression'

input_f1 = 'results/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.vfi_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f2 = 'results/Longitudinal_ilm/fullysup_efficientnet_ilm_progression.outcome.vfi_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f3 = 'results/Longitudinal_rnflt+ilm/fullysup_efficientnet_rnflt+ilm_progression.outcome.vfi_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
output_f = 'roc_vfi'
title_name = '(b) VFI Progression'

input_f1 = 'results/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.td.pointwise_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f2 = 'results/Longitudinal_ilm/fullysup_efficientnet_ilm_progression.outcome.td.pointwise_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f3 = 'results/Longitudinal_rnflt+ilm/fullysup_efficientnet_rnflt+ilm_progression.outcome.td.pointwise_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
output_f = 'roc_tds'
title_name = '(c) TD Pointwise Progression'

input_f1 = 'results/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.md.fast_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f2 = 'results/Longitudinal_ilm/fullysup_efficientnet_ilm_progression.outcome.md.fast_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
input_f3 = 'results/Longitudinal_rnflt+ilm/fullysup_efficientnet_rnflt+ilm_progression.outcome.md.fast_Taskcls_lr5e-5_bz6/all_pred_gt_fold0.npz'
output_f = 'roc_mdfast'
title_name = '(d) MD Fast Progression'

expr_name = 'train_predictor_auc'
step = 20

plot_step = -1
showLegend = True


npfile = np.load(input_f1)
pred_1 = npfile['val_pred']
gt_1 = npfile['val_gt']

npfile = np.load(input_f2)
pred_2 = npfile['val_pred']
gt_2 = npfile['val_gt']

npfile = np.load(input_f3)
pred_3 = npfile['val_pred']
gt_3 = npfile['val_gt']

plt.figure(figsize=(12,5))
ax = plt.subplot()


auc_1, fpr_1, tpr_1 = compute_auc(gt_1, pred_1)
auc_2, fpr_2, tpr_2 = compute_auc(gt_2, pred_2)
auc_3, fpr_3, tpr_3 = compute_auc(gt_3, pred_3)

labels = [f'RNFLT (AUC={round_down_float(auc_1):.2f})', f'ILM (AUC={round_down_float(auc_2):.2f})', f'RNFLT+ILM (AUC={round_down_float(auc_3):.2f})']

print(f'auc 1: {auc_1:.2f}, auc 2: {auc_2:.2f}, auc 3: {auc_3:.2f}')

plt.plot(1-fpr_1, tpr_1 , c='black',
		linewidth=2,
        label=labels[0])
plt.plot(1-fpr_2, tpr_2, c='blue',
		linewidth=2,
        label=labels[1])
plt.plot(1-fpr_3, tpr_3, c='red',
		linewidth=2,
        label=labels[2])

plt.title(title_name, fontsize=37)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
ax.invert_xaxis()
plt.xlabel("Specificity", fontsize=30)
plt.ylabel("Sensitivity", fontsize=30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
if showLegend:
    plt.legend(loc=0,fontsize=25)

plt.savefig('{}/{}.svg'.format('data4paper', output_f), bbox_inches='tight', dpi=300)
plt.show()