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

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = pred_prob.detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y, pred_prob)
    AUC = auc(fpr, tpr)
    
    return AUC

def round_down_float(a_float):
	return floor(a_float*100)*0.01

input_f1 = 'results_3fold/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.md_Taskcls_lr5e-5_bz6_beta-1/all_pred_gt_fold0.npz'
input_f2 = 'results_with_race/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.md_Taskcls_lr5e-5_bz6_beta0.9999/model_fold0_ep008_eval.npz'
output_f = 'roc_md'
title_name = '(a) MD Progression'

# input_f1 = 'results_3fold/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.vfi_Taskcls_lr5e-5_bz6_beta-1/all_pred_gt_fold0.npz'
# input_f2 = 'results_3fold/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.vfi_Taskcls_lr5e-5_bz6_beta0.9999/all_pred_gt_fold0.npz'
# output_f = 'roc_vfi'
# title_name = '(b) VFI Progression'

# input_f1 = 'results_3fold/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.td.pointwise_Taskcls_lr5e-5_bz6_beta-1/all_pred_gt_fold0.npz'
# input_f2 = 'results_3fold/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.td.pointwise_Taskcls_lr5e-5_bz6_beta0.9999/all_pred_gt_fold0.npz'
# output_f = 'roc_tds'
# title_name = '(c) TD Pointwise Progression'

# input_f1 = 'results/Longitudinal_rnflt_bk/fullysup_efficientnet_rnflt_progression.outcome.md.fast_Taskcls_lr5e-5_bz6_beta-1/all_pred_gt_fold0.npz'
# input_f2 = 'results_with_race/Longitudinal_rnflt/fullysup_efficientnet_rnflt_progression.outcome.md.fast_Taskcls_lr5e-5_bz6_beta0.9999/model_fold0_ep004_eval.npz'
# output_f = 'roc_mdfast'
# title_name = '(d) MD Fast Progression'

races = ['Asian', 'Black', 'White']
mycolors = ['blue', 'red', 'black']

expr_name = 'train_predictor_auc'
step = 20

plot_step = -1
showLegend = True

npfile = np.load(input_f1)
pred_1 = npfile['val_pred']
gt_1 = npfile['val_gt']
attr_1 = npfile['val_attr']

npfile = np.load(input_f2)
pred_2 = npfile['val_pred']
gt_2 = npfile['val_gt']
attr_2 = npfile['val_attr']

plt.figure(figsize=(12,5))
ax = plt.subplot()

auc_1, fpr_1, tpr_1 = compute_auc(gt_1, pred_1)
auc_2, fpr_2, tpr_2 = compute_auc(gt_2, pred_2)

fprs_1 = []
tprs_1 = []
fprs_2 = []
tprs_2 = []
labels_1 = []
labels_2 = []
for i, one_attr in enumerate(np.unique(attr_1).astype(int)):
    auc_1, fpr_1, tpr_1 = compute_auc(gt_1[attr_1 == one_attr], pred_1[attr_1 == one_attr])
    fprs_1.append(fpr_1)
    tprs_1.append(tpr_1)
    labels_1.append(f'{races[i]} (AUC = {auc_1:.2f})')
for i, one_attr in enumerate(np.unique(attr_2).astype(int)):
    auc_2, fpr_2, tpr_2 = compute_auc(gt_2[attr_2 == one_attr], pred_2[attr_2 == one_attr])
    fprs_2.append(fpr_2)
    tprs_2.append(tpr_2)
    labels_2.append(f'{races[i]} (AUC = {auc_2:.2f})')

print(labels_1)
print(labels_2)

for i in range(len(races)):
	plt.plot(1-fprs_1[i], tprs_1[i] , c=mycolors[i],
			linewidth=2,
			linestyle='dashed',
	        label=labels_1[i])
	plt.plot(1-fprs_2[i], tprs_2[i] , c=mycolors[i],
			linewidth=2,
	        label=f'{labels_2[i]} with Fair Loss')

plt.title(title_name, fontsize=37, y=1.01)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
ax.invert_xaxis()
# ax.set_xlim(left=1.)
plt.xlabel("Specificity", fontsize=30)
plt.ylabel("Sensitivity", fontsize=30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
if showLegend:
    plt.legend(loc=0,fontsize=18)

plt.savefig('{}/{}.svg'.format('data4paper', output_f), bbox_inches='tight', dpi=300)
plt.show()