import os
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import umap

class_names = ['Asian', 'Black', 'White']
color_map = ['blue', 'red', 'purple']

# ----------------------------
class_names = ['Asian', 'Black', 'White']

input_folder = 'results_analysis/dr_fundus_race_baseline/fullysup_efficientnet_fundus_Taskcls_lr5e-5_bz18_beta-1_split1_balancefalse_7133_auc0.9776'
file_npz = 'pred_gt_ep008.npz'
output_file = 'data4paper/umap_race_effnet.pdf'
fids = [44.76]

# ----------------------------
class_names = ['Female', 'Male']

input_folder = 'results_analysis/dr_fundus_gender_baseline/fullysup_efficientnet_fundus_Taskcls_lr5e-5_bz18_beta-1_split1_balancefalse_4129_auc0.9746'
file_npz = 'pred_gt_ep005.npz'
output_file = 'data4paper/umap_gender_effnet.pdf'
fids = [13.02]

# ----------------------------
class_names = ['Non-Hispanic', 'Hispanic']

input_folder = 'results_analysis/dr_fundus_hispanic_baseline/fullysup_efficientnet_fundus_Taskcls_lr5e-5_bz18_beta-1_split1_balancefalse_2037_auc0.9761'
file_npz = 'pred_gt_ep009.npz'
output_file = 'data4paper/umap_hispanic_effnet.pdf'
fids = [47.32]

# ----------------------------


raw_data = np.load(os.path.join(input_folder, file_npz), allow_pickle=True)
feat_in_classes = raw_data['val_feat'].tolist()

plt.figure(figsize=(6.5,6))
ax = plt.subplot()

centroids_classes = []
for i in range(len(feat_in_classes)):
    feat = feat_in_classes[i]

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feat)
    centroids = np.mean(embedding, axis=1)
    centroids_classes.append(centroids)

    plt.scatter(embedding[:, 0], embedding[:, 1], color=color_map[i], label=f'{class_names[i]}')

plt.legend(fontsize=12, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, frameon=True)

plt.savefig(os.path.join(output_file), bbox_inches='tight', dpi=300)
plt.show()