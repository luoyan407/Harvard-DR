import numpy as np
import os 
import cv2
import random
import sys 
import csv
import pandas as pd
import bz2
from bz2 import BZ2File
from datetime import datetime
from collections import Counter
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

# sshfs yl535@erisxdl.partners.org:/data/elzelab ~/elzelab
# sshfs yl535@10.100.137.40:/shared/hdds_20T/cirrus_from_dicom_merged ~/ws_2gpu/cirrus_from_dicom_merged

newscanfolder = '/shared/hdds_20T/cirrus_from_dicom/scans'
newscanfolder_p2 = '/shared/hdds_20T/cirrus_from_dicom_part2/scans'

# vffile = '/shared/ssd_16T/yl535/project/python/datasets/baseline.vf.csv'
# octfile = '/shared/ssd_16T/yl535/project/python/datasets/oct.table.with.six.progression.outcomes.csv'

vffile = '/shared/ssd_16T/meecs/progression_data/old_data/baseline.vf.with.time.csv'
octfile = '/shared/ssd_16T/meecs/progression_data/old_data/oct.table.with.six.progression.outcomes.with.time.csv'

# meta_octfile = '/shared/hdds_20T/cirrus_from_dicom_merged/metadata_OpticDiscCube_cleaned.tsv'

# save_folder = '/shared/ssd_16T/yl535/project/python/datasets/havo_longitudinal_new_export/longitudinal_v3'

# clockhours_labels_npz = './results/sanity_check.npz'
clockhours_labels_npz = './results/reconstructed_sanity_check.npz'

if not os.path.exists(clockhours_labels_npz):
    
    vfmetadata = csv.reader(open(vffile))
    vfheader = next(vfmetadata)
    tdid_timetd = {}
    k = 0
    for row in vfmetadata:
        pid = row[0] + '_' + str(k)
        tds = []
        for i in range(54):
            if i == 25 or i == 34:
                continue
            tds.append(float(row[vfheader.index('td'+str(i+1))]))
        md = row[vfheader.index('md')]
        tdid_timetd[pid] = [tds, md]
        k += 1
        
    dict_octeye_time = {}
    octmetadata = csv.reader(open(octfile))
    octheader = next(octmetadata)
    # count = 0
    ilms_all = []
    ilm2nds_all = []
    maps_all = []
    tds_all = []
    labels_all = []
    clockhours_all = []
    avgT_all = []
    pids = []
    k = 0
    ncount = 0
    for row in octmetadata:
        pid = row[octheader.index('id')]
        datadir = row[octheader.index('datadir')]
        righteye = row[octheader.index('righteye')]
        timeoftest = row[octheader.index('timeoftest')]
        if row[octheader.index('avgthickness')] != 'NA':
            avgthickness = float(row[octheader.index('avgthickness')])
        else:
            avgthickness = -1
        signalstrength = 0.
        if row[octheader.index('signalstrength')] != 'NA':
            signalstrength = float(row[octheader.index('signalstrength')])
        if signalstrength >= 6:
            ncount = ncount + 1
        
        progression_outcome_md = row[octheader.index('progression.outcome.md')]
        progression_outcome_vfi = row[octheader.index('progression.outcome.vfi')]
        progression_outcome_td_pointwise = row[octheader.index('progression.outcome.td.pointwise')]
        progression_outcome_md_fast = row[octheader.index('progression.outcome.md.fast')]
        progression_outcome_md_fast_no_p_cut = row[octheader.index('progression.outcome.md.fast.no.p.cut')]
        progression_outcome_td_pointwise_no_p_cut = row[octheader.index('progression.outcome.td.pointwise.no.p.cut')]
        label = [progression_outcome_md, progression_outcome_vfi, progression_outcome_td_pointwise,
                progression_outcome_md_fast, progression_outcome_md_fast_no_p_cut, progression_outcome_td_pointwise_no_p_cut]
        tds, md = tdid_timetd[pid + '_' + str(k)]

        try:

            clockhours = []
            for i in range(1,13):
                clock_v = float(row[octheader.index(f'clockhour{i}')])
                clockhours.append(clock_v)
            clockhours = np.array(clockhours).astype(float)

            octpath = newscanfolder
            if not os.path.exists(os.path.join(octpath, datadir)): 
                octpath = newscanfolder_p2
            
            # disc
            data_disk = open(os.path.join(octpath, datadir, 'mask_disc.csv'))
            img_disk = np.reshape([float(row) for row in data_disk], (200, -1))
            # cup
            data_cup = open(os.path.join(octpath, datadir, 'mask_cup.csv'))
            img_cup = np.reshape([float(row) for row in data_cup], (200, -1))
            # ilm
            data_ilm2nd = open(os.path.join(octpath, datadir, 'segmentation_ilm_2nd.csv'))
            img_ilm2nd = np.reshape([float(row) for row in data_ilm2nd], (200, -1))
            data_ilm = open(os.path.join(octpath, datadir, 'segmentation_ilm.csv'))
            img_ilm = np.reshape([float(row) for row in data_ilm], (200, -1))
            # gcl
            data_gcl = open(os.path.join(octpath, datadir, 'segmentation_rnfl_to_gcl.csv'))
            img_gcl = np.reshape([float(row) for row in data_gcl], (200, -1))
            # rnflt map
            rnflt_map = (img_gcl - img_ilm2nd) * 0.00195503 * 1000 # bscanPixelspacingDepth * 1000
            rnflt_map[img_disk==1] = -1
            rnflt_map[img_cup==1] = -2
            img_gcl = img_gcl * 0.00195503 * 1000
            img_ilm2nd = np.abs(img_ilm2nd-np.max(img_ilm2nd)) * 0.00195503 * 1000
            img_ilm = np.abs(img_ilm-np.max(img_ilm)) * 0.00195503 * 1000

            if int(righteye) == 0:
                rnflt_map = np.fliplr(rnflt_map)
                img_ilm2nd = np.fliplr(img_ilm2nd)
                img_ilm = np.fliplr(img_ilm)
            
            octID = pid + '_' + righteye + '_' + timeoftest
            
            if signalstrength >= 6:
                ilms_all.append(img_ilm)
                ilm2nds_all.append(img_ilm2nd)
                maps_all.append(rnflt_map)
                labels_all.append(label)
                tds_all.append([tds, md])
                pids.append([octID, datadir])
                avgT_all.append(avgthickness)
                clockhours_all.append(clockhours)

                octeye_id = pid + '_' + righteye
                if octeye_id not in dict_octeye_time:
                    dict_octeye_time[octeye_id] = list()
                dict_octeye_time[octeye_id].append(timeoftest)
        except Exception as e:
            print(e)
            
        k += 1

    print(f'total number: {ncount}')

    samples_to_save = []
    for i, (k,v) in enumerate(dict_octeye_time.items()):
        if len(v) > 0:
            v.sort(reverse=True)
            # print(v)
            # datetime.strptime(timestr, '%y%m%d%H%M')
            samples_to_save.append(f'{k}_{v[0]}')
    new_clockhours_all = []
    new_labels_all = []
    for i in range(len(maps_all)):
        ilm = ilms_all[i]
        ilm2nd = ilm2nds_all[i]
        rnflt = maps_all[i]
        tds, md = list(tds_all[i])
        labels = labels_all[i]
        octID, datadir = pids[i]
        avgthickness = avgT_all[i]
        clockhours = clockhours_all[i]
        
        if octID not in samples_to_save:
            continue
        new_clockhours_all.append(clockhours)
        new_labels_all.append(labels)

    clockhours_all = np.vstack(new_clockhours_all)
    labels_all = np.array(new_labels_all)
    np.savez(clockhours_labels_npz, clockhours=clockhours_all, labels=labels_all)
else:
    raw_data = np.load(clockhours_labels_npz)
    clockhours_all = raw_data['clockhours']
    labels_all = raw_data['labels']

labels_all = labels_all.astype(float)

split_seed=338
n_splits = 3
cv = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)

for j in range(labels_all.shape[1]):
    progression_labels = labels_all[:, j]

    avg_auc = 0
    preds = []
    gts = []
    for i, (idx_train, idx_test) in enumerate(cv.split(progression_labels)):
        train_data = clockhours_all[idx_train, :]
        train_label = progression_labels[idx_train]
        test_data = clockhours_all[idx_test, :]
        test_label = progression_labels[idx_test]
        
        label_samples = {}
        possible_labels = np.unique(train_label)
        balanced_max = 0
        for x in possible_labels:
            # if x not in label_samples:
            #     label_samples[x] = list()
            label_samples[x] = train_data[train_label==x, :]
            balanced_max = len(label_samples[x]) \
                if label_samples[x].shape[0] > balanced_max else balanced_max
            
        all_balanced_samples = []
        all_balanced_labels = []
        for i, (k,v) in enumerate(label_samples.items()):
            # print(f'{k}-th class training samples: {v.shape[0]}')
            if v.shape[0] != balanced_max:
                tm_samples = v.tolist()
                while len(tm_samples) < balanced_max:
                    tm_samples.append(random.choice(tm_samples))
                all_balanced_labels += [k]*balanced_max
                all_balanced_samples += tm_samples
            else:
                all_balanced_samples += v.tolist()
                all_balanced_labels += [k]*v.shape[0]
        train_data = np.array(all_balanced_samples)
        train_label = np.array(all_balanced_labels)

        regressor = LogisticRegression(random_state=0).fit(train_data, train_label)
        preds.append(regressor.predict_proba(test_data)[:,1])
        gts.append(test_label)
        # fpr, tpr, thresholds = roc_curve(test_label, regressor.predict_proba(test_data)[:,1])
        # val_AUC = auc(fpr, tpr)
        # avg_auc += val_AUC
        break
    # avg_auc /= n_splits
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    fpr, tpr, thresholds = roc_curve(gts, preds)
    val_AUC = auc(fpr, tpr)
    print(f'============> {j}-th progression type AUC: {val_AUC:.4f}')


# octid_all = []
# for x in pids:
#     octid_all.append(x[0])
# octid_from_meta = list(octid_to_avgthickness.keys())
# overlap_items = list(set(octid_all) & set(octid_from_meta))
# # overlap_items = list((Counter(octid_all) & Counter(octid_from_meta)).elements())
# print(f'overlap number: {len(overlap_items)}')

# isExist = os.path.exists(save_folder)
# if not isExist:
#    # Create a new directory because it does not exist
#    os.makedirs(save_folder)

# eval_progression_labels = [[], [], [], [], [], []]
# eval_avgthickness = []
# count = 1
# for i in range(len(maps_all)):
#     ilm = ilms_all[i]
#     ilm2nd = ilm2nds_all[i]
#     rnflt = maps_all[i]
#     tds, md = list(tds_all[i])
#     labels = labels_all[i]
#     octID, datadir = pids[i]
#     avgthickness = avgT_all[i]
#     clockhours = clockhours_all[i]
    
# #     artifact_ratio = format((225*225 - np.count_nonzero(rnflt)) / float(225*225), '.3f')
    
#     np.savez(os.path.join(save_folder, f"data_{count:05d}.npz"),
#                  rnflt=rnflt, 
#                  ilm=ilm,
#                  ilm2nd=ilm2nd,
#                  progression=labels,
#                  md=md,
#                  tds=tds,
#                  pid=octID,
#                  clockhours=clockhours,
#                  avgthickness=avgthickness,
#                  datadir=datadir)
             
#     count += 1

#     if avgthickness >= 0:
#         for j in range(len(labels)):
#             eval_progression_labels[j].append(float(labels[j]))
#         eval_avgthickness.append(avgthickness)

# # np.savez(os.path.join(save_folder, f"check.npz"),
# #                  progression_labels=np.array(eval_progression_labels), avgthickness=np.array(eval_avgthickness))

# for i in range(len(eval_progression_labels)):
#     fpr, tpr, thresholds = roc_curve(eval_progression_labels[i], eval_avgthickness)
#     val_AUC = auc(fpr, tpr)
#     r = np.corrcoef(eval_progression_labels[i], eval_avgthickness)
#     print(f'{i}-th progression outcome AUC: {val_AUC:.4f}, Pearson correlation: {r[0,1]:.4f}')
