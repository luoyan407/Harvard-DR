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
from collections import Counter

newscanfolder = '/shared/hdds_20T/cirrus_from_dicom/scans'
newscanfolder_p2 = '/shared/hdds_20T/cirrus_from_dicom_part2/scans'

# vffile = '/shared/ssd_16T/yl535/project/python/datasets/baseline.vf.csv'
# octfile = '/shared/ssd_16T/yl535/project/python/datasets/oct.table.with.six.progression.outcomes.csv'

vffile = '/shared/ssd_16T/meecs/progression_data/old_data/baseline.vf.with.time.csv'
octfile = '/shared/ssd_16T/meecs/progression_data/old_data/oct.table.with.six.progression.outcomes.with.time.csv'

meta_octfile = '/shared/hdds_20T/cirrus_from_dicom_merged/metadata_OpticDiscCube_cleaned.tsv'
exclude_file = '/home/yanluo/server_ssd_project/python/datasets/crosssectional/test1000_pidmaps.npy'

# save_folder = '/shared/ssd_16T/yl535/project/python/datasets/havo_longitudinal_new_export/longitudinal_v3_race'
save_folder = '/shared/ssd_16T/yl535/project/python/datasets/havo_longitudinal_new_export/longitudinal_oneeye_md-1_timegap'

#-------start to process---------
# get exluded patient list if any
blacklist = []
if exclude_file != '':
    raw_data = np.load(exclude_file, allow_pickle=True)
    blacklist = [x[:x.find('_')] for x in list(raw_data.values())]

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

#====================
# octmetanew = pd.read_csv(meta_octfile, sep='\t')
# existed_paths = []
# for octpath in octmetanew['datadir']:
#     if os.path.exists(os.path.join(newscanfolder, octpath)):
#         existed_paths.append(octpath)

# octmetanew_exist = octmetanew[octmetanew['datadir'].isin(existed_paths)]

# octid_to_avgthickness = {}
# for index, row in octmetanew_exist.iterrows():
#     pid = str(row[0])
#     male = str(row['male'])
#     signalstrength = float(row['signalstrength'])
#     righteye = str(row['righteye'])
#     avgthickness = float(row['avgthickness'])
#     timeoftest = str(row['timeoftest']).strip()[:-2]
# #     bscanPixelspacingDepth = float(row['bscanPixelspacingDepth'])
#     datadir = row['datadir']
    
#     #cid = pid + '_' + righteye + '_' + male 
#     cid = pid + '_' + righteye
#     octID = pid + '_' + righteye + '_' + timeoftest
#     octid_to_avgthickness[octID] = avgthickness
# print(f'octID number: {len(list(octid_to_avgthickness.keys()))}')
#====================

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
race_all = []
timegaps_all = []
pids = []
k = 0
ncount = 0
for row in octmetadata:
    pid = row[octheader.index('id')]
    datadir = row[octheader.index('datadir')]
    righteye = row[octheader.index('righteye')]
    timeoftest = row[octheader.index('timeoftest')]
    time_gap_between_vf = float(row[octheader.index('time.between.baseline.oct.last.vf')])

    if row[octheader.index('avgthickness')] != 'NA':
        avgthickness = float(row[octheader.index('avgthickness')])
    else:
        avgthickness = -1
    signalstrength = 0.
    if row[octheader.index('signalstrength')] != 'NA':
        signalstrength = float(row[octheader.index('signalstrength')])
    if signalstrength >= 6:
        ncount = ncount + 1
    race = -1
    if row[octheader.index('race')] != 'NA':
        race = int(row[octheader.index('race')])
    
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
#         rnflt_bz = os.path.join(scan_folder, datadir, 'rnfl_thickness_map.csv.bz2')
#         rnflt_map = np.genfromtxt(BZ2File(rnflt_bz))
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
        
        if signalstrength >= 6 and float(md) < -1:
            ilms_all.append(img_ilm)
            ilm2nds_all.append(img_ilm2nd)
            maps_all.append(rnflt_map)
            labels_all.append(label)
            tds_all.append([tds, md])
            pids.append([octID, datadir])
            avgT_all.append(avgthickness)
            clockhours_all.append(clockhours)
            race_all.append(race)
            timegaps_all.append(time_gap_between_vf)

            octeye_id = pid + '_' + righteye
            if octeye_id not in dict_octeye_time:
                dict_octeye_time[octeye_id] = list()
            dict_octeye_time[octeye_id].append(timeoftest)
    except Exception as e:
        print(e)
        
    k += 1

print(f'total number: {ncount}, total eye number: {len(list(dict_octeye_time.keys()))}')

# examples: sorted '1801160947', '1801160946', '1702070949'
# datetime.datetime(2018, 1, 16, 9, 47), datetime.datetime(2018, 1, 16, 9, 46), datetime.datetime(2017, 2, 7, 9, 49)
samples_to_save = []
for i, (k,v) in enumerate(dict_octeye_time.items()):
    if len(v) > 0:
        v.sort(reverse=True)
        # print(v)
        # datetime.strptime(timestr, '%y%m%d%H%M')
        samples_to_save.append(f'{k}_{v[0]}')

# octid_all = []
# for x in pids:
#     octid_all.append(x[0])
# octid_from_meta = list(octid_to_avgthickness.keys())
# overlap_items = list(set(octid_all) & set(octid_from_meta))
# # overlap_items = list((Counter(octid_all) & Counter(octid_from_meta)).elements())
# print(f'overlap number: {len(overlap_items)}')

isExist = os.path.exists(save_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(save_folder)

selected_timegaps = []
eval_progression_labels = [[], [], [], [], [], []]
eval_avgthickness = []
count = 1
for i in range(len(maps_all)):
    ilm = ilms_all[i]
    ilm2nd = ilm2nds_all[i]
    rnflt = maps_all[i]
    tds, md = list(tds_all[i])
    labels = labels_all[i]
    octID, datadir = pids[i]
    avgthickness = avgT_all[i]
    clockhours = clockhours_all[i]
    race = race_all[i]
    time_gap_between_vf = timegaps_all[i]
    
    if octID not in samples_to_save:
        continue

#     artifact_ratio = format((225*225 - np.count_nonzero(rnflt)) / float(225*225), '.3f')
    
    np.savez(os.path.join(save_folder, f"data_{count:05d}.npz"),
                 rnflt=rnflt, 
                 ilm=ilm,
                 ilm2nd=ilm2nd,
                 progression=labels,
                 md=md,
                 tds=tds,
                 pid=octID,
                 avgthickness=avgthickness,
                 race=race,
                 clockhours=clockhours,
                 timegap=time_gap_between_vf,
                 datadir=datadir)
    
    count += 1

    selected_timegaps.append(time_gap_between_vf)
    if avgthickness >= 0:
        for j in range(len(labels)):
            eval_progression_labels[j].append(float(labels[j]))
        eval_avgthickness.append(avgthickness)

selected_timegaps = np.array(selected_timegaps).mean()
print(f'average time gap: {selected_timegaps:.4f}')

# np.savez(os.path.join(save_folder, f"check.npz"),
#                  progression_labels=np.array(eval_progression_labels), avgthickness=np.array(eval_avgthickness))
# print('statistics on the extracted data')
for i in range(len(eval_progression_labels)):
    fpr, tpr, thresholds = roc_curve(eval_progression_labels[i], eval_avgthickness)
    val_AUC = auc(fpr, tpr)
    r = np.corrcoef(eval_progression_labels[i], eval_avgthickness)
    print(f'{i}-th progression outcome AUC: {val_AUC:.4f}, Pearson correlation: {r[0,1]:.4f}')

race_set = Counter(race_all).keys()
race_occurrence = Counter(race_all).values()
print(f'# of all samples: {len(race_all)}')
for i, (i_race, i_occur) in enumerate(zip(race_set, race_occurrence)):
    print(f'race code {i} with {i_occur} occurrences')