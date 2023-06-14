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

newscanfolder = '/shared/hdds_20T/cirrus_from_dicom/scans'
newscanfolder_p2 = '/shared/hdds_20T/cirrus_from_dicom_part2/scans'

vffile = "/shared/hdds_20T/yl535/elzelab/hfa_ongoing/hfa_ongoing_merged_24-2_subset_cleaned_all_in.csv"
newoctfile = '/shared/hdds_20T/cirrus_from_dicom_merged/metadata_OpticDiscCube_cleaned.tsv'

save_folder = '/shared/ssd_16T/yl535/project/python/datasets/havo_crosssectional_new_export/crosssectional_v1'


def nearst_match(target, candidate_mds):
    curr_diff = float('inf')
    for curr in candidate_mds:
        timestr = curr[1].strip()
        if len(target) != 10:
            target = '0'*(10-len(target))+target
        if len(timestr) != 10:
            timestr = '0'*(10-len(timestr))+timestr
        
        target_time = datetime.strptime(target, '%y%m%d%H%M')
        curr_time = datetime.strptime(timestr, '%y%m%d%H%M')
        diff = curr_time - target_time
        diff = abs(diff.days*24*3600+diff.seconds+diff.microseconds/1e6)
        if diff < curr_diff:
            curr_diff = diff
            return_curr = curr
    return return_curr

vfmetadata = csv.reader(open(vffile))
vfheader = next(vfmetadata)
vfid_timetd = {}
for row in vfmetadata:
    pid = row[0]
    righteye = row[vfheader.index('righteye')]
    timeoftest = row[vfheader.index('timeoftest')].strip()
    age = row[vfheader.index('age')]
    male = row[vfheader.index('male')]
    md = row[vfheader.index('md')]
    ght = row[vfheader.index('ght')]
    vfi = row[vfheader.index('vfi')]
    mdprob = row[30]
    psd = row[vfheader.index('psd')]
    psdprob = row[vfheader.index('psdprob')]
    p30 = float(row[vfheader.index('p30')])

    malfixrate = row[vfheader.index('malfixrate')]
    falsenegrate = row[vfheader.index('falsenegrate')]
    falseposrate = row[vfheader.index('falseposrate')]
    
    if falsenegrate!='NA' and malfixrate!='NA' and falseposrate!='NA':
        malfixrate, falsenegrate, falseposrate = float(malfixrate), float(falsenegrate), float(falseposrate)
        if (p30==0) and (malfixrate <= 0.33) and (falsenegrate) <= 0.2 and (falseposrate) <= 0.2:
            pass
        else:
            continue
    else:
        continue

    tds = [float(row[i+87]) for i in range(52)]

    cid = pid + '_' + righteye
    if not cid in vfid_timetd.keys():
        vfid_timetd[cid] = [(float(md), timeoftest, float(ght), float(psdprob), vfi, float(mdprob), psd, age, tds)]
    else:
        vfid_timetd[cid].append((float(md), timeoftest, float(ght), float(psdprob), vfi, float(mdprob), psd, age, tds))
		
		
octmetanew = pd.read_csv(newoctfile, sep='\t')
existed_paths = []
for octpath in octmetanew['datadir']:
    if os.path.exists(os.path.join(newscanfolder, octpath)):
        existed_paths.append(octpath)

octmetanew_exist = octmetanew[octmetanew['datadir'].isin(existed_paths)]

clockhours_all = []
count = 0
unmatched = []
ilm_maps = {}
rnflt_maps = {}
rnflt_attrs = {}
upids = []
for index, row in octmetanew_exist.iterrows():
    pid = str(row[0])
    male = str(row['male'])
    signalstrength = float(row['signalstrength'])
    righteye = str(row['righteye'])
    timeoftest = str(row['timeoftest']).strip()
#     bscanPixelspacingDepth = float(row['bscanPixelspacingDepth'])
    datadir = row['datadir']
    
    #cid = pid + '_' + righteye + '_' + male 
    cid = pid + '_' + righteye
    octID = pid + '_' + righteye + '_' + timeoftest
    
    if cid not in vfid_timetd.keys():
        unmatched.append(pid)
        continue
    try:

        clockhours = []
        for i in range(1,13):
            clock_v = float(row[f'clockhour{i}'])
            clockhours.append(clock_v)
        clockhours = np.array(clockhours).astype(float)

        candidate_mds = vfid_timetd[cid]
        timeoftest = timeoftest[:-2]
        md, md_timeoftest, ght, psdprob, vfi, mdprob, psd, age, tds = nearst_match(timeoftest, candidate_mds)
        
        if len(md_timeoftest) != 10:
            md_timeoftest = '0'*(10-len(md_timeoftest))+md_timeoftest
        oct_testoftime = datetime.strptime(timeoftest, '%y%m%d%H%M')
        vf_testoftime = datetime.strptime(md_timeoftest, '%y%m%d%H%M')
        diff = vf_testoftime - oct_testoftime
        diff_month = diff.days/30

        glaucoma = 0
        if (diff_month<=1) and (diff_month>=-1) and (signalstrength>=6):
            if (md<-3) and (ght==3) and (psdprob>1):
                glaucoma = 1 # glaucoma
            elif (md>=-1) and (ght==1) and (psdprob<=1):
                glaucoma = 0 # non-glaucoma
            else:
                glaucoma = 2 # uncertain
        else:
            continue
        
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
        img_ilm = np.abs(img_ilm-np.max(img_ilm)) * 0.00195503 * 1000
        
        count += 1
        if pid not in rnflt_maps.keys():
            rnflt_maps[pid] = [rnflt_map]
            rnflt_attrs[pid] = [[glaucoma, md, tds, octID, datadir, clockhours]]
            ilm_maps[pid] = [img_ilm2nd]
            upids.append(pid)
        else:
            rnflt_maps[pid].append(rnflt_map)
            rnflt_attrs[pid].append([glaucoma, md, tds, octID, datadir, clockhours])
            ilm_maps[pid].append(img_ilm)
    except Exception as e:
        unmatched.append(pid)
        print(e)
    
    if count % 10000 == 0:
        print(count)
		
print(f'total num of samples: {count}')

save_cross_folder = os.path.join(save_folder, 'crosssectional')
isExist = os.path.exists(save_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(save_cross_folder)

idmaps = {}
count = 1
for pid in upids:
    for k in range(len(rnflt_maps[pid])):
        rnflt = rnflt_maps[pid][k]
        glaucoma, md, tds, octID, datadir, clockhours = rnflt_attrs[pid][k]
        ilm = ilm_maps[pid][k]
        np.savez(os.path.join(save_cross_folder,
                f"data_{count:06d}.npz"), 
                rnflt=rnflt,
                ilm=ilm,
                md=md,
                tds=tds,
                pid=octID,
                clockhours=clockhours,
                glaucoma=glaucoma,
                datadir=datadir)  
        idmaps[count] = octID
        count += 1

np.save(os.path.join(save_folder, 'idmaps_td_prediction'), idmaps)


