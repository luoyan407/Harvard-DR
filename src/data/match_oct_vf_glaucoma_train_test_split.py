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

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def pprint(out_str, redirect_out=None, flag='a'):
    # print(out_str)
    if redirect_out is not None:
        with open(redirect_out, flag) as sys.stdout:
            print(out_str)

newscanfolder = '/shared/hdds_20T/yl535/elzelab/cirrus_ongoing/scans'
newscanfolder_p2 = '/shared/external_hdds/external20TB_1/cirrus_from_dicom/scans'

# need to mount eris data elzelab by
# sshfs yl535@erisxdl.partners.org:/data/elzelab ~/elzelab

vffile = "/shared/hdds_20T/yl535/elzelab/hfa_ongoing/hfa_ongoing_merged_24-2_subset_cleaned_all_in.csv"
newoctfile = '/shared/hdds_20T/yl535/elzelab/cirrus_ongoing/metadata_MacularCube_ongoing_cleaned.csv'
# exclude_file = '/shared/ssd_16T/yl535/project/python/datasets/crosssectional/test1000_pidmaps.npy'
exclude_file = ''

# save_folder = '/shared/ssd_16T/yl535/project/python/datasets/havo_crosssectional_new_export/crosssectional_fairness_all'
# save_folder = '/shared/ssd_16T/meecs/fairness/crosssectional_fairness_all'

# selected_patient_num = 2000
# selected_patient_num_test = 1000
ratio_of_patients_for_test = 0.3
num_attr = 3

save_folder = f'/shared/ssd_16T/yl535/project/python/datasets/havo_crosssectional_new_export/crosssectional_fairness_split{ratio_of_patients_for_test}'

seed = -1 #2740 # -1

log_output = save_folder+'.log'
# samples_per_attr = [int(selected_patient_num/num_attr), int(selected_patient_num/num_attr), selected_patient_num-2*int(selected_patient_num/num_attr)]
# samples_per_attr_test = [int(selected_patient_num_test/num_attr), int(selected_patient_num_test/num_attr), selected_patient_num_test-2*int(selected_patient_num_test/num_attr)]

# dict_race = {1:'American Indian or Alaska Native', 
#                 2:'Asian', 
#                 3:'Black or African American', 
#                 4:'Hispanic or Latino', 
#                 5:'Native Hawaiian or Other Pacific Islander', 
#                 6:'Other', 
#                 7:'White or Caucasian'}
dict_race = {2:'Asian', 
                3:'Black or African American', 
                7:'White or Caucasian'}

if seed < 0:
    seed = int(np.random.randint(10000, size=1)[0])
set_random_seed(seed)

pprint(f'====> seed: {seed}', log_output, 'w')

#-------start to process---------
# get exluded patient list if any
blacklist = []
if exclude_file != '':
    raw_data = np.load(exclude_file, allow_pickle=True)
    blacklist = [x[:x.find('_')] for x in list(raw_data.item().values())]


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
race_all = []
male_all = []
ilm_maps = {}
rnflt_maps = {}
rnflt_attrs = {}
upids = []
for index, row in octmetanew_exist.iterrows():
    pid = str(row[0])
    male = str(row['male'])
    if male not in ['0', '1']:
        continue
    male = int(male)
    signalstrength = float(row['signalstrength'])
    righteye = str(row['righteye'])
    timeoftest = str(row['timeoftest']).strip()
#     bscanPixelspacingDepth = float(row['bscanPixelspacingDepth'])
    datadir = row['datadir']
    race = row['race']
    hisp = row['hispanic']
    if (np.isnan(race)) or (int(race) not in dict_race) or (hisp not in ['0', '1']):
        continue
    race = dict_race[int(race)]
    hisp = int(hisp)

    # if (row['race']) != 'NA' and (not np.isnan(row['race'])):
    #     race = int(row['race'])
    
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
                continue
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
        
        race_all.append(race)
        male_all.append(male)

        count += 1
        if pid not in rnflt_maps.keys():
            rnflt_maps[pid] = [rnflt_map]
            rnflt_attrs[pid] = [[glaucoma, md, tds, octID, datadir, clockhours, race, male, hisp, row.to_dict()]]
            ilm_maps[pid] = [img_ilm2nd]
            upids.append(pid)
        else:
            rnflt_maps[pid].append(rnflt_map)
            rnflt_attrs[pid].append([glaucoma, md, tds, octID, datadir, clockhours, race, male, hisp, row.to_dict()])
            ilm_maps[pid].append(img_ilm)
    except Exception as e:
        unmatched.append(pid)
        pprint(e, log_output)
    
    # if (count>0) and (count % 10000 == 0):
    #     break

    if count % 10000 == 0:
        pprint(count, log_output)
		
pprint(f'total num of samples: {count}', log_output)
pprint(f'total num of patients: {len(upids)}', log_output)

possible_races, ppl_counts = np.unique(race_all, return_counts=True)
pprint('statistics on race:', log_output)
pprint(dict(zip(possible_races, ppl_counts)), log_output)

ppl_race_proportion = [x/sum(ppl_counts) for x in ppl_counts]
# samples_per_attr = [int(ppl_race_proportion[0]*selected_patient_num), int(ppl_race_proportion[1]*selected_patient_num), selected_patient_num-int(ppl_race_proportion[0]*selected_patient_num)-int(ppl_race_proportion[1]*selected_patient_num)]

# dict_race_smp = dict(zip(list(dict_race.values()), samples_per_attr))
# dict_race_smp_test = dict(zip(list(dict_race.values()), samples_per_attr_test))

tmp_upids = []
if len(blacklist) > 0:
    for x in upids:
        if x not in blacklist:
            tmp_upids.append(x)
    random.shuffle(tmp_upids)
    upids = tmp_upids
    # upids = tmp_upids[:selected_patient_num]

save_cross_folder = os.path.join(save_folder, 'train')
isExist = os.path.exists(save_cross_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(save_cross_folder)
# save_cross_folder_oct = os.path.join(save_folder, 'train', 'oct_bscan')
# isExist = os.path.exists(save_cross_folder_oct)
# if not isExist:
#    # Create a new directory because it does not exist
#    os.makedirs(save_cross_folder_oct)

save_cross_test_folder = os.path.join(save_folder, 'test')
isExist = os.path.exists(save_cross_test_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(save_cross_test_folder)
# save_cross_test_folder_oct = os.path.join(save_folder, 'test', 'oct_bscan')
# isExist = os.path.exists(save_cross_test_folder_oct)
# if not isExist:
#    # Create a new directory because it does not exist
#    os.makedirs(save_cross_test_folder_oct)

selected_mds = []
dict_race_mds = {'Asian':[], 
                'Black or African American':[], 
                'White or Caucasian':[]}

selected_data = []
selected_data_test = []

random.shuffle(upids)
idmaps = {}
count = 1
test_patients = int(ratio_of_patients_for_test*len(upids))

set_of_races = list(dict_race.values())
for pid in upids[:test_patients]:
    indices = list(range(len(rnflt_maps[pid])))
    random.shuffle(indices)

    tmp_save_cross_folder = save_cross_test_folder
    for ind in indices:
        rnflt = rnflt_maps[pid][ind]
        glaucoma, md, tds, octID, datadir, clockhours, race, male, hisp, raw_info_dict = rnflt_attrs[pid][ind]

        # if (race not in dict_race_smp) or (dict_race_smp[race]<=0):
        #     continue

        if (race in set_of_races) :
            # if (dict_race_smp[race]>0):
            #     dict_race_smp[race] = dict_race_smp[race] - 1
            #     tmp_save_cross_folder = save_cross_folder

            #     raw_info_dict['md'] = md
            #     raw_info_dict['glaucoma'] = glaucoma
            #     selected_data.append(raw_info_dict)
            # elif (dict_race_smp_test[race]>0):
            #     dict_race_smp_test[race] = dict_race_smp_test[race] - 1
            #     tmp_save_cross_folder = save_cross_test_folder

            # else:
            #     continue

            # tmp_save_cross_folder = save_cross_folder

            raw_info_dict['md'] = md
            raw_info_dict['glaucoma'] = glaucoma
            
            selected_data_test.append(raw_info_dict)

            # dict_race_smp[race] = dict_race_smp[race] - 1
            
            bscans = []
            for i in range(1, 201):
                bscan = cv2.imread(os.path.join(newscanfolder, datadir, f'bscan{i}.jp2'))
                bscan = bscan[:,:,0]
                bscan = cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE)
                bscan = cv2.resize(bscan, (200,200))
                # bscan = cv2.resize(cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE), (300,200))
                bscan = np.flip(bscan, axis=1)
                bscan = bscan[None,:,:]
                bscans.append(bscan)
            bscans = np.vstack(bscans)

            np.savez(os.path.join(tmp_save_cross_folder,
                    f"data_{count:06d}.npz"), 
                    rnflt=rnflt,
                    ilm=ilm,
                    oct_bscans=bscans,
                    clockhours=clockhours,
                    md=md,
                    tds=tds,
                    race=race,
                    male=male,
                    hispanic=hisp,
                    glaucoma=glaucoma)  
                    # clockhours=clockhours,
                    # pid=octID,
                    # datadir=datadir
            idmaps[count] = octID

            selected_mds.append(md)
            dict_race_mds[race].append(md)

            # bscans = []
            # for i in range(1, 201):
            #     bscan = cv2.imread(os.path.join(newscanfolder, datadir, f'bscan{i}.jp2'))
            #     bscan = bscan[:,:,0]
            #     bscan = cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE)
            #     bscan = cv2.resize(bscan, (200,200))
            #     # bscan = cv2.resize(cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE), (300,200))
            #     bscan = np.flip(bscan, axis=1)
            #     bscan = bscan[None,:,:]
            #     bscans.append(bscan)
            # bscans = np.vstack(bscans)
            # np.savez(os.path.join(tmp_save_cross_folder_oct, f"data_{count:06d}.npz"), 
            #             oct_bscans=bscans,
            #             md=md,
            #             tds=tds,
            #             race=race,
            #             male=male,
            #             glaucoma=glaucoma)
            count += 1

    # if count > selected_patient_num:
    #     break
pprint(f'instance counts: {count}, num in list: {len(list(idmaps.keys()))}', log_output)

# for training
for pid in upids[test_patients:]:
    indices = list(range(len(rnflt_maps[pid])))
    random.shuffle(indices)

    tmp_save_cross_folder = save_cross_folder
    for ind in indices:
        rnflt = rnflt_maps[pid][ind]
        glaucoma, md, tds, octID, datadir, clockhours, race, male, hisp, raw_info_dict = rnflt_attrs[pid][ind]

        # if (race not in dict_race_smp) or (dict_race_smp[race]<=0):
        #     continue

        if (race in set_of_races) :
            # if (dict_race_smp[race]>0):
            #     dict_race_smp[race] = dict_race_smp[race] - 1
            #     tmp_save_cross_folder = save_cross_folder

            #     raw_info_dict['md'] = md
            #     raw_info_dict['glaucoma'] = glaucoma
            #     selected_data.append(raw_info_dict)
            # elif (dict_race_smp_test[race]>0):
            #     dict_race_smp_test[race] = dict_race_smp_test[race] - 1
            #     tmp_save_cross_folder = save_cross_test_folder

            # else:
            #     continue

            # tmp_save_cross_folder = save_cross_folder

            raw_info_dict['md'] = md
            raw_info_dict['glaucoma'] = glaucoma
            
            selected_data.append(raw_info_dict)

            # dict_race_smp[race] = dict_race_smp[race] - 1
            
            bscans = []
            for i in range(1, 201):
                bscan = cv2.imread(os.path.join(newscanfolder, datadir, f'bscan{i}.jp2'))
                bscan = bscan[:,:,0]
                bscan = cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE)
                bscan = cv2.resize(bscan, (200,200))
                # bscan = cv2.resize(cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE), (300,200))
                bscan = np.flip(bscan, axis=1)
                bscan = bscan[None,:,:]
                bscans.append(bscan)
            bscans = np.vstack(bscans)

            np.savez(os.path.join(tmp_save_cross_folder,
                    f"data_{count:06d}.npz"), 
                    rnflt=rnflt,
                    ilm=ilm,
                    oct_bscans=bscans,
                    md=md,
                    tds=tds,
                    race=race,
                    male=male,
                    hispanic=hisp,
                    glaucoma=glaucoma)  
                    # clockhours=clockhours,
                    # pid=octID,
                    # datadir=datadir
            idmaps[count] = octID

            selected_mds.append(md)
            dict_race_mds[race].append(md)

            # bscans = []
            # for i in range(1, 201):
            #     bscan = cv2.imread(os.path.join(newscanfolder, datadir, f'bscan{i}.jp2'))
            #     bscan = bscan[:,:,0]
            #     bscan = cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE)
            #     bscan = cv2.resize(bscan, (200,200))
            #     # bscan = cv2.resize(cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE), (300,200))
            #     bscan = np.flip(bscan, axis=1)
            #     bscan = bscan[None,:,:]
            #     bscans.append(bscan)
            # bscans = np.vstack(bscans)
            # np.savez(os.path.join(tmp_save_cross_folder_oct, f"data_{count:06d}.npz"), 
            #             oct_bscans=bscans,
            #             md=md,
            #             tds=tds,
            #             race=race,
            #             male=male,
            #             glaucoma=glaucoma)
            count += 1

    # if count > selected_patient_num:
    #     break
pprint(f'instance counts: {count}, num in list: {len(list(idmaps.keys()))}', log_output)

np.save(os.path.join(save_folder, 'idmaps_td_prediction'), idmaps)

selected_data_df = pd.DataFrame(selected_data)
selected_data_df.to_csv(os.path.join(save_folder, 'train_selected_patients_meta.csv'), index=False)
selected_data_df = pd.DataFrame(selected_data_test)
selected_data_df.to_csv(os.path.join(save_folder, 'test_selected_patients_meta.csv'), index=False)

pprint(f'overall statistics on mds - mu: {np.mean(selected_mds):.4f}, sigma: {np.std(selected_mds):.4f}', log_output)
for x in list(dict_race_mds.keys()):
    pprint(f'{x} statistics on mds - mu: {np.mean(dict_race_mds[x]):.4f}, sigma: {np.std(dict_race_mds[x]):.4f}', log_output)