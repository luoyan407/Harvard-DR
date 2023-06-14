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

from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def pprint(out_str, redirect_out=None, flag='a'):
    # print(out_str)
    if redirect_out is not None:
        with open(redirect_out, flag) as sys.stdout:
            print(out_str)
    else:
        print(out_str)

def select_samples(arr, indices):
    ret = []
    for i in indices:
        ret.append(arr[i])
    return ret

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

# newscanfolder = '/shared/hdds_20T/cirrus_from_dicom/scans'
# newscanfolder_p2 = '/shared/hdds_20T/cirrus_from_dicom_part2/scans'
newscanfolder = '/shared/hdds_20T/yl535/elzelab/cirrus_ongoing/scans'
newscanfolder_p2 = '/shared/external_hdds/external20TB_1/cirrus_from_dicom/scans'

# need to mount eris data elzelab by
# sshfs yl535@erisxdl.partners.org:/data/elzelab ~/elzelab

vffile = "/shared/hdds_20T/yl535/elzelab/hfa_ongoing/hfa_ongoing_merged_24-2_subset_cleaned_all_in.csv"
newoctfile = '/shared/hdds_20T/yl535/elzelab/cirrus_ongoing/metadata_OpticDiscCube_ongoing_cleaned.csv'
# exclude_file = '/shared/ssd_16T/yl535/project/python/datasets/crosssectional/test1000_pidmaps.npy'
exclude_file = ''

# save_folder = '/shared/ssd_16T/yl535/project/python/datasets/havo_crosssectional_new_export/crosssectional_fairness_all'

selected_patient_num = 3000
selected_patient_num_train = 2000 # 31031 #
balanced_num = -1 # 1000
ratio_of_patients_for_test = 0.3
num_attr = 3
extract_to_files = True
identity_type = 'race'

# save_folder = f'/shared/ssd_16T/yl535/project/python/datasets/harvard/glaucoma.csv'
match_table_file = f'/shared/ssd_16T/yl535/project/python/datasets/harvard/glaucoma.csv'
save_to_folder = os.path.dirname(match_table_file)
save_file_name = os.path.basename(match_table_file)

seed = 9220 # 7652 # 2740 # -1

# samples_per_attr = [int(selected_patient_num/num_attr), int(selected_patient_num/num_attr), selected_patient_num-2*int(selected_patient_num/num_attr)]
# samples_per_attr_test = [int(selected_patient_num_test/num_attr), int(selected_patient_num_test/num_attr), selected_patient_num_test-2*int(selected_patient_num_test/num_attr)]

# dict_race = {1:'American Indian or Alaska Native', 
#                 2:'Asian', 
#                 3:'Black or African American', 
#                 4:'Hispanic or Latino', 
#                 5:'Native Hawaiian or Other Pacific Islander', 
#                 6:'Other', 
#                 7:'White or Caucasian'}
# dict_race = {2:'Asian', 
#                 3:'Black or African American', 
#                 7:'White or Caucasian'}
dict_race = {2: 0, 
                3: 1, 
                7: 2}

if seed < 0:
    seed = int(np.random.randint(10000, size=1)[0])
set_random_seed(seed)

output_folder = f'/shared/ssd_16T/yl535/project/python/datasets/harvard/glaucoma_{seed}'
log_output = os.path.join(save_to_folder, f'{save_file_name}_selected_{seed}.log')
log_output = None

pprint(f'====> seed: {seed}', log_output, 'w')

isExist = os.path.exists(save_to_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(save_to_folder)

if extract_to_files:
    isExist = os.path.exists(output_folder)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(output_folder)

#-------start to process---------
# get exluded patient list if any
blacklist = []
if exclude_file != '':
    raw_data = np.load(exclude_file, allow_pickle=True)
    blacklist = [x[:x.find('_')] for x in list(raw_data.item().values())]
		
octmetanew = pd.read_csv(match_table_file, sep=',')
# existed_paths = []
# for octpath in octmetanew['datadir']:
#     if os.path.exists(os.path.join(newscanfolder, octpath)):
#         existed_paths.append(octpath)
# octmetanew_exist = octmetanew[octmetanew['datadir'].isin(existed_paths)]

pprint(f'now process oct samples', log_output, 'a')
clockhours_all = []
glaucoma_all = []
datadir_all = []
octpath_all = []
md_all = []
count = 0
unmatched = []
race_all = []
male_all = []
hispanic_all = []
ilm_maps = {}
rnflt_maps = {}
rnflt_attrs = {}
upids = []
pid_all = []
for index, row in octmetanew.iterrows():
    pid = str(row[0])
    male = int(row['male'])
    signalstrength = float(row['signalstrength'])
    righteye = str(row['righteye'])
    timeoftest = str(row['timeoftest']).strip()
    datadir = row['datadir']
    race = row['race']
    # pprint(row, log_output)
    hisp = row['hispanic']
    race = dict_race[int(race)]
    hisp = int(hisp)
    md = float(row['md'])
    glaucoma = int(row['glaucoma'])

    # if (row['race']) != 'NA' and (not np.isnan(row['race'])):
    #     race = int(row['race'])
    
    #cid = pid + '_' + righteye + '_' + male 
    cid = pid + '_' + righteye
    octID = pid + '_' + righteye + '_' + timeoftest
    
    # if cid not in vfid_timetd.keys():
    #     unmatched.append(pid)
    #     continue

    try:

        clockhours = []
        for i in range(1,13):
            clock_v = float(row[f'clockhour{i}'])
            clockhours.append(clock_v)
        clockhours = np.array(clockhours).astype(float)

        race_all.append(race)
        male_all.append(male)
        hispanic_all.append(hisp)
        glaucoma_all.append(glaucoma)
        clockhours_all.append(clockhours)
        md_all.append(md)
        datadir_all.append(datadir)
        pid_all.append(pid)
        
        octpath = newscanfolder
        if not os.path.exists(os.path.join(octpath, datadir)): 
            octpath = newscanfolder_p2
        octpath_all.append(octpath)
        
        # if extract_to_files:
        #     # disc
        #     data_disk = open(os.path.join(octpath, datadir, 'mask_disc.csv'))
        #     img_disk = np.reshape([float(row) for row in data_disk], (200, -1))
        #     # cup
        #     data_cup = open(os.path.join(octpath, datadir, 'mask_cup.csv'))
        #     img_cup = np.reshape([float(row) for row in data_cup], (200, -1))
        #     # ilm
        #     data_ilm2nd = open(os.path.join(octpath, datadir, 'segmentation_ilm_2nd.csv'))
        #     img_ilm2nd = np.reshape([float(row) for row in data_ilm2nd], (200, -1))
        #     data_ilm = open(os.path.join(octpath, datadir, 'segmentation_ilm.csv'))
        #     img_ilm = np.reshape([float(row) for row in data_ilm], (200, -1))
        #     # gcl
        #     data_gcl = open(os.path.join(octpath, datadir, 'segmentation_rnfl_to_gcl.csv'))
        #     img_gcl = np.reshape([float(row) for row in data_gcl], (200, -1))
        #     # rnflt map
        #     rnflt_map = (img_gcl - img_ilm2nd) * 0.00195503 * 1000 # bscanPixelspacingDepth * 1000
        #     rnflt_map[img_disk==1] = -1
        #     rnflt_map[img_cup==1] = -2
        #     img_gcl = img_gcl * 0.00195503 * 1000
        #     img_ilm = np.abs(img_ilm-np.max(img_ilm)) * 0.00195503 * 1000

        #     bscans = []
        #     for i in range(1, 201):
        #         bscan = cv2.imread(os.path.join(octpath, datadir, f'bscan{i}.jp2'))
        #         bscan = bscan[:,:,0]
        #         bscan = cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE)
        #         bscan = cv2.resize(bscan, (200,200))
        #         # bscan = cv2.resize(cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE), (300,200))
        #         bscan = np.flip(bscan, axis=1)
        #         bscan = bscan[None,:,:]
        #         bscans.append(bscan)
        #     bscans = np.vstack(bscans)

        #     np.savez(os.path.join(output_folder,
        #             f"data_{index_shuf[i]:06d}.npz"), 
        #             rnflt=rnflt_map,
        #             ilm=img_ilm,
        #             oct_bscans=bscans,
        #             clockhours=clockhours,
        #             md=md,
        #             race=race,
        #             male=male,
        #             hispanic=hisp,
        #             glaucoma=glaucoma)
        #             # tds=tds,
        #             # clockhours=clockhours,
        #             # pid=octID,
        #             # datadir=datadir
        
    except Exception as e:
        unmatched.append(pid)
        pprint(e, log_output)

    if index % 50000 == 0:
        pprint(index, log_output)

unique_pids = np.unique(pid_all)
pprint(f'total # of patients: {len(unique_pids)}, total # of match records: {len(glaucoma_all)}', log_output)

race_all_new = []
male_all_new = []
hispanic_all_new = []
clockhours_all_new = []
glaucoma_all_new = []
pid_all_new = []
index_shuf = list(range(len(glaucoma_all)))
random.shuffle(index_shuf)
for i in index_shuf:
    # if i % 10000 == 0:
    #     pprint(f'sampling {index}', log_output)

    race = race_all[index_shuf[i]]
    male = male_all[index_shuf[i]]
    hisp = hispanic_all[index_shuf[i]]
    glaucoma = glaucoma_all[index_shuf[i]]
    clockhours = clockhours_all[index_shuf[i]]
    md = md_all[index_shuf[i]]
    datadir = datadir_all[index_shuf[i]]
    octpath = octpath_all[index_shuf[i]]
    pid = pid_all[index_shuf[i]]

    if selected_patient_num > 0 and len(glaucoma_all_new) >= selected_patient_num:
        break

    if pid in pid_all_new:
        continue

    if selected_patient_num < 0 or balanced_num < 0 or np.sum(np.array(race_all_new) == race) < balanced_num:
        race_all_new.append(race)
        male_all_new.append(male)
        hispanic_all_new.append(hisp)
        clockhours_all_new.append(clockhours)
        glaucoma_all_new.append(glaucoma)
        pid_all_new.append(pid)

        if extract_to_files:
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

            bscans = []
            for j in range(1, 201):
                bscan = cv2.imread(os.path.join(octpath, datadir, f'bscan{j}.jp2'))
                bscan = bscan[:,:,0]
                bscan = cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE)
                bscan = cv2.resize(bscan, (200,200))
                # bscan = cv2.resize(cv2.rotate(bscan, cv2.ROTATE_90_CLOCKWISE), (300,200))
                bscan = np.flip(bscan, axis=1)
                bscan = bscan[None,:,:]
                bscans.append(bscan)
            bscans = np.vstack(bscans)

            np.savez(os.path.join(output_folder,
                    f"data_{index_shuf[i]:06d}.npz"), 
                    rnflt=rnflt_map,
                    ilm=img_ilm,
                    oct_bscans=bscans,
                    clockhours=clockhours,
                    md=md,
                    race=race,
                    male=male,
                    hispanic=hisp,
                    glaucoma=glaucoma)
                    # tds=tds,
                    # clockhours=clockhours,
                    # pid=octID,
                    # datadir=datadir

index_shuf = list(range(len(glaucoma_all_new)))
random.shuffle(index_shuf)

train_data = np.array(select_samples(clockhours_all_new, index_shuf[:selected_patient_num_train]))
train_labels = np.array(select_samples(glaucoma_all_new, index_shuf[:selected_patient_num_train]))
train_race_info = np.array(select_samples(race_all_new, index_shuf[:selected_patient_num_train]))
train_sex_info = np.array(select_samples(male_all_new, index_shuf[:selected_patient_num_train]))
train_hispanic_info = np.array(select_samples(hispanic_all_new, index_shuf[:selected_patient_num_train]))

test_data = np.array(select_samples(clockhours_all_new, index_shuf[selected_patient_num_train:]))
test_labels = np.array(select_samples(glaucoma_all_new, index_shuf[selected_patient_num_train:]))
test_race_info = np.array(select_samples(race_all_new, index_shuf[selected_patient_num_train:]))
test_sex_info = np.array(select_samples(male_all_new, index_shuf[selected_patient_num_train:]))
test_hispanic_info = np.array(select_samples(hispanic_all_new, index_shuf[selected_patient_num_train:]))

regressor = LogisticRegression(random_state=0).fit(train_data, train_labels)
preds = regressor.predict_proba(test_data)[:,1]

fpr, tpr, thresholds = roc_curve(test_labels, preds)
val_AUC = auc(fpr, tpr)
pprint(f'overall AUC: {val_AUC:.4f}', log_output)

# Race
unique, tmp_count = np.unique(race_all_new, return_counts=True)
pprint('Race distribution', log_output)
pprint(tmp_count, log_output)
for i in unique:
    train_cond_index = train_race_info == i
    test_cond_index = test_race_info == i
    sub_train_data = train_data[train_cond_index]
    sub_train_labels = train_labels[train_cond_index]
    sub_test_data = test_data[test_cond_index]
    sub_test_labels = test_labels[test_cond_index]

    regressor = LogisticRegression(random_state=0).fit(sub_train_data, sub_train_labels)
    preds = regressor.predict_proba(sub_test_data)[:,1]
    fpr, tpr, thresholds = roc_curve(sub_test_labels, preds)
    val_AUC = auc(fpr, tpr)
    pprint(f'Race {i}-th identity AUC: {val_AUC:.4f}', log_output)

# Sex
unique, tmp_count = np.unique(male_all_new, return_counts=True)
pprint('Sex distribution', log_output)
pprint(tmp_count, log_output)
for i in unique:
    train_cond_index = train_sex_info == i
    test_cond_index = test_sex_info == i
    sub_train_data = train_data[train_cond_index]
    sub_train_labels = train_labels[train_cond_index]
    sub_test_data = test_data[test_cond_index]
    sub_test_labels = test_labels[test_cond_index]

    regressor = LogisticRegression(random_state=0).fit(sub_train_data, sub_train_labels)
    preds = regressor.predict_proba(sub_test_data)[:,1]
    fpr, tpr, thresholds = roc_curve(sub_test_labels, preds)
    val_AUC = auc(fpr, tpr)
    pprint(f'Sex {i}-th identity AUC: {val_AUC:.4f}', log_output)

# Ethnicity
unique, tmp_count = np.unique(hispanic_all_new, return_counts=True)
pprint('Ethnicity distribution', log_output)
pprint(tmp_count, log_output)
for i in unique:
    train_cond_index = train_hispanic_info == i
    test_cond_index = test_hispanic_info == i
    sub_train_data = train_data[train_cond_index]
    sub_train_labels = train_labels[train_cond_index]
    sub_test_data = test_data[test_cond_index]
    sub_test_labels = test_labels[test_cond_index]

    regressor = LogisticRegression(random_state=0).fit(sub_train_data, sub_train_labels)
    preds = regressor.predict_proba(sub_test_data)[:,1]
    fpr, tpr, thresholds = roc_curve(sub_test_labels, preds)
    val_AUC = auc(fpr, tpr)
    pprint(f'Ethnicity {i}-th identity AUC: {val_AUC:.4f}', log_output)