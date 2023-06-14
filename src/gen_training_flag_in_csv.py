# https://github.com/developer0hye/PyTorch-ImageNet
import os
import argparse
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim import *
import torch.nn.functional as F

from sklearn.metrics import *
from sklearn.model_selection import KFold

# import sys
# sys.path.append('.')

# from src.modules import *
# from src.data_handler import *
# from src import logger

def find_all_files(folder, suffix='npz', prefix='data'):
    # refer to https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

def get_all_pids(data_dir):
    pids = []
    fnames = []
    dict_pid_fid = {}
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        pid = raw_data['pid'].item()
        pid = pid[:pid.find('_')]
        fnames.append(f)
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
    pids=list(set(pids))
    return pids, dict_pid_fid, fnames

if __name__ == '__main__':

    csv_folder = '/shared/ssd_16T/meecs/progression_data/old_data/oct.table.with.six.progression.outcomes.with.time.csv'
    new_csv_file = 'oct.table.with.six.progression.outcomes.with.time.trainflag.csv'
    data_folder = '/shared/ssd_16T/yl535/project/python/datasets/havo_longitudinal_new_export/longitudinal_v3'

    all_files = find_all_files(data_folder)
    pids, dict_pid_fid, fnames = get_all_pids(data_folder)
    print(all_files[:5])
    print(fnames[:5])
    sys.exit()
    pids = np.array(pids)
    print(f"# of patients: {len(pids)}")

    split_seed = 5
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    for fold, (train_pids, test_pids) in enumerate(kfold.split(pids)):
        print(f'============= start fold {fold} =============')
        train_ids = []
        for x in train_pids:
            train_ids = train_ids + dict_pid_fid[pids[x]]
        test_ids = []
        for x in test_pids:
            test_ids = test_ids + dict_pid_fid[pids[x]]
        break

    octmetadata = csv.reader(open(csv_folder))
    with open(open(csv_folder),'r') as csvinput:
        octheader = next(octmetadata)

    for row in octmetadata:
        pass
    dir_folder = os.path.dirname(csv_folder)
    print(dir_folder)
    sys.exit()
    all_files = find_all_files(data_folder)
    for x in range(10):
        pass