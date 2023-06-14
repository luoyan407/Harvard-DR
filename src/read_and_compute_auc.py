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

import sys
# sys.path.append('.')

# from src.modules import *
# from src.data_handler import *
# from src import logger
# from src.class_balanced_loss import *
# from typing import NamedTuple

input_file = ''

raw_data = np.load(input_file)

raw_data

for one_attr in np.unique(attrs).astype(int):
    preds_by_attr_tmp.append(preds[attrs == one_attr])
    gts_by_attr_tmp.append(gts[attrs == one_attr])
    aucs_by_attr.append(auc_score(preds[attrs == one_attr], gts[attrs == one_attr]))
    print(f'{one_attr}-attr auc: {aucs_by_attr[-1]:.4f}')