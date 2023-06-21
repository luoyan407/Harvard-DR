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
from loss_fair_con import FairSupConLoss

from sklearn.metrics import *
from sklearn.model_selection import KFold

import sys
sys.path.append('.')

from src.modules import *
from src.data_handler import *
from src import logger
from src.class_balanced_loss import *
from typing import NamedTuple
from src.models import * 
from fairlearn.metrics import *


class Imbalanced_Info(NamedTuple):
    beta: float = 0.9999
    gamma: float = 2.0
    samples_per_attr: list[int] = [0,0,0]
    loss_type: str = "sigmoid"
    no_of_classes: int = 2
    no_of_attr: int = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--pretrained-weights', default='', type=str)

# parser.add_argument('--model-architecture', default='whitenet', type=str)
parser.add_argument('--cont_method', default='FSCL*', type=str, help='FSCL vs FSCL* vs SupCon vs SimCLR')

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--data_dir', default='./results', type=str)
parser.add_argument('--model_type', default='./results', type=str)
parser.add_argument('--task', default='./results', type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--progression_outcome', default='', type=str)
parser.add_argument('--modality_types', default='rnflt', type=str, help='rnflt|bscans')
parser.add_argument('--fuse_coef', default=1.0, type=float)
parser.add_argument('--imbalance_beta', default=-1, type=float, help='default: 0.9999, if beta<0, then roll back to conventional loss')
parser.add_argument('--split_seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--attribute_type', default='race', type=str, help='race|gender')
parser.add_argument('--use_fair_scaling', default=0, type=int)
parser.add_argument('--split_ratio', default=0.0, type=float)
parser.add_argument("--need_balance", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")
parser.add_argument('--test_set_name', default='test', type=str)

                    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def train(model, criterion, optimizer, scaler, \
          train_dataset_loader, epoch, \
            total_iteration, imbalanced_info=None,\
                adv_model=None, adv_criterion=None, adv_optimizer=None):
    global device

    model.train()
    loss_batch = []
   
    for i, (input1, input2, target, attr) in enumerate(train_dataset_loader):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            input1 = input1.to(device)
            input2 = input2.to(device)
            target = target.to(device)
            attr = attr.to(device)
            input = torch.cat([input1, input2], dim=0)

            pred_feature = model(input) # .squeeze(1)
            
            bsz = target.shape[0]

            f1, f2 = torch.split(pred_feature, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)


            loss = criterion(features, target, attr, 0,\
                              "FSCL", i, device=device) 
        
        loss_batch.append(loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
    return np.mean(loss_batch)
    

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    if args.split_seed < 0:
        args.split_seed = int(np.random.randint(10000, size=1)[0])

    logger.log(f'===> random seed: {args.seed}')
    args.result_dir = './results_pretrain/{}_cont_pretrain_{}_{}_{}'.format(args.modality_types, \
                                        args.cont_method, args.attribute_type, args.model_type)

    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.model_type == 'vit' or args.model_type == 'swin':
        args.image_size = 224
    
    mean = (0.5000, 0.5000, 0.5000)
    std = (0.5000, 0.5000, 0.5000)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    trn_havo_dataset = HAVO_Diabetic_Retinopathy_Contrast(args.data_dir, subset='train', modality_type=args.modality_types, 
        task=args.task, resolution=args.image_size, \
            attribute_type=args.attribute_type, split_ratio=args.split_ratio, needBalance=args.need_balance,  transform=train_transform)
   
    logger.log(f'trn patients {len(trn_havo_dataset)} with {len(trn_havo_dataset)} samples!!')
    
    train_dataset_loader = torch.utils.data.DataLoader(
        trn_havo_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
   
    _, samples_per_attr = get_num_by_group(train_dataset_loader)
    logger.log(f'training group information:')
    logger.log(samples_per_attr)
    imb_info = Imbalanced_Info(beta=args.imbalance_beta, samples_per_attr=np.array(samples_per_attr))

    name_sen_at_diff_spe = ['sen_at_80spe', 'sen_at_85spe', 'sen_at_90spe', 'sen_at_95spe']

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    lastep_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'last_{args.perf_file}')

    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            acc_head_str = ', '.join([f'acc_class{x}' for x in range(len(samples_per_attr))])
            auc_head_str = ', '.join([f'auc_class{x}' for x in range(len(samples_per_attr))])
            sensitivity_head_str = ', '.join([f'sensitivity_class{x}' for x in range(len(samples_per_attr))])
            specificity_head_str = ', '.join([f'specificity_class{x}' for x in range(len(samples_per_attr))])

            sen_at_diff_spe_str = ''
            for i in range(len(name_sen_at_diff_spe)):
                sen_at_diff_spe_str += f'{name_sen_at_diff_spe[i]}, '
                for x in range(len(samples_per_attr)):
                    sen_at_diff_spe_str += f'class{x}_{name_sen_at_diff_spe[i]}, '

            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, es_acc, acc, {acc_head_str}, es_auc, wgd_auc, auc, {auc_head_str}, wgd_f1, f1, es_sensitivity, sensitivity, {sensitivity_head_str}, es_specificity, specificity, {specificity_head_str}, dpd, dpr, eod, eor, {sen_at_diff_spe_str} path\n')
        if not os.path.exists(lastep_global_perf_file):
            acc_head_str = ', '.join([f'acc_class{x}' for x in range(len(samples_per_attr))])
            auc_head_str = ', '.join([f'auc_class{x}' for x in range(len(samples_per_attr))])
            with open(lastep_global_perf_file, 'w') as f:
                f.write(f'epoch, es_acc, acc, {acc_head_str}, es_auc, auc, {auc_head_str}, dpd, dpr, eod, eor, path\n')

    
    out_dim = 128
    criterion = FairSupConLoss(temperature=0.1)
    predictor_head = nn.Identity()

    # criterion = nn.CrossEntropyLoss()
    
    if args.modality_types == 'rpet' or args.modality_types == 'fundus':
        in_dim = 3
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    elif 'bscan' in args.modality_types:
        in_dim = 128 # 200
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    
    # model = nn.Sequential(backbone, predictor_head)
    # model = torch.compile(model)
    model = model.to(device)
    
    scaler = torch.cuda.amp.GradScaler()


    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
     
    # print("len of train_dataset: ", len(trn_havo_dataset))
    # print("len of validation_dataset: ", len(tst_havo_dataset))
    
    start_epoch = 0
    best_top1_accuracy = 0.

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
       
    total_iteration = len(trn_havo_dataset)//args.batch_size


    best_wgd_f1 = 0
    best_wgd_auc = 0
    best_sensitivity_groups = None
    best_specificity_groups = None
    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_f1 = sys.float_info.min
    best_sensitivity = sys.float_info.min
    best_specificity = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_es_sensitivity = sys.float_info.min
    best_es_specificity = sys.float_info.min
    best_sen_at_diff_spe = None
    best_ep = 0
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(model, criterion, \
                    optimizer, scaler, train_dataset_loader, \
                        epoch, total_iteration, imbalanced_info=imb_info)
        scheduler.step()
        state = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict()
        }
        torch.save(state, os.path.join(args.result_dir, f"model_ep{epoch:03d}.pth"))

        
