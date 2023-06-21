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
    top1_accuracy_batch = []
    top5_accuracy_batch = []
    
    lambdas =0.2
    
    preds = []
    gts = []
    attrs = []
    datadirs = []

    preds_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]
    t1 = time.time()
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            input = input.to(device)
            target = target.to(device)
            attr = attr.to(device)

            pred = model(input) # .squeeze(1)
            pred = pred.squeeze()
            
            p_z = adv_model(pred)
            
            if args.attribute_type=='race':
                one_hot_attr = F.one_hot(attr.long(), num_classes=3).float().to(device)
                loss_adv = adv_criterion(p_z, one_hot_attr) * lambdas
            
            else:
                loss_adv = adv_criterion(p_z.squeeze(), attr.float()) * lambdas

            if imbalanced_info.beta <= 0.:
                loss = criterion(pred, target.long())
            else:
                loss_weights = compute_rescaled_weight(imbalanced_info.samples_per_attr[attr.detach().cpu().numpy()], imbalanced_info.no_of_attr, imbalanced_info.beta)
                loss = F.binary_cross_entropy_with_logits(pred, target, weight=loss_weights.type_as(pred))
           
            loss = loss - loss_adv
            loss_adv.backward(retain_graph=True)
            adv_optimizer.step()
            pred_prob = torch.sigmoid(pred.detach())
            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attrs.append(attr.detach().cpu().numpy())

            for j, x in enumerate(attr.detach().cpu().numpy()):
                preds_by_attr[x].append(pred_prob[j])
                gts_by_attr[x].append(target[j].item())

        loss_batch.append(loss.item())
        
        top1_accuracy = accuracy(pred, target, topk=(1,))[0]
        
        top1_accuracy_batch.append(top1_accuracy)

        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=0).astype(int)
    cur_auc, _ = auc_score_multiclass(preds, gts)
    acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))[0]

    pred_labels = (preds >= 0.5).astype(float)
    
    dpd = multiclass_demographic_parity(preds, gts, attrs)
    dpr = 0
    eod = 0
    eor = 0

    torch.cuda.synchronize()
    t2 = time.time()

    print(f"train ====> epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f} time: {t2 - t1:.4f}")

    preds_by_attr_tmp = []
    gts_by_attr_tmp = []
    aucs_by_attr = []
    for one_attr in np.unique(attrs).astype(int):
        preds_by_attr_tmp.append(preds[attrs == one_attr])
        gts_by_attr_tmp.append(gts[attrs == one_attr])
        tmp_auc, _ = auc_score_multiclass(preds[attrs == one_attr], gts[attrs == one_attr])
        aucs_by_attr.append(tmp_auc)
        print(f'{one_attr}-attr auc: {aucs_by_attr[-1]:.4f}')

    t1 = time.time()

    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs, [preds_by_attr_tmp, gts_by_attr_tmp, aucs_by_attr], [acc, dpd, dpr, eod, eor]
    

def validation(model, criterion, optimizer, validation_dataset_loader, epoch, result_dir=None, imbalanced_info=None):
    global device

    model.eval()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []

    preds = []
    gts = []
    attrs = []
    datadirs = []

    preds_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]

    with torch.no_grad():
        for i, (input, target, attr) in enumerate(validation_dataset_loader):
            input = input.to(device)
            target = target.to(device)
            
            pred = model(input) # .squeeze(1)

            if imbalanced_info.beta <= 0.:
                loss = criterion(pred, target.long())
            else:
                loss_weights = compute_rescaled_weight(imbalanced_info.samples_per_attr[attr.detach().cpu().numpy()], imbalanced_info.no_of_attr, imbalanced_info.beta)
                loss = F.binary_cross_entropy_with_logits(pred, target, weight=loss_weights.type_as(pred))
            # loss = CB_loss_(target, pred, 
            #         imbalanced_info.samples_per_attr, imbalanced_info.no_of_attr, imbalanced_info.no_of_classes, 
            #         imbalanced_info.loss_type, imbalanced_info.beta, imbalanced_info.gamma)

            pred_prob = torch.sigmoid(pred.detach())
            # pred_prob = F.softmax(pred.detach(), dim=1)
            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attrs.append(attr.detach().cpu().numpy())
            # datadirs = datadirs + datadir

            for j, x in enumerate(attr.detach().cpu().numpy()):
                preds_by_attr[x].append(pred_prob[j])
                gts_by_attr[x].append(target[j].item())
            

            loss_batch.append(loss.item())

            # pred_prob = torch.sigmoid(pred.detach())
            # preds.append(pred_prob.detach().cpu().numpy())
            # gts.append(target.detach().cpu().numpy())
            
            top1_accuracy = accuracy(pred, target, topk=(1,))[0]
        
            top1_accuracy_batch.append(top1_accuracy)
            # top5_accuracy_batch.append(top5_accuracy)
        
    loss = np.mean(loss_batch)
    # top5_accuracy = np.mean(top5_accuracy_batch)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=0).astype(int)

    cur_auc, cur_sen_at_diff_spe = auc_score_multiclass(preds, gts)
    # acc = np.mean(top1_accuracy_batch)
    acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))[0]
    cur_f1 = f1_score(gts, np.argmax(preds, axis=1), average='macro')

    # pred_labels = np.argmax(preds, axis=1)
    pred_labels = (preds >= 0.5).astype(float)
    # dpd = demographic_parity_difference(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # dpr = demographic_parity_ratio(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # eod = equalized_odds_difference(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # eor = equalized_odds_ratio(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    dpd = multiclass_demographic_parity(preds, gts, attrs)
    dpr = 0
    eod = 0
    eor = 0

    # datadirs = np.array(datadirs)

    print(f"test <==== epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f}")

    preds_by_attr_tmp = []
    gts_by_attr_tmp = []
    aucs_by_attr = []
    for one_attr in np.unique(attrs).astype(int):
        preds_by_attr_tmp.append(preds[attrs == one_attr])
        gts_by_attr_tmp.append(gts[attrs == one_attr])
        tmp_auc, _ = auc_score_multiclass(preds[attrs == one_attr], gts[attrs == one_attr])
        aucs_by_attr.append(tmp_auc)
        print(f'{one_attr}-attr auc: {aucs_by_attr[-1]:.4f}')

    # print("val-", "epoch: ", epoch, " loss: ", loss, " top1 acc: ", top1_accuracy, " top5 acc: ", top5_accuracy)
    
    return loss, acc, cur_auc, preds, gts, attrs, [preds_by_attr_tmp, gts_by_attr_tmp, aucs_by_attr], [acc, dpd, dpr, eod, eor, cur_f1, cur_sen_at_diff_spe]


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    if args.split_seed < 0:
        args.split_seed = int(np.random.randint(10000, size=1)[0])

    logger.log(f'===> random seed: {args.seed}')

    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.model_type == 'vit' or args.model_type == 'swin':
        args.image_size = 224

    trn_havo_dataset = Harvard_Diabetic_Retinopathy(args.data_dir, subset='train', modality_type=args.modality_types, 
        task=args.task, resolution=args.image_size, \
            attribute_type=args.attribute_type, \
                split_ratio=args.split_ratio, needBalance=args.need_balance, args=args)
    tst_havo_dataset = Harvard_Diabetic_Retinopathy(args.data_dir, subset='test', modality_type=args.modality_types, 
        task=args.task, resolution=args.image_size, attribute_type=args.attribute_type, args=args)
    
    logger.log(f'trn patients {len(trn_havo_dataset)} with {len(trn_havo_dataset)} samples, val patients {len(tst_havo_dataset)} with {len(tst_havo_dataset)} samples')
   
    train_dataset_loader = torch.utils.data.DataLoader(
        trn_havo_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    validation_dataset_loader = torch.utils.data.DataLoader(
        tst_havo_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    _, samples_per_attr = get_num_by_group(validation_dataset_loader)
    logger.log(f'testing group information:')
    logger.log(samples_per_attr)
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

    # model_dict = {'whitenet': whitenet.WhiteNet(),
    #               'tiny': tiny.YOLOv3TinyBackbone()}
    # model = model_dict[args.model_architecture]
    # model = model.to(device)

    if args.task == 'md':
        out_dim = 3
        # predictor_head = General_Logistic(trn_dataset.min_vf_val/trn_dataset.max_vf_val*args.stretch_ratio_vf, args.stretch_ratio_vf)
        criterion = nn.MSELoss()
        predictor_head = nn.Identity() # nn.Tanhshrink()
    elif args.task == 'cls': 
        out_dim = 3
        # predictor_head = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        predictor_head = nn.Sigmoid()
    elif args.task == 'tds': 
        out_dim = 52
        criterion = nn.MSELoss()
        # predictor_head = General_Logistic(trn_dataset.min_vf_val/trn_dataset.max_vf_val*args.stretch_ratio_vf, args.stretch_ratio_vf)
        predictor_head = nn.Identity()

    criterion = nn.CrossEntropyLoss()
    out_dim = 3
    if args.modality_types == 'rpet' or args.modality_types == 'fundus':
        in_dim = 3
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    elif 'bscan' in args.modality_types:
        in_dim = 128 # 200
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    elif args.modality_types == 'rnflt+ilm':
        in_dim = 2
        model = OphBackbone(model_type=args.model_type, in_dim=in_dim, coef=args.fuse_coef)
    # model = nn.Sequential(backbone, predictor_head)
    # model = torch.compile(model)
    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    if args.attribute_type=='race':
        adv_model = Adversary(n_sensitive=3).to(device)
        adv_criterion = torch.nn.CrossEntropyLoss()
    else:
        adv_model = Adversary(n_sensitive=1).to(device)
        adv_criterion = torch.nn.BCEWithLogitsLoss()
    
    adv_optimizer = AdamW(adv_model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)

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
        train_loss, train_acc, train_auc, trn_preds, trn_gts, trn_attrs, \
            trn_pred_gt_by_attrs, \
                trn_other_metrics = train(model, criterion, \
                    optimizer, scaler, train_dataset_loader, \
                        epoch, total_iteration, imbalanced_info=imb_info, adv_model=adv_model, adv_criterion=adv_criterion,  adv_optimizer=adv_optimizer)
        test_loss, test_acc, test_auc, tst_preds, tst_gts, tst_attrs, tst_pred_gt_by_attrs, tst_other_metrics = validation(model, criterion, optimizer, validation_dataset_loader, epoch, imbalanced_info=imb_info)
        scheduler.step()


        trn_acc_groups = []
        trn_auc_groups = []
        for i_group in range(len(trn_pred_gt_by_attrs[0])):
            trn_acc_groups.append(accuracy(torch.from_numpy(trn_pred_gt_by_attrs[0][i_group]).cuda(), torch.from_numpy(trn_pred_gt_by_attrs[1][i_group]).cuda(), topk=(1,))[0]) 
            tmp_auc, sen_at_diff_spe = auc_score_multiclass(trn_pred_gt_by_attrs[0][i_group], trn_pred_gt_by_attrs[1][i_group])
            trn_auc_groups.append(tmp_auc)
        
        wgd_f1 = 0
        wgd_auc = 0
        acc_groups = []
        auc_groups = []
        sensitivity_groups = []
        specificity_groups = []
        tst_sen_at_diff_spe = []
        tst_sen_at_diff_spe_groups = []
        for i_group in range(len(tst_pred_gt_by_attrs[0])):
            acc_groups.append(accuracy(torch.from_numpy(tst_pred_gt_by_attrs[0][i_group]).cuda(), torch.from_numpy(tst_pred_gt_by_attrs[1][i_group]).cuda(), topk=(1,))[0]) 
            tmp_auc, sen_at_diff_spe = auc_score_multiclass(tst_pred_gt_by_attrs[0][i_group], tst_pred_gt_by_attrs[1][i_group])
            auc_groups.append(tmp_auc)
            tst_sen_at_diff_spe.append(sen_at_diff_spe)
            # sensitivity, specificity = compute_sensitivity_specificity(tst_pred_gt_by_attrs[0][i_group], tst_pred_gt_by_attrs[1][i_group])
            sensitivity, specificity = 0., 0.
            sensitivity_groups.append(sensitivity)
            specificity_groups.append(specificity)
            wgd_auc += np.abs(test_auc - tmp_auc)

            tmp_f1 = f1_score(tst_pred_gt_by_attrs[1][i_group], np.argmax(tst_pred_gt_by_attrs[0][i_group], axis=1), average='macro')
            wgd_f1 += np.abs(tst_other_metrics[5] - tmp_f1)
        wgd_auc = wgd_auc / len(tst_pred_gt_by_attrs[0])
        wgd_f1 = wgd_f1 / len(tst_pred_gt_by_attrs[0])

        es_acc = equity_scaled_accuracy(tst_preds, tst_gts, tst_attrs)
        es_auc = 0 # equity_scaled_AUC(tst_preds, tst_gts, tst_attrs)
        es_sensitivity, es_specificity = 0., 0. # equity_scaled_sensitivity_specificity(tst_preds, tst_gts, tst_attrs)
        # test_sensitivity, test_specificity = compute_sensitivity_specificity(tst_preds, tst_gts)
        test_sensitivity, test_specificity = 0., 0.

        if best_auc <= test_auc:
            best_auc = test_auc
            best_acc = test_acc
            best_ep = epoch
            # all_preds[fold] = tst_preds
            # all_gts[fold] = tst_gts
            # all_attrs[fold] = tst_attrs
            # all_datadirs[fold] = tst_datadirs
            best_pred_gt_by_attr = tst_pred_gt_by_attrs
            best_tst_other_metrics = tst_other_metrics
            best_acc_groups = acc_groups
            best_auc_groups = auc_groups
            best_es_acc = es_acc
            best_es_auc = es_auc
            best_f1 = tst_other_metrics[5]

            best_wgd_auc = wgd_auc
            best_wgd_f1 = wgd_f1

            best_sensitivity = test_sensitivity
            best_specificity = test_specificity
            best_es_sensitivity = es_sensitivity
            best_es_specificity = es_specificity
            best_sensitivity_groups = sensitivity_groups
            best_specificity_groups = specificity_groups

            best_sen_at_diff_spe = tst_other_metrics[-1]
            best_sen_at_diff_spe_groups = tst_sen_at_diff_spe

            state = {
            'epoch': epoch,# zero indexing
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'train_auc': train_auc,
            'test_auc': test_auc
            }
            # torch.save(state, os.path.join(args.result_dir, f"model_fold{fold}_ep{epoch:03d}.pth"))

        print(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        for i_attr in range(len(best_pred_gt_by_attr[-1])):
            print(f'---- best AUC at {i_attr}-attr {best_pred_gt_by_attr[-1][i_attr]:.4f} at epoch {best_ep}')
            logger.log(f'---- best AUC at {i_attr}-attr {best_pred_gt_by_attr[-1][i_attr]:.4f} at epoch {best_ep}')
    
        if args.result_dir is not None:
            np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'), 
                        val_pred=tst_preds, val_gt=tst_gts, val_attr=tst_attrs)


        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(train_loss,4))
        logger.logkv('trn_acc', round(train_acc,4))
        logger.logkv('trn_auc', round(train_auc,4))
        logger.logkv('trn_acc', round(trn_other_metrics[0],4))
        logger.logkv('trn_dpd', round(trn_other_metrics[1],4))
        logger.logkv('trn_dpr', round(trn_other_metrics[2],4))
        logger.logkv('trn_eod', round(trn_other_metrics[3],4))
        logger.logkv('trn_eor', round(trn_other_metrics[4],4))
        for i_group in range(len(trn_acc_groups)):
            logger.logkv(f'trn_acc_class{i_group}', round(trn_acc_groups[i_group],4))
        for i_group in range(len(trn_auc_groups)):
            logger.logkv(f'trn_auc_class{i_group}', round(trn_auc_groups[i_group],4))

        logger.logkv('val_loss', round(test_loss,4))
        # logger.logkv('val_acc', round(test_acc,4))
        # logger.logkv('val_auc', round(test_auc,4))
        # logger.logkv('val_es_acc', round(es_acc,4))
        # logger.logkv('val_es_auc', round(es_auc,4))
        logger.logkv('val_acc', round(tst_other_metrics[0],4))
        logger.logkv('val_dpd', round(tst_other_metrics[1],4))
        logger.logkv('val_dpr', round(tst_other_metrics[2],4))
        logger.logkv('val_eod', round(tst_other_metrics[3],4))
        logger.logkv('val_eor', round(tst_other_metrics[4],4))
        logger.logkv('val_f1', round(tst_other_metrics[5],4))

        logger.logkv('val_wgd_f1', round(wgd_f1,4))
        logger.logkv('val_wgd_auc', round(wgd_auc,4))

        logger.logkv('val_sen_at_80spe', round(tst_other_metrics[-1][0],4))
        logger.logkv('val_sen_at_85spe', round(tst_other_metrics[-1][1],4))
        logger.logkv('val_sen_at_90spe', round(tst_other_metrics[-1][2],4))
        logger.logkv('val_sen_at_95spe', round(tst_other_metrics[-1][3],4))
        for i_group in range(len(tst_sen_at_diff_spe)):
            for j in range(len(name_sen_at_diff_spe)):
                logger.logkv(f'val_{name_sen_at_diff_spe[j]}_class{i_group}', round(tst_sen_at_diff_spe[i_group][j],4))

        logger.logkv('val_es_acc', round(es_acc,4))
        logger.logkv('val_acc', round(test_acc,4))
        for i_group in range(len(acc_groups)):
            logger.logkv(f'val_acc_class{i_group}', round(acc_groups[i_group],4))

        logger.logkv('val_es_auc', round(es_auc,4))
        logger.logkv('val_auc', round(test_auc,4))
        for i_group in range(len(auc_groups)):
            logger.logkv(f'val_auc_class{i_group}', round(auc_groups[i_group],4))

        logger.logkv('val_es_sensitivity', round(es_sensitivity,4))
        logger.logkv('val_sensitivity', round(test_sensitivity,4))
        for i_group in range(len(sensitivity_groups)):
            logger.logkv(f'val_sensitivity_class{i_group}', round(sensitivity_groups[i_group],4))

        logger.logkv('val_es_specificity', round(es_specificity,4))
        logger.logkv('val_specificity', round(test_specificity,4))
        for i_group in range(len(specificity_groups)):
            logger.logkv(f'val_specificity_class{i_group}', round(specificity_groups[i_group],4))

        logger.dumpkvs()

        if (epoch == args.epochs-1) and (args.perf_file != ''):
            if os.path.exists(lastep_global_perf_file):
                with open(lastep_global_perf_file, 'a') as f:
                    acc_head_str = ', '.join([f'{x:.4f}' for x in acc_groups])
                    auc_head_str = ', '.join([f'{x:.4f}' for x in auc_groups])
                    sensitivity_head_str = ', '.join([f'{x:.4f}' for x in sensitivity_groups])
                    specificity_head_str = ', '.join([f'{x:.4f}' for x in specificity_groups])
                    path_str = f'{args.result_dir}'
                    f.write(f'{best_ep}, {es_acc:.4f}, {best_acc:.4f}, {acc_head_str}, {es_auc:.4f}, {test_auc:.4f}, {auc_head_str}, {best_f1:.4f}, {es_sensitivity:.4f}, {test_sensitivity:.4f}, {sensitivity_head_str}, {es_specificity:.4f}, {test_specificity:.4f}, {specificity_head_str}, {tst_other_metrics[1]:.4f}, {tst_other_metrics[2]:.4f}, {tst_other_metrics[3]:.4f}, {tst_other_metrics[4]:.4f}, {path_str}\n')

    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:
                acc_head_str = ', '.join([f'{x:.4f}' for x in best_acc_groups])
                auc_head_str = ', '.join([f'{x:.4f}' for x in best_auc_groups])
                sensitivity_head_str = ', '.join([f'{x:.4f}' for x in best_sensitivity_groups])
                specificity_head_str = ', '.join([f'{x:.4f}' for x in best_specificity_groups])

                sen_at_diff_spe_str = ''
                for i in range(len(name_sen_at_diff_spe)):
                    sen_at_diff_spe_str += f'{best_sen_at_diff_spe[i]:.4f}, '
                    for x in range(len(samples_per_attr)):
                        sen_at_diff_spe_str += f'{best_sen_at_diff_spe_groups[x][i]:.4f}, '

                path_str = f'{args.result_dir}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_es_acc:.4f}, {best_acc:.4f}, {acc_head_str}, {best_es_auc:.4f}, {best_wgd_auc:.4f}, {best_auc:.4f}, {auc_head_str}, {best_wgd_f1:.4f}, {best_f1:.4f}, {best_es_sensitivity:.4f}, {best_sensitivity:.4f}, {sensitivity_head_str}, {best_es_specificity:.4f}, {best_specificity:.4f}, {specificity_head_str}, {best_tst_other_metrics[1]:.4f}, {best_tst_other_metrics[2]:.4f}, {best_tst_other_metrics[3]:.4f}, {best_tst_other_metrics[4]:.4f}, {sen_at_diff_spe_str} {path_str}\n')
                # f.write('epoch, acc, auc, dpd, dpr, eod, eor, path\n')

    os.rename(args.result_dir, f'{args.result_dir}_{args.seed}_auc{best_auc:.4f}')
