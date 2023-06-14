import sys, os

import blobfile as bf
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torchvision.models import *
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, StepLR

from sklearn.metrics import *


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item() # [0, batch_size]
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy

def num_to_onehot(nums):
    # nums = [1, 0, 3]
    nums = nums.astype(int)
    n_values = np.max(nums) + 1
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec

def multiclass_demographic_parity(pred_prob, y, attrs):
    # https://github.com/wbawakate/fairtorch/blob/master/examples/demo.ipynb
    # expected value of y_pred = 1 for each sensitive group
    # E[h(X)|A=a]
    # correspond to demographic parity

    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred==i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        # pred_by_attrs = []
        # for j in attrs_set:
        #     idx = attrs==j
        #     pred_by_attrs.append(tmp_preds[idx].mean().item())
        # mc_dpd += np.max(pred_by_attrs) - np.min(pred_by_attrs)

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs==j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]
        
    return mc_dpd

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0]>2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)
    
    return AUC

def auc_score_(pred_prob, y, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    # specificity: 80% 85% 90% 95%
    # FPR: 20% 15% 10% 5%
    sensitivity_at_diff_specificity = [-1]*4
    if np.unique(y).shape[0]>2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        for i in range(len(fpr)):
            cur_fpr = fpr[i]
            cur_tpr = tpr[i]
            if np.abs(cur_fpr-0.2) <= eps:
                sensitivity_at_diff_specificity[0] = cur_tpr
            if np.abs(cur_fpr-0.15) <= eps:
                sensitivity_at_diff_specificity[1] = cur_tpr
            if np.abs(cur_fpr-0.1) <= eps:
                sensitivity_at_diff_specificity[2] = cur_tpr
            if np.abs(cur_fpr-0.05) <= eps:
                sensitivity_at_diff_specificity[3] = cur_tpr
        AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def auc_score_multiclass(pred_prob, y, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    # specificity: 80% 85% 90% 95%
    # FPR: 20% 15% 10% 5%
    # if np.unique(y).shape[0]>2:
    #     AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    # else:
    sensitivity_at_diff_specificity = [-1]*4
    y_onehot = num_to_onehot(y)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr-0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr-0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr-0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr-0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def compute_sensitivity_specificity(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    tn, fp, fn, tp = confusion_matrix(y, np.rint(pred_prob)).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)

    return sensitivity, specificity

def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]
        acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc

def equity_scaled_AUC(output, target, attrs, alpha=1.):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    fpr, tpr, thresholds = roc_curve(target, output)
    overall_auc = auc(fpr, tpr)
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
        group_auc = auc(fpr, tpr)

        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    return es_auc

def equity_scaled_sensitivity_specificity(output, target, attrs, alpha=1.):
    es_auc = 0
    identity_wise_sens = []
    identity_wise_spec = []
    identity_wise_num = []
    sensitivity, specificity = compute_sensitivity_specificity(output, target)
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        sensitivity, specificity = compute_sensitivity_specificity(pred_group, gt_group)

        identity_wise_sens.append(sensitivity)
        identity_wise_spec.append(specificity)
        identity_wise_num.append(gt_group.shape[0])

    tmp = 0
    for i in range(len(identity_wise_sens)):
        tmp += np.abs(identity_wise_sens[i]-sensitivity)
    es_sensitivity = (sensitivity / (alpha*tmp + 1))

    tmp = 0
    for i in range(len(identity_wise_spec)):
        tmp += np.abs(identity_wise_spec[i]-specificity)
    es_specificity = (specificity / (alpha*tmp + 1))

    return es_sensitivity, es_specificity

def get_num_by_group(train_dataset_loader, n_group=3):
    attr_num_dict = {}
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        attr_array = attr.detach().cpu().numpy().tolist()
        for j in attr_array:
            if j not in attr_num_dict:
                attr_num_dict[j] = 1
            else:
                attr_num_dict[j] = attr_num_dict[j] + 1
    all_attrs = list(attr_num_dict.keys())
    all_attrs.sort(reverse=False)
    samples_per_attr = []
    for x in all_attrs:
        samples_per_attr.append(attr_num_dict[x])

    return all_attrs, samples_per_attr

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class Rescaled_Softsign(nn.Module):
    def __init__(self, rescale=1.):
        super().__init__()
        self.rescale = rescale
        self.acti_func = nn.Softsign()

    def forward(self, x):
        y = self.acti_func(x)
        y = y*self.rescale
        return y

class Rescaled_Sigmoid(nn.Module):
    def __init__(self, rescale=1.):
        super().__init__()
        self.rescale = rescale
        self.acti_func = nn.Sigmoid()

    def forward(self, x):
        y = self.acti_func(x)-0.5
        y = y*self.rescale
        return y

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

class General_Logistic(nn.Module):
    def __init__(self, min_val=-38.0, max_val=26.0): # , th=-1.
        super().__init__()
        self.min_nml_val = min_val # -1.46153846154
        self.max_nml_val = max_val # 1

        self.A = self.min_nml_val
        self.K = self.max_nml_val
        self.C = 1
        self.Q = 1
        self.B = 1

        self.nu = np.log2(self.C+self.Q*np.exp(1))/np.log2((self.K-self.A)/(-self.A))
        
    def forward(self, x):
        out = self.A + (self.K-self.A)/(self.C+self.Q*torch.exp( -self.B * x ))**(1/self.nu)
        return out

class Attribute_Grouped_Normalizer(nn.Module):
    def __init__(self, num_attr=0, dim=0, mus=None, sigmas=None, momentum=0.9):
        super().__init__()
        self.num_attr = num_attr
        self.dim=0
        self.mus = mus
        self.sigmas = sigmas
        self.eps = 1e-6
        self.momentum = momentum

    def forward(self, x, attr):
        if self.mus is None:
            self.mus = []
            for i in range(self.num_attr):
                self.mus.append(torch.zeros(x.shape[1]).type(x.type()))
        if self.sigmas is None:
            self.sigmas = []
            for i in range(self.num_attr):
                self.sigmas.append(torch.ones(x.shape[1]).type(x.type()))
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx]])/(self.sigmas[attr[idx]] + self.eps)
        
        return x

    # @torch.no_grad()
    def update_mus_sigmas(self, mus, sigmas):
        if self.momentum >= 0 and self.momentum < 1:
            for i in range(self.num_attr):
                self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
                self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            out_str = ', '.join([f'G{i}: ({torch.mean(m).item():f}, {torch.mean(s).item():f})' for i, (m, s) in enumerate(zip(self.mus, self.sigmas))])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

def forward_model_with_fin(model, data, attr):
    feat = model[0](data)
    if type(model[1]).__name__ != 'Fair_Identity_Normalizer':
        nml_feat = model[1](feat)
    else:
        nml_feat = model[1](feat, attr)
    logit = model[2](nml_feat)
    return logit, feat

class Fair_Identity_Normalizer(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1+torch.exp(self.sigmas[attr[idx], :])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Multiple_Binary_Classifiers(nn.Module):
    def __init__(self, num_attr=1, model_type=None, in_dim=1, out_dim=1):
        super().__init__()
        self.num_attr = num_attr
        self.model_type = model_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.classifiers = nn.ModuleList([create_model(model_type=model_type, in_dim=in_dim, out_dim=out_dim) for i in range(self.num_attr)])

    def forward(self, x, attr):
        ret = []
        for i in range(self.classifiers.__len__()):
            ret.append(self.classifiers[i](x))
        ret = torch.cat(ret, dim=1)
        ret = ret.gather(1, attr.long().view(-1,1))
        return ret

class Learnable_BatchNorm1d(nn.Module):
    def __init__(self, dim=0, mu=0, sigma=0.1, momentum=0.9):
        super().__init__()
        self.dim = dim
        self.mus = nn.Parameter(torch.ones(1, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.ones(1, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum

    def forward(self, x):
        for idx in range(x.shape[0]):
            x = (x - self.mus)/( torch.log(1+torch.exp(self.sigmas)) + self.eps)
        
        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma)
            mu = torch.mean(self.mus)
            out_str = f'Learnable BatchNorm: ({mu:f}, {sigma:f})'
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class MD_Mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 52
        self.out_features = 1
        
        weight = [0.010346387,0.010663622,0.010032727,0.007129987,0.014017274,0.018062957,0.018842243,0.016647837,0.015124109,0.011389459,0.014017678,0.022160035,0.02378898,0.02383174,0.02191793,0.019983033,0.0159671,0.0115242,0.007463015,0.018917531,0.023298082,0.02881147,0.027520778,0.025385285,0.023138773,0.016495131,0.008567998,0.017318338,0.028689633,0.02881154,0.028483851,0.025037148,0.023584995,0.016130119,0.015494349,0.024661184,0.028129123,0.028682529,0.026372951,0.024033034,-0.001105303,0.016997128,0.01889403,0.023627078,0.024890497,0.023402898,0.0218989,0.017713769,0.015848428,0.018916324,0.018597527,0.019021584]
        bias = 0.000592563
        self.weight = torch.nn.Parameter(torch.tensor(weight), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.tensor(bias), requires_grad=False)

    def forward(self, input):
        assert input.shape[1] == self.in_features
        output = input @ self.weight.t() + self.bias
        return output

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

def create_model(model_type='efficientnet', in_dim=1, out_dim=1, use_pretrained=True, include_final=True):
    classification_model = None
    if model_type == 'vit':
        classification_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        classification_model.conv_proj = nn.Conv2d(in_dim, 768, kernel_size=(16, 16), stride=(16, 16))
        if include_final:
            classification_model.heads[0] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
        else:
            classification_model.heads[0] = nn.Identity()
    elif model_type == 'efficientnet':
        load_weights = None
        if use_pretrained:
            load_weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        classification_model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
        if in_dim != 3:
            classification_model.features[0][0] = nn.Conv2d(in_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if include_final:
            classification_model.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=False)
        else:
            classification_model.classifier[1] = nn.Identity()
    elif model_type == 'efficientnet_v2':
        classification_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        if include_final:
            classification_model.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
        else:
            classification_model.classifier[1] = nn.Identity()
    elif model_type == 'resnet':
        classification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        classification_model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if include_final:
            classification_model.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
        else:
            classification_model.fc = nn.Identity()
    elif model_type == 'swin':
        classification_model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        classification_model.features[0][0] = nn.Conv2d(in_dim, 128, kernel_size=(4, 4), stride=(4, 4))
        classification_model.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
    elif model_type == 'vgg':
        classification_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        classification_model.features[0] = nn.Conv2d(in_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        classification_model.classifier[6] = nn.Linear(in_features=4096, out_features=out_dim, bias=True)
    elif model_type == 'resnext':
        classification_model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        classification_model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        classification_model.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'wideresnet':
        classification_model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        classification_model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        classification_model.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'convnext':
        classification_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        classification_model.features[0][0] = nn.Conv2d(in_dim, 96, kernel_size=(4, 4), stride=(4, 4))
        classification_model.classifier[2] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    elif model_type == 'yanet':
        classification_model = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.SiLU(inplace=True),
                                    nn.MaxPool2d((2,2)),
                                    nn.Conv2d(32, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                    nn.AdaptiveAvgPool2d(output_size=1),
                                    nn.Flatten(),
                                    nn.Dropout(p=0.2,inplace=True),
                                    nn.Linear(in_features=512, out_features=out_dim, bias=True))
    return classification_model

class Classifier_Eval_Mode(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1, out_dim=1, use_pretrained=True, include_final=True):
        super().__init__()
    
        self.backbone = None
        self.head = None

        classification_model = None
        if model_type == 'vit':
            classification_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            classification_model.conv_proj = nn.Conv2d(in_dim, 768, kernel_size=(16, 16), stride=(16, 16))
            classification_model.heads[0] = nn.Identity()
            self.backbone = classification_model
            self.head = nn.Linear(in_features=768, out_features=out_dim, bias=True)
        elif model_type == 'efficientnet':
            load_weights = None
            if use_pretrained:
                load_weights = EfficientNet_B1_Weights.IMAGENET1K_V2
            classification_model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
            if in_dim != 3:
                classification_model.features[0][0] = nn.Conv2d(in_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            classification_model.classifier[1] = nn.Identity()

            self.backbone = classification_model
            self.head = nn.Linear(in_features=1280, out_features=out_dim, bias=False)
        elif model_type == 'resnet':
            classification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            classification_model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            classification_model.fc = nn.Identity()

            self.backbone = classification_model
            self.head = nn.Linear(in_features=2048, out_features=out_dim, bias=True)

    def forward(self, x):
        x_lin = self.backbone(x)
        y = self.head(x_lin)

        return y, x_lin


class Model_With_Time(nn.Module):
    def __init__(self, encoder=None, bias=True):
        super(Model_With_Time, self).__init__()
        self.encoder = encoder
        self.bias = bias
        self.classifier = nn.Linear(in_features=2, out_features=1, bias=self.bias)

    def forward(self, x, t):
        x_feat = self.encoder(x)
        x_feat = torch.cat((x_feat, t), dim=1)
        x_out = self.classifier(x_feat)
        return x_out

class OphBackbone_concat(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            # cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            # cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # cur_encoder.classifier[1] = nn.Identity()

            cur_encoder = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
            cur_encoder.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=False)

            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)
        # self.encoder.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)

        self.linear = nn.Linear(in_features=self.unit_feat_dim*self.in_dim, out_features=1, bias=True)

    def forward(self, x):
        x_out = []
        for i, l in enumerate(self.encoders):
            x_out.append(self.encoders[i](x[:,i:i+1,:,:]))
        x_out = torch.cat(x_out, dim=1)
        y = self.linear(x_out)
        return y

class OphBackbone(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1, coef=1.):
        super(OphBackbone, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280
        self.coefs = [1., coef]

        encoders = []
        for i in range(self.in_dim):
            # cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            # cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)

            cur_encoder = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
            cur_encoder.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=False)
            
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        x_out = None
        for i, l in enumerate(self.encoders):
            if x_out is None:
                x_out = self.coefs[i] * self.encoders[i](x[:,i:i+1,:,:])
            else:
                x_out += self.coefs[i] * self.encoders[i](x[:,i:i+1,:,:])
        y = x_out
        return y

class OphBackbone_(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone_, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)
        # self.encoder.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)

        # self.linear = nn.Linear(in_features=self.in_dim, out_features=1, bias=True)

    def forward(self, x):
        x_0 = self.encoders[0](x[:,0:0+1,:,:])
        x_1 = self.encoders[1](x[:,1:1+1,:,:])
        y = x_0 + x_1/(1+torch.abs(x_0))
        return y

class OphBackbone_Multiply(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone_, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        x_out = None
        for i, l in enumerate(self.encoders):
            if x_out is None:
                x_out = self.encoders[i](x[:,i:i+1,:,:])
            else:
                x_out *= self.encoders[i](x[:,i:i+1,:,:])
        y = x_out
        return y

class GlauClassifier(nn.Module):
    def __init__(self, ):
        super(GlauClassifier, self).__init__(model_type='efficientnet', in_dim=1, out_dim=1)
        self.model_type = model_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rnflt_encoder = create_model_(model_type=model_type, in_dim=in_dim, out_dim=out_dim)
        out_feat = -1
        if model_type == 'efficientnet':
            out_feat = 1280
        self.tds_encoder = nn.Sequential(nn.Linear(in_features=52, out_features=128, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=128, out_features=512, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=out_feat, bias=False))
        self.classifier = nn.Linear(in_features=out_feat*2, out_features=out_dim, bias=True)

    def forward(self, x, tds):
        rnflt_feat = self.rnflt_encoder(x)
        tds_feat = self.tds_encoder(tds)
        in_feat = torch.cat((rnflt_feat,tds_feat))
        y_hat = self.classifier(in_feat)
        return y_hat

class Model_Wrapper():
    """docstring for Model_Wrapper"""
    def __init__(self, model,
                result_dir='.',
                resume_checkpoint=None,
                ema_rate=0.9,
                identifier='predictor',
                logger=None):
        super().__init__()
        self.model = model
        self.model_params = list(model.parameters())


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())

def classify_glaucoma(mds, num_class=2, th=-1):
    if num_class==3:
        borderline = np.where((mds>=-3.0) & (mds<-1.0), np.ones_like(mds), np.zeros_like(mds))
        non_glau = np.where(mds>=-1.0, np.ones_like(mds), np.zeros_like(mds))
        y_pred = borderline + non_glau * 2
    elif num_class==2:
        y_pred = np.where(mds>=th, np.ones_like(mds), np.zeros_like(mds))
    return y_pred

def classify(prob):
    y = (prob>=0.5).astype(float)
    return y

def to_one_hot_vector(arr):
    arr = arr.astype(int)
    shape = (arr.shape[0], arr.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(arr.shape[0])
    one_hot[rows, arr] = 1
    return one_hot

def compute_weight(weight, step, rampup_step=4000):
    return weight if rampup_step == 0 or step > rampup_step else weight * step / rampup_step

#=====> lr schedulers

# https://github.com/godofpdog/ViT_PyTorch/blob/af086058764e55a48043db7f6f7c32b685db9427/vit_pytorch/solver.py
def get_scheduler(optimizer, args):
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, args.t_max, args.eta_min)
        elif args.scheduler == 'step':
            return StepLR(optimizer, args.step_size, args.gamma)
        elif args.scheduler == 'exp':
            return ExponentialLR(optimizer, args.gamma)
        else:
            raise ValueError('Invalid scheduler.')
    else:
        return ConstantScheduler(optimizer)

# https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
#<===== lr schedulers