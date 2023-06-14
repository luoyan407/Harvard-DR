import sys, os
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import random
import csv
import pickle
import statsmodels.api as sm
from datetime import datetime
import scipy.stats as stats
from skimage.transform import resize
from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

def find_all_files_(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

def find_all_files(folder, str_pattern='*.npz'):
    files = [os.path.basename(y) for x in os.walk(folder) for y in glob(os.path.join(x[0], str_pattern))]
    return files

def get_all_pids(data_dir):
    pids = []
    dict_pid_fid = {}
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        pid = raw_data['pid'].item()
        pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
    # pids=list(set(pids))
    # pids.sort()
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid

def get_all_pids_filter(data_dir, keep_list=None):
    race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

    pids = []
    dict_pid_fid = {}
    files = []
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        # race = int(raw_data['race'].item())
        race = raw_data['race'].item()
        if keep_list is not None and race not in keep_list:
            continue

        if not hasattr(raw_data, 'pid'):
            pid = f[f.find('_')+1:f.find('.')]
        else:
            pid = raw_data['pid'].item()
            pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
        files.append(f)
    # pids=list(set(pids))
    # pids.sort()
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid, files

def vf_to_matrix(vec, fill_in=-50):
    mat = np.empty((8,9))
    mat[:] = fill_in # np.nan

    mat[0, 3:7] = vec[0:4]
    mat[1, 2:8] = vec[4:10]
    mat[2, 1:] = vec[10:18]
    mat[3, :7] = vec[18:25]
    mat[3, 8] = vec[25]
    mat[4, :7] = vec[26:33]
    mat[4, 8] = vec[33]
    mat[5, 1:] = vec[34:42]
    mat[6, 2:8] = vec[42:48]
    mat[7, 3:7] = vec[48:52]

    # mat = np.rot90(mat, k=1).copy()

    return mat

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class Harvard_Diabetic_Retinopathy(Dataset):
    # subset: train | val | test 
    def __init__(self, data_path='./data/', split_file='', subset='train', modality_type='rnflt', task='md', resolution=224, need_shift=True, stretch=1.0,
                    depth=1, indices=None, attribute_type='race', transform=None, needBalance=False, split_ratio=-1, min_balance=False):

        self.data_path = data_path
        self.modality_type = modality_type
        self.subset = subset
        self.task = task
        self.attribute_type = attribute_type
        self.transform = transform
        self.needBalance = needBalance
        self.min_balance = min_balance

        self.data_files = find_all_files(self.data_path, f'{subset}_*.npz')
        if indices is not None:
            self.data_files = [self.data_files[i] for i in indices]
        if split_ratio > 0:
            random.shuffle(self.data_files)
            self.data_files = self.data_files[:int(len(self.data_files)*split_ratio)]

        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []

        self.normalize_vf = 30.0

        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        if self.subset == 'train' and self.needBalance:
            for idx in range(0, len(self.data_files)):
                rnflt_file = os.path.join(self.data_path, self.data_files[idx])
                raw_data = np.load(rnflt_file, allow_pickle=True)
                cur_label = raw_data['race'].item()
                if cur_label not in self.label_samples:
                    self.label_samples[cur_label] = list()
                self.label_samples[cur_label].append(self.data_files[idx])
                self.balanced_max = len(self.label_samples[cur_label]) \
                    if len(self.label_samples[cur_label]) > self.balanced_max else self.balanced_max
            ttl_num_samples = 0
            self.class_samples_num = [0]*len(list(self.label_samples.keys()))
            for i, (k,v) in enumerate(self.label_samples.items()):
                self.class_samples_num[int(k)] = len(v)
                ttl_num_samples += len(v)
                print(f'{k}-th identity training samples: {len(v)}')
            print(f'total number of training samples: {ttl_num_samples}')
            self.class_samples_num = np.array(self.class_samples_num)

            # Oversample the classes with fewer elements than the max
            for i_label in self.label_samples:
                while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
                    self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))
            
            data_files = []
            for i, (k,v) in enumerate(self.label_samples.items()):
                data_files = data_files + v
            self.data_files = data_files

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        rnflt_file = os.path.join(self.data_path, self.data_files[item])
        sample_id = self.data_files[item][:self.data_files[item].find('.')]
        raw_data = np.load(rnflt_file, allow_pickle=True)

        if self.modality_type == 'fundus':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            rnflt_sample = np.transpose(rnflt_sample, (2, 0, 1))
            data_sample = rnflt_sample.astype(np.float32)
        elif self.modality_type == 'rpet':
            rnflt_sample = raw_data['macular']
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
        elif 'bscan' in self.modality_type:
            oct_img = raw_data['oct_bscans']
            oct_img_array = []
            for img in oct_img:
                oct_img_array.append(resize(img, (self.resolution, self.resolution)))
            data_sample = np.stack([oct_img_array]*(1), axis=0)
            data_sample = data_sample.squeeze().astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample).float()

        y = torch.tensor(float(raw_data['dr_class'].item()))

        attr = 0
        if self.attribute_type == 'race':
            attr = raw_data['race'].item()
            attr = torch.tensor(attr).int()
        elif self.attribute_type == 'gender':
            attr = torch.tensor(raw_data['male'].item()).int()
        elif self.attribute_type == 'hispanic':
            attr = torch.tensor(raw_data['hispanic'].item()).int()
        elif self.attribute_type == 'maritalstatus':
            attr = torch.tensor(raw_data['maritalstatus'].item()).int()
        elif self.attribute_type == 'language':
            attr = torch.tensor(raw_data['language'].item()).int()
            

        return data_sample, y, attr

def load_data_(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, isHAVO=0, subset='train'
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not isHAVO:
        if not data_dir:
            raise ValueError("unspecified data directory")
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    elif isHAVO == 1:
        dataset = HAVO(data_dir, subset=subset, resolution=image_size)
    elif isHAVO == 2:
        dataset = HAVO_RNFLT(data_dir, subset=subset, resolution=image_size)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict
