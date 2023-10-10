import pdb

import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
import pandas as pd
import pickle


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_useful_start_idx(sequence_length, list_each_length, ol, ds):
    # 统计每个动作的持续帧数，确定ds和ol
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        end = list_each_length[i] - (sequence_length-1) * ds
        for j in range(count, count + end, ol):
            idx.append(j)
        count += list_each_length[i]
    # print('idx', idx)

    train_idx = []
    for i in range(len(idx)):
        for j in range(sequence_length):
            train_idx.append(idx[i] + j * ds)
    # print('train idx ', train_idx)
    return train_idx


class CholecT50(data.Dataset):
    def __init__(self, file_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels
        self.transform = transform
        self.loader = loader

        # classes_file='./lists/cholect50/t50_labels_101.csv',
        # classes_i_file='./lists/cholect50/t50_labels_i.csv',
        # classes_v_file='./lists/cholect50/t50_labels_v.csv',
        # classes_t_file='./lists/cholect50/t50_labels_t.csv',
        # classes_vt_file='./lists/cholect50/t50_labels_vt.csv'):

        # self.classes_file = classes_file
        # self.classes_i_file = classes_i_file
        # self.classes_v_file = classes_v_file
        # self.classes_t_file = classes_t_file
        # self.classes_vt_file = classes_vt_file

        # self.class_dir = class_dir
        # with open(self.class_dir, 'r') as f:
        #     self.classes = json.load(f)
        #     self.classes = {int(k): v for k, v in self.classes.items()}
    # @property
    # def classes(self):
    #     classes_all = pd.read_csv(self.classes_file)
    #     return classes_all.values.tolist()
    #
    # @property
    # def ins_classes(self):
    #     classes_all = pd.read_csv(self.classes_i_file)
    #     return classes_all.values.tolist()
    #
    # @property
    # def verb_classes(self):
    #     classes_all = pd.read_csv(self.classes_v_file)
    #     return classes_all.values.tolist()
    #
    # @property
    # def target_classes(self):
    #     classes_all = pd.read_csv(self.classes_t_file)
    #     return classes_all.values.tolist()
    #
    # @property
    # def vt_classes(self):
    #     classes_all = pd.read_csv(self.classes_vt_file)
    #     return classes_all.values.tolist()

    def __getitem__(self, index):

        img_name = self.file_paths[index]
        imgs = self.loader(img_name)
        # imgs = self.resize(imgs)
        if self.transform is not None:
            imgs = self.transform(imgs)
        labels = self.file_labels[index]  # [1,0,1,0,...]  len=num_classes.txt
        return imgs, labels


    # def __getitem__(self, index):
    #     img_name = self.file_paths[index]
    #     imgs = self.loader(img_name)
    #     if self.transform is not None:
    #         imgs = self.transform(imgs)
    #     labels = self.file_labels[index]  # [1,0,1,0,...]  len=num_classes.txt
    #     return imgs, labels

    def __len__(self):
        return len(self.file_paths)


class CholecT50_SAM(data.Dataset):
    def __init__(self, file_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.file_paths[index]
        imgs = self.loader(img_name)
        imgs_np = np.array(imgs)
        if self.transform is not None:
            imgs = self.transform(imgs)
        labels = self.file_labels[index]  # [1,0,1,0,...]  len=num_classes.txt
        return imgs, labels, imgs_np

    def __len__(self):
        return len(self.file_paths)


def get_data_sam(data_path, train_transform, val_transform, soft=False):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]

    train_labels = train_test_paths_labels[2]
    val_labels = train_test_paths_labels[3]

    train_num_each = train_test_paths_labels[4]
    val_num_each = train_test_paths_labels[5]

    if soft:
        train_labels = np.asarray(train_labels)
    else:
        train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    train_dataset = CholecT50_SAM(train_paths, train_labels, train_transform)
    val_dataset = CholecT50_SAM(val_paths, val_labels, val_transform)

    return train_dataset, train_num_each, val_dataset, val_num_each


def get_data(data_path, train_transform, val_transform, soft=False):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]

    train_labels = train_test_paths_labels[2]
    val_labels = train_test_paths_labels[3]

    train_num_each = train_test_paths_labels[4]
    val_num_each = train_test_paths_labels[5]

    if soft:
        train_labels = np.asarray(train_labels)
    else:
        train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    train_dataset = CholecT50(train_paths, train_labels, train_transform)
    val_dataset = CholecT50(val_paths, val_labels, val_transform)

    return train_dataset, train_num_each, val_dataset, val_num_each

