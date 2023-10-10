import pdb

import torchvision.transforms as transforms
from lib.dataset.cocodataset import CoCoDataset

from lib.utils.cutout import SLCutoutPIL
from torchvision.transforms import RandAugment
import os.path as osp
from .t50_dataset import *


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((250, 250)),
                                 transforms.ColorJitter(hue=(-0.5, 0.5)),
                                 transforms.RandomCrop(224),
                                 transforms.RandomVerticalFlip(p=0.2),
                                 transforms.RandomHorizontalFlip(p=0.2),
                                 transforms.RandomAutocontrast(),
                                 # RandAugment(num_ops=2, magnitude=2),
                                 transforms.ToTensor(),
                                 normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    # test_data_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
    #                                           transforms.ToTensor(),
    #                                           normalize])

    test_data_transform = transforms.Compose([transforms.Resize((250, 250)),
                                              transforms.CenterCrop((224, 224)),
                                              transforms.ToTensor(),
                                              normalize])

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )
        print("len(train_dataset):", len(train_dataset))
        print("len(val_dataset):", len(val_dataset))
        return train_dataset, val_dataset

    elif args.dataname == 'cholect50':
        train_dataset, train_idx, val_dataset, val_idx = get_data(data_path=args.pkl,
                                                                  train_transform=train_data_transform,
                                                                  val_transform=test_data_transform)
        return train_dataset, train_idx, val_dataset, val_idx
    # cholect50_sam
    elif args.dataname == 'cholect50_sam':
        train_dataset, train_idx, val_dataset, val_idx = get_data_sam(data_path=args.pkl,
                                                                  train_transform=train_data_transform,
                                                                  val_transform=test_data_transform)
        return train_dataset, train_idx, val_dataset, val_idx
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)


