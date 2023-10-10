import argparse
import math
import os, sys
import pdb
import random
import datetime
import shutil
import time
from typing import List
import json
import numpy as np
from copy import deepcopy
import yaml
from dotmap import DotMap

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12346'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4'
from torch import Tensor
from typing import Sequence, List

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DistributedSampler, Sampler

import _init_paths
from lib.dataset.get_dataset import get_datasets

from lib.utils.logger import setup_logger
import lib.models as models
import lib.models.aslloss
from lib.models.time_transformer_clip import *
from lib.models.time_transformer_aim import *
# from lib.models.model_aim import *
from lib.utils.metric import voc_mAP
from lib.utils.misc import clean_state_dict
from lib.utils.slconfig import get_raw_dict

from lib.utils.loss_cl import *

import ivtmetrics.recognition as ivt_metrics

recognize = ivt_metrics.Recognition(num_class=100)
recognize_att = ivt_metrics.Recognition(num_class=100)

import warnings

warnings.filterwarnings("ignore")


def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label MSCOCO Training')
    parser.add_argument('--dataname', help='dataname', default='cholect50', choices=['coco14'])
    parser.add_argument('--pkl', default='./lists/cholect50/cholect50_classes_101.pkl')
    parser.add_argument('--text_features', default='./text_features/bioclip_phrase_cls101.npy')
    parser.add_argument('--loss_text', default=0.1)
    parser.add_argument('--model_weight', default=None)

    parser.add_argument('--model', default='task_ft', type=str, choices=['attention','ft_only', 'aim_only', 'task_only',
                                                                         'spatial_only', 'task_ft', 'task_spatial'])
    parser.add_argument('--attention_type', default='time', type=str,
                        choices=['time', 'space', 'divided_space_time', 'joint_space_time'])

    parser.add_argument('--dataset_dir', help='dir of dataset', default='/comp_robot/liushilong/data/COCO14/')
    parser.add_argument('--img_size', default=448, type=int, help='size of input images')

    parser.add_argument('--num_frames', default=10, type=int, help='sequence length')
    parser.add_argument('--ds', default=1, type=int, help='down sample rate')
    parser.add_argument('--ol', default=10, type=int, help='overlap rate')

    parser.add_argument('--output', metavar='DIR', help='path to output folder')
    parser.add_argument('--num_class', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='use class weighted')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'], help='which optim to use')

    # asy loss
    parser.add_argument('--eps', default=1e-5, type=float, help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, help='disable_torch_grad_focal_loss in asl')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                        help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',  # 存在evaluate中
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')

    # distribution training
    # parser.add_argument('--world-size', default=-1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')  # 140
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')  # 124

    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=0,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ')

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=['sine', 'learned'],
                        help="Type of positional embedding to use on top of the image features")

    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")

    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * raining
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


best_mAP = 0


def main():
    args = get_args()

    if args.seed is not None:
        # pdb.set_trace()
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url)
    args.rank = torch.distributed.get_rank()
    torch.cuda.set_device(args.rank)

    cudnn.benchmark = True
    #
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: " + ' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    # logger.info('local_rank: {}'.format(args.local_rank))

    return main_worker(args, logger)


def main_worker(args, logger):
    global best_mAP

    #     parser.add_argument('--model', type=str, choices=['backbone', 'spatio_temporal', 'st_mt'],
    #                         help='which model to use')
    #     parser.add_argument('--model_temporal', type=str, choices=['meanP', 'LSTM', 'Conv', 'transf'],
    #                         help='which temporal model to use')

    if args.model == 'ft_only':
        model = build_general_only(general='ft')
    elif args.model == 'aim_only':
        model = build_general_only(general='aim')
    elif args.model == 'spatial_only':
        model = build_general_only(general='spatial')
    elif args.model == 'task_only':
        model = build_task_only(args, hidden_dim=512)
    elif args.model == 'task_ft':
        model = build_task_general(args, general='ft')
    elif args.model == 'task_spatial':
        model = build_task_general(args, general='spatial')
    elif args.model == 'attention':
        model = build_attention_ablation(args)
    else:
        raise ValueError('no model!')

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    num_params_grad = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params_grad += p.numel()
    print(f"Total number of param with grad: {num_params_grad}")

    pdb.set_trace()

    shutil.copy('./train_moe.py', args.output)
    shutil.copy('./lib/models/time_transformer_aim.py', args.output)
    shutil.copy('./lib/models/model_aim.py', args.output)

    # if args.model_weight is not None:
    #     checkpoint = torch.load(args.model_weight, map_location='cpu')['state_dict']
    #     msg = model.load_state_dict(checkpoint, strict=False)
    #     print(f"Loaded model with msg: {msg}")
        # for name, param in model.named_parameters():
        #     param.requires_grad = True

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    #
    # pdb.set_trace()

    local_rank = int(os.environ["LOCAL_RANK"])

    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay)  # 0.9997
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    # criterion
    # criterion = models.aslloss.AsymmetricLossOptimized(
    #     gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
    #     clip=args.loss_clip,
    #     disable_torch_grad_focal_loss=args.dtgfl,
    #     eps=args.eps,
    # )

    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    # pdb.set_trace()
    args.lr_mult = args.batch_size / 256  # 1.0
    base_lr = args.lr
    if args.optim == 'AdamW':
        # pdb.set_trace()
        # param_dicts = [
        #     {"params": [p for n, p in model.module.named_parameters() if 'clip' in n and p.requires_grad],
        #      "lr": base_lr * 0.1},
        #     {"params": [p for n, p in model.module.named_parameters() if 'clip' not in n and p.requires_grad],
        #      "lr": base_lr}
        # ]
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad],
             "lr": base_lr}
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else:
        raise NotImplementedError

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            # import ipdb; ipdb.set_trace()
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            # model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code

    train_dataset, train_num_each, val_dataset, val_num_each = get_datasets(args)

    if args.weighted:
        class_weights = [1] * 101
        class_weights[17] = 0.3
        class_weights[19] = 0.5
        class_weights[60] = 0.4
        train_sampler = WeightedRandomSamplerDDP(train_dataset, class_weights, num_replicas=dist.get_world_size(),
                                                 rank=dist.get_rank(), num_samples=len(train_dataset))
    else:
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_sampler = Idx_DistributedSampler(train_dataset, args.num_frames, train_num_each, args.ds, args.ol,
                                               shuffle=True)

    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'

    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_sampler = Idx_DistributedSampler(val_dataset, args.num_frames, val_num_each, ds=1, ol=1, shuffle=False)

    shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    # if args.evaluate:
    #     # Todo 重新validate
    #     _, mAP = validate(val_loader, model, criterion, args, logger, recognize)
    #     logger.info(' * mAP {mAP:.5f}'.format(mAP=mAP))
    #     return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, mAPs, losses_ema, mAPs_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    torch.cuda.empty_cache()

    # print(train_loader.sampler.indices)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        # print(train_loader.sampler.indices)
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            # summary_writer.add_scalar('train_acc1', acc1, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:
            # pdb.set_trace()
            # evaluate on validation set
            loss, mAP = validate(val_loader, model, criterion, args, logger, recognize, val_num_each)
            loss_ema, mAP_ema = validate(val_loader, ema_m.module, criterion, args, logger, recognize, val_num_each)
            losses.update(loss)
            mAPs.update(mAP)
            losses_ema.update(loss_ema)
            mAPs_ema.update(mAP_ema)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            regular_mAP_list.append(mAP)
            ema_mAP_list.append(mAP_ema)

            progress.display(epoch, logger)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_loss', loss, epoch)
                summary_writer.add_scalar('val_mAP', mAP, epoch)
                summary_writer.add_scalar('val_loss_ema', loss_ema, epoch)
                summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

            # remember best (regular) mAP and corresponding epochs
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
            if mAP_ema > best_ema_mAP:
                best_ema_mAP = max(mAP_ema, best_ema_mAP)

            if mAP_ema > mAP:
                mAP = mAP_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = mAP > best_mAP
            if is_best:
                best_epoch = epoch
            best_mAP = max(mAP, best_mAP)

            logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
            logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

            if dist.get_rank() == 0:
                if epoch > 5:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.backbone,
                        'state_dict': state_dict,
                        'best_mAP': best_mAP,
                        'optimizer': optimizer.state_dict(),
                    }, is_best=is_best, epoch=epoch, filename=os.path.join(args.output, 'checkpoint.pth.tar'))
            # filename=os.path.join(args.output, 'checkpoint_{:04d}.pth.tar'.format(epoch))

            if math.isnan(loss) or math.isnan(loss_ema):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.backbone,
                    'state_dict': model.state_dict(),
                    'best_mAP': best_mAP,
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, epoch=epoch, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)

            # early stop
            if args.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                    if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        if dist.get_rank() == 0 and args.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist))
                        break

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()

    return 0


IVT, I, V, T = [], [], [], []
with open('./configs/maps.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0: continue
        v1, v2, v3, v4 = line.strip().split(',')[:4]
        IVT.append(int(v1))
        I.append(int(v2))
        V.append(int(v3))
        T.append(int(v4))
IVT, I, V, T = torch.tensor(IVT), torch.tensor(I), torch.tensor(V), torch.tensor(T)


def label_mapping(label_ivt):
    # label_ivt: [bs, 100]
    label_i = torch.zeros(label_ivt.shape[0], 6)
    # label_i = torch.zeros(label_ivt.shape[0], 7)
    label_v = torch.zeros(label_ivt.shape[0], 10)
    label_t = torch.zeros(label_ivt.shape[0], 15)

    for i in range(label_ivt.shape[0]):
        item = label_ivt[i]
        classes_i = torch.unique(I[torch.nonzero(item)])
        classes_v = torch.unique(V[torch.nonzero(item)])
        classes_t = torch.unique(T[torch.nonzero(item)])

        label_i[i, classes_i] = 1.0
        label_v[i, classes_v] = 1.0
        label_t[i, classes_t] = 1.0

    return label_i, label_v, label_t


def soft_label_mapping(label_ivt):
    # label_ivt: [bs, 100]
    # label_i = torch.zeros(label_ivt.shape[0], 6)
    label_i = label_ivt[:, :7]
    label_v = label_ivt[:, 7:17]
    label_t = label_ivt[:, 17:32]
    label_ivt = label_ivt[:, 32:]

    return label_i, label_v, label_t, label_ivt


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True, reduction="batchmean")):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


def get_text_features(text_features_dict, target):
    # text_features_dict: numpy, [101, 512]
    # target: tensor, [bs, 100]
    b = target.shape[0]
    text_features_array = np.zeros((b, text_features_dict.shape[1]*3))
    text_features = []
    for i in range(b):
        lb = torch.nonzero(target[i])
        if lb.shape[0] != 0:
            tf = np.concatenate([text_features_dict[j] for j in lb])
            text_features.append(tf)
        else:
            text_features.append(text_features_dict[-1])
    for i, arr in enumerate(text_features):
        text_features_array[i, :len(arr)] = arr

    return text_features_array


def normlize_0_1(tensor):
    min_vals, _ = torch.min(tensor, dim=1, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=1, keepdim=True)
    ranges = max_vals - min_vals

    epsilon = torch.tensor(1e-8)
    ranges += epsilon

    normalized_tensor = (tensor - min_vals) / ranges

    return normalized_tensor


classes_longtail_idx = np.load('./dataset_analysis/num_classe_arg_des.npy')


# def tml_mixgen(image, label, M, lam=0.5):
#     for i in range(M):
#         image[i, :] = (lam * image[i, :] + (1-lam) * image[i+M, :])
#         label[i] = torch.max(label[i], label[i+M])
#     return image, label

def tml_mixgen(image, label, M, lam=0.5):
    image[:M, :] = lam * image[:M, :] + (1 - lam) * image[M:2*M, :]
    label[:M] = torch.max(label[:M], label[M:2*M])
    return image, label


def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')

    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()
    t = args.num_frames
    end = time.time()
    # klloss = KLLoss()
    # text_features_dict = np.load(args.text_features)

    for i, (images, target) in enumerate(tqdm(train_loader, total=len(train_loader) // dist.get_world_size())):
        # measure data loading time
        images, target = tml_mixgen(images, target, M=int(images.shape[0] * 0.5))

        data_time.update(time.time() - end)
        target_a = target[t - 1::t, :100]
        target_i, target_v, target_t = label_mapping(target_a)

        images = images.view(-1, t, 3, args.img_size, args.img_size)
        images = images.cuda(non_blocking=True)  # [bs, 3, 448, 448]

        # target = target.cuda(non_blocking=True)  # [bs, 101]
        target_a = target_a.cuda(non_blocking=True).to(torch.float)  # [bs, 100]
        target_i = target_i.cuda(non_blocking=True).to(torch.float)
        target_v = target_v.cuda(non_blocking=True).to(torch.float)
        target_t = target_t.cuda(non_blocking=True).to(torch.float)

        # compute output
        # pdb.set_trace()
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            # pdb.set_trace()

            # loss_task = criterion(output[0], target_a)
            if args.model in ('attention', 'aim_only', 'task_spatial'):
                loss = criterion(output, target_a)
            elif args.model in ('task_only', 'ft_only', 'spatial_only'):
                loss_i = criterion(output[-4], target_i)
                loss_v = criterion(output[-3], target_v)
                loss_t = criterion(output[-2], target_t)
                loss_a = criterion(output[-1], target_a)
                loss = (loss_i + loss_v + loss_t + loss_a) / 4.0
            else:
                loss_clip = criterion(output[-2], target_a)
                loss_joint = criterion(output[-1], target_a)
                loss = (loss_clip + loss_joint) / 2.0

            if args.loss_dev > 0:
                loss *= args.loss_dev

        # record loss

        # print(i, loss.item())

        losses.update(loss.item(), images.size(0))

        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        if epoch >= args.ema_epoch:
            ema_m.update(model)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
        speed_all.update(images.size(0) * dist.get_world_size() / batch_time.val, batch_time.val)

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, criterion, args, logger, recognize, val_num_each):

    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    # Acc1 = AverageMeter('Acc@1', ':5.2f')
    # top5 = AverageMeter('Acc@5', ':5.2f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    # mAP = AverageMeter('mAP', ':5.3f', val_only=)

    recognize.reset_global()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    model.eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        predict_list = []
        predict_att_list = []

        label_list = []
        t = args.num_frames
        for i, (images, target) in enumerate(tqdm(val_loader, total=len(val_loader) // dist.get_world_size())):
            images = images.view(-1, t, 3, args.img_size, args.img_size)
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target = target[t - 1::t, :100]

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):

                if args.model in ('attention', 'aim_only', 'task_spatial'):
                    output = model(images)
                else:
                    output = model(images)[-1]

                loss = criterion(output, target.to(torch.float))
                if args.loss_dev > 0:
                    loss *= args.loss_dev
                output_sm = nn.functional.sigmoid(output)

                if torch.isnan(loss):
                    saveflag = True

            # record loss
            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # save some data
            # output_sm = nn.functional.sigmoid(output)

            # output_sm: [bs, 100]
            # target: [bs, 101]
            predict_list.append(output.detach().cpu().numpy())
            label_list.append(target.detach().cpu().numpy())

            # _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            # 看下_item 的大小

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0 and dist.get_rank() == 0:
            #     progress.display(i, logger)
        # pdb.set_trace()
        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )

        # import ipdb; ipdb.set_trace()
        # calculate mAP
        # saved_data = torch.cat(saved_data, 0).numpy()
        # saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        # np.savetxt(os.path.join(args.output, saved_name), saved_data)
        predict_list = np.concatenate(predict_list)
        label_list = np.concatenate(label_list)

        saved_name_pred = 'saved_data_pred_{}.npy'.format(dist.get_rank())
        saved_name_label = 'saved_data_label_{}.npy'.format(dist.get_rank())

        np.save(os.path.join(args.output, saved_name_pred), predict_list)
        np.save(os.path.join(args.output, saved_name_label), label_list)

        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            print("Calculating mAP:")

            filename_list_pred = ['saved_data_pred_{}.npy'.format(ii) for ii in range(dist.get_world_size())]
            # filename_list_pred_att = ['saved_data_pred_att_{}.npy'.format(ii) for ii in range(dist.get_world_size())]
            filename_list_label = ['saved_data_label_{}.npy'.format(ii) for ii in range(dist.get_world_size())]
            predict_list = np.concatenate(
                [np.load(os.path.join(args.output, file_name)) for file_name in filename_list_pred])
            label_list = np.concatenate(
                [np.load(os.path.join(args.output, file_name)) for file_name in filename_list_label])

            test_num_each = [i-9 for i in val_num_each]

            # for task_model
            mega_pred = np.load('./new_output/test/general/spatial_only_large/val_saved_data_pred_0.npy')  # [10320, 100], w/o sigmoid
            predict_list = (predict_list + mega_pred)/2.0

            for i in range(len(test_num_each)):
                video_label = label_list[sum(test_num_each[:i]): sum(test_num_each[:i + 1])]
                video_predt = predict_list[sum(test_num_each[:i]): sum(test_num_each[:i + 1])]
                recognize.update(video_label, video_predt)
                recognize.video_end()

            # recognize.update(label_list, predict_list)
            val_ap_i = recognize.compute_video_AP('i')["mAP"]
            val_ap_v = recognize.compute_video_AP('v')["mAP"]
            val_ap_t = recognize.compute_video_AP('t')["mAP"]
            val_ap_iv = recognize.compute_video_AP('iv')["mAP"]
            val_ap_it = recognize.compute_video_AP('it')["mAP"]

            val_ap_ivt = recognize.compute_video_AP('ivt', ignore_null=True)
            logger.info('valid ap (I/V/T/IV/IT/IVT): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                val_ap_i, val_ap_v, val_ap_t, val_ap_iv, val_ap_it, val_ap_ivt["mAP"]))
            # longtail_ivt = val_ap_ivt['AP'][classes_longtail_idx]  # 包含nan
            # longtail_ivt_nan = []
            # for i in range(len(longtail_ivt)):
            #     if not np.isnan(longtail_ivt[i]):
            #         longtail_ivt_nan.append(longtail_ivt[i])
            # tmp = len(longtail_ivt_nan)
            # many_ivt = np.mean(longtail_ivt_nan[:int(0.4 * tmp)])
            # mid_ivt = np.mean(longtail_ivt_nan[int(0.4 * tmp): int(0.8 * tmp)])
            # few_ivt = np.mean(longtail_ivt_nan[int(0.8 * tmp):])
            # logger.info('ivt ap (many/mid/few): {:.4f}/{:.4f}/{:.4f}'.format(many_ivt, mid_ivt, few_ivt))
            mAP = val_ap_ivt["mAP"]

        else:
            mAP = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, mAP


##################################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/epoch_'+str(epoch).zfill(2)+'_model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
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

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist


class WeightedRandomSamplerDDP(torch.utils.data.distributed.DistributedSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        data_set: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """

    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, data_set, weights: Sequence[float], num_replicas: int, rank: int, num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        super(WeightedRandomSamplerDDP, self).__init__(data_set, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor = self.rank + rand_tensor * self.num_replicas
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


class Idx_DistributedSampler(DistributedSampler):
    def __init__(self, dataset, sequence_length, list_each_length, ds, ol, **kwargs):
        super(Idx_DistributedSampler, self).__init__(dataset, **kwargs)
        self.seq_len = sequence_length
        self.num_each = list_each_length
        self.ds = ds
        self.ol = ol
        self.start_idx = self.get_start_idx()
        # print("start_idx_len: ", len(self.start_idx))
        self.num_samples = len(self.get_idx(self.start_idx))

    def __iter__(self):
        rank = self.rank
        world_size = self.num_replicas
        start_idx = self.start_idx

        if self.shuffle:
            self.random_sample(start_idx)

        data_indices = self.get_idx(start_idx)

        # print('indices: ', data_indices[:50])
        # pdb.set_trace()

        data_size = len(data_indices)

        # Divide the data into num_replicas blocks

        block_size = data_size // world_size
        block_begin = block_size * rank
        block_end = block_begin + block_size

        # For the last rank, include the remainder of the data
        if rank == world_size - 1:
            block_end = data_size

        # Get the indices for the current block
        indices = data_indices[block_begin:block_end]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def random_sample(self, indices):
        # Implement your own random sampling algorithm here
        # This is just an example implementation that shuffles the indices randomly
        # return torch.randperm(len(indices)).tolist()
        return random.shuffle(indices)

    def get_start_idx(self):
        # 统计每个动作的持续帧数，确定ds和ol
        count = 0
        idx = []
        for i in range(len(self.num_each)):
            end = self.num_each[i] - (self.seq_len - 1) * self.ds
            for j in range(count, count + end, self.ol):
                idx.append(j)
            count += self.num_each[i]
        # print('idx', idx)
        return idx

    def get_idx(self, start_idx):
        train_idx = []
        for i in range(len(start_idx)):
            for j in range(self.seq_len):
                train_idx.append(start_idx[i] + j * self.ds)
        # print('train idx ', train_idx)
        return train_idx


if __name__ == '__main__':
    main()
