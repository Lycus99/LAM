import pdb

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
import torch.nn.functional as F

from .backbone import build_backbone
from .transformer import build_transformer
from utils.misc import clean_state_dict
from collections import OrderedDict
from torchvision import models
from einops import rearrange

import torch.nn.init as init
from .model_aim import *
# from model_aim import CLIP_Layer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6, pos_embed=None):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        self.embed_dim = embed_dim
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))
        print('transformer layers: ', n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if pos_embed == 'Sine':
            self.pos_embed = None
            with torch.no_grad():
                trunc_normal_(self.cls_token, std=.02)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
            with torch.no_grad():
                trunc_normal_(self.pos_embed, std=.02)
                trunc_normal_(self.cls_token, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]  # x: [bs, t, dim]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # x: [bs, t+1, dim]
        if not self.pos_embed is None:
            x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]


class Ablation_Attention(nn.Module):
    def __init__(self, backbone, hidden_dim=512, attention='time', num_frames=10):
        super(Ablation_Attention, self).__init__()
        self.backbone = backbone
        self.num_class = 100
        self.attention_type = attention
        self.dim_t = hidden_dim
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, self.dim_t, kernel_size=1)

        if self.attention_type == 'time':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.transformer_time = TAggregate(clip_length=self.num_frames, embed_dim=self.dim_t, n_layers=4)
        elif self.attention_type == 'space':
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(self.dim_t, 1))
            self.transformer_space = TAggregate(clip_length=7 * 7, embed_dim=self.dim_t, n_layers=4)
        elif self.attention_type == 'joint_space_time':
            self.transformer_space_time = TAggregate(clip_length=7 * 7 * self.num_frames, embed_dim=self.dim_t, n_layers=4)
        elif self.attention_type == 'divided_space_time':
            self.transformer_space = TAggregate(clip_length=7 * 7, embed_dim=self.dim_t, n_layers=2)
            self.transformer_time = TAggregate(clip_length=self.num_frames, embed_dim=self.dim_t, n_layers=2)
        self.fc_a = nn.Linear(self.dim_t, self.num_class, bias=False)

    def forward(self, input):
        b, t, _, h, w = input.shape
        input = input.view(-1, 3, h, w)
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]  # src: [b*t, dim_backbone, h//32, w//32]
        src = self.input_proj(src)  # src: [b*t, self.dim_t, h//32, w//32]

        def attention(x, transf, attn='space'):
            x_ori = x
            if attn == 'space':
                x = x.contiguous().view(b * t, self.dim_t, -1).transpose(1, 2)  # [b*t, hw, self.dim_t]
            elif attn == 'time':
                x = x.contiguous().view(b, t, self.dim_t)  # [b, t, self.dim_t]
            elif attn == 'space_time':
                x = x.contiguous().view(b, -1, self.dim_t)  # [b, hwt, self.dim_t]
            return transf(x).type(x_ori.dtype)  # [b*t, self.dim_t]/[b, self.dim_t]

        if self.attention_type == 'divided_space_time':
            # src += pos
            src = attention(src, self.transformer_space)  # [b*t, dim]
            src = attention(src, self.transformer_time, 'time')
            return self.fc_a(src)

        elif self.attention_type == 'joint_space_time':
            # xi, xv, xt, xa = self.conv_i(src), self.conv_v(src), self.conv_t(src), self.conv_a(src)
            # [b*t, d, h/32, w/32]
            xa = attention(src, self.transformer_space_time, 'space_time')
            return self.fc_a(xa)

        elif self.attention_type == 'time':
            src = self.avg_pool(src).view(-1, self.dim_t)
            # yi, yt = self.fc_i(src[t - 1::t]), self.fc_t(src[t - 1::t])
            src = attention(src, self.transformer_time, 'time')
            # yi, yt = self.fc_i(src), self.fc_t(src)
            # yv, ya = self.fc_v(src), self.fc_a(src)
            return self.fc_a(src)
            # return [src, ya]

        elif self.attention_type == 'space':
            xa = attention(src, self.transformer_space)  # [bs*t, dim]
            xa = xa.contiguous().view(b, t, self.dim_t).transpose(1, 2)
            ya = self.fc_a(self.avg_pool(xa).squeeze(-1))
            return ya


class Spatiotemporal_Transformer(nn.Module):
    def __init__(self, backbone, hidden_dim=512, num_frames=10, pretrained=None, output=None):
        super(Spatiotemporal_Transformer, self).__init__()
        self.backbone = backbone
        self.num_class = 100
        self.dim_t = hidden_dim
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, self.dim_t, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.transformer_space = TAggregate(clip_length=7 * 7, embed_dim=self.dim_t, n_layers=2)
        self.transformer_time = TAggregate(clip_length=self.num_frames, embed_dim=self.dim_t, n_layers=2)
        # self.transformer_time = TAggregate(clip_length=self.num_frames, embed_dim=self.dim_t, n_layers=4)
        if not pretrained is None:
            self.init_weights(pretrained=pretrained)
        self.output = output
        if self.output == 'task':
            self.fc_i = nn.Linear(self.dim_t, 6, bias=False)
            self.fc_v = nn.Linear(self.dim_t, 10, bias=False)
            self.fc_t = nn.Linear(self.dim_t, 15, bias=False)
            self.fc_a = nn.Linear(self.dim_t, 100, bias=False)

    def init_weights(self, pretrained=None):
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            print(f'=> load task expert model from: {self.pretrained}')
            task_model = torch.load(self.pretrained, map_location='cpu')['state_dict']
            msg = self.load_state_dict(task_model, strict=False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()

    def forward(self, input):
        b, t, _, h, w = input.shape
        input = input.view(-1, 3, h, w)
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]  # src: [b*t, dim_backbone, h//32, w//32]
        del pos
        src = self.input_proj(src)  # src: [b*t, self.dim_t, h//32, w//32]

        def attention(x, transf, attn='space'):
            x_ori = x
            if attn == 'space':
                x = x.contiguous().view(b * t, transf.embed_dim, -1).transpose(1, 2)  # [b*t, hw, self.dim_t]
            elif attn == 'time':
                x = x.contiguous().view(b, t, transf.embed_dim)  # [b, t, self.dim_t]
            elif attn == 'space_time':
                x = x.contiguous().view(b, -1, transf.embed_dim)  # [b, hwt, self.dim_t]
            return transf(x).type(x_ori.dtype)  # [b*t, self.dim_t]/[b, self.dim_t]

        # src_s = self.avg_pool(src).view(-1, self.dim_t)
        # src_st = attention(src_s, self.transformer_time, 'time')  # [bs, self.dim_t]

        # if self.output == 'feature':
        #     return src_st
        if self.output == 'task':
            # src += pos
            src_s = attention(src, self.transformer_space)  # [b*t, dim]
            src_st = attention(src_s, self.transformer_time, 'time')
            return [self.fc_i(src_s[t-1::t]), self.fc_v(src_st), self.fc_t(src_s[t-1::t]), self.fc_a(src_st)]
            # return [src_st, self.fc_a(src_st)]  # get_model_features
        elif self.output == 'feature':
            src_s = attention(src, self.transformer_space)  # [b*t, dim]
            src_st = attention(src_s, self.transformer_time, 'time')
            return [src_s, src_st]
            # yi, yt = self.fc_i(src_s[t - 1::t]), self.fc_t(src_s[t - 1::t])
            # yi, yt = self.fc_i(src_st), self.fc_t(src_st)
            # yv, ya = self.fc_v(src_st), self.fc_a(src_st)
            # return [yi, yv, yt, ya]


class Task_General(nn.Module):
    def __init__(self, model_task, model_general, general_type=None, integrate=None):
        super(Task_General, self).__init__()
        self.model_task = model_task
        self.model_general = model_general
        self.num_frames = 10
        self.num_class = 100
        self.dim_t = 512
        self.dim_general = 768
        self.integrate = integrate
        self.dim_joint = 512

        # self.fc_general2latent = nn.Linear(self.dim_general, self.dim_joint)
        # self.fc_task2latent = nn.Linear(self.dim_t, self.dim_joint)
        self.t_general = TAggregate(clip_length=self.num_frames, embed_dim=self.dim_general, n_layers=4)

        s = 1
        if self.integrate == 'concat':
            s = 2

        # self.fc_joint_i = nn.Linear(self.dim_joint * s, 6)
        # self.fc_joint_v = nn.Linear(self.dim_joint * s, 10)
        # self.fc_joint_t = nn.Linear(self.dim_joint * s, 15)
        self.fc_joint_a = nn.Linear(self.dim_joint + self.dim_general, 100)

        # self.fc_general = nn.Linear(self.dim_general, self.num_class, bias=False)
        # self.fc_joint_task_general = nn.Linear(self.dim_joint, self.num_class, bias=False)
        # self.fc_joint_task_general = nn.Linear(self.dim_joint*2, self.num_class, bias=False)


    def forward(self, input):
        b, t, _, h, w = input.shape
        [task_s, task_st] = self.model_task(input)  # [bs, self.dim_t]
        # task_s = self.fc_task2latent(task_s)[t-1::t]
        # task_st = self.fc_task2latent(task_st)

        def attention(x, transf):
            x_ori = x
            x = x.contiguous().view(b, t, transf.embed_dim)  # [b, t, self.dim_t]
            return transf(x).type(x_ori.dtype)  # [b*t, self.dim_t]/[b, self.dim_t]

        input = input.view(-1, 3, h, w)
        general_s = self.model_general(input)  # [bs*t, self.dim_general]
        general_st = attention(general_s, self.t_general)  # [bs, self.dim_gen]
        # general_s = self.fc_general2latent(general_s)[t-1::t]
        # general_st = self.fc_general2latent(general_st)

        ya = self.fc_joint_a(torch.cat((task_st, general_st), dim=1))
        return ya


class General_Only(nn.Module):
    def __init__(self, model_general, general_type):
        super(General_Only, self).__init__()
        self.model_general = model_general
        self.num_class = 100
        self.dim_general = 1024
        self.fc_general = nn.Linear(self.dim_general, self.num_class, bias=False)
        self.general_type = general_type
        if general_type == 'ft' or general_type == 'spatial':
            self.num_frames = 10
            self.fc_i = nn.Linear(self.dim_general, 6, bias=False)
            self.fc_v = nn.Linear(self.dim_general, 10, bias=False)
            self.fc_t = nn.Linear(self.dim_general, 15, bias=False)
            self.t_general = TAggregate(clip_length=self.num_frames, embed_dim=self.dim_general, n_layers=4)

    def forward(self, input):
        b, t, _, h, w = input.shape
        input = input.view(-1, 3, h, w)  # [bs*t, 768]

        def attention(x, transf):
            x_ori = x
            x = x.contiguous().view(b, t, transf.embed_dim)  # [b, t, self.dim_t]
            return transf(x).type(x_ori.dtype)  # [b*t, self.dim_t]/[b, self.dim_t]

        output = self.model_general(input)
        if self.general_type == 'ft' or self.general_type == 'spatial':
            yi, yt = self.fc_i(output[t - 1::t]), self.fc_t(output[t - 1::t])
            output = attention(output, self.t_general)
            yv, ya = self.fc_v(output), self.fc_general(output)
            return [yi, yv, yt, ya]
            # return [output, ya]  # get_model_features
        else:
            return self.fc_general(output)  # [bs, 100]
            # return [output, self.fc_general(output)]  # get_model_features


def build_attention_ablation(args):
    backbone = build_backbone(args)
    model = Ablation_Attention(
        backbone=backbone,
        attention=args.attention_type,
        num_frames=args.num_frames
    )
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model


def build_lite_only(args, hidden_dim):
    task_backbone = build_backbone(args)
    task_model = Spatiotemporal_Transformer(task_backbone, hidden_dim=hidden_dim, output='task')
    for name, param in task_model.named_parameters():
        param.requires_grad = True
    return task_model


def build_mega_only(general='None'):
    # pretrained_pth = '/home/liyc/Code/query2labels-main/pretrained_weights/ViT-B-16.pt'
    pretrained_pth = '/home/liyc/Code/query2labels-main/pretrained_weights/ViT-L-14.pt'

    if general == 'aim':
        num_tadapter = 1
        adapter_scale = 2.0
        mlp_adapter_ratio = 0.25
        general_model = CLIP_Aim(pretrained=pretrained_pth, num_tadapter=num_tadapter, adapter_scale=adapter_scale,
                                 mlp_adapter_ratio=mlp_adapter_ratio)
        print('num_tadapter: ', num_tadapter, 'adapter_scale: ', adapter_scale, 'mlp_adapter_ratio: ', mlp_adapter_ratio)
        pdb.set_trace()
    elif general == 'ft':
        general_model = CLIP_Only(pretrained=pretrained_pth)
    elif general == 'spatial':
        adapter_scale = 2.0
        mlp_adapter_ratio = 0.25
        # general_model = CLIP_SpatialAdapter(pretrained=pretrained_pth, adapter_scale=adapter_scale,
        #                                     mlp_adapter_ratio=mlp_adapter_ratio)
        # input_resolution=224, patch_size=16, width=768, layers=12, heads=8, pretrained=None,
        general_model = CLIP_SpatialAdapter(patch_size=14, width=1024, layers=24, heads=16,
                                            pretrained=pretrained_pth, adapter_scale=adapter_scale,
                                            mlp_adapter_ratio=mlp_adapter_ratio)
    else:
        raise ValueError('no general model!')

    model = General_Only(general_model, general)

    if general == 'aim':
        for name, param in model.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name \
                and 'Adapter' not in name and 't_general' not in name and 'fc_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif general == 'ft':
        # freeze_n = 12
        # freeze_name = ['resblocks.{}.'.format(i) for i in range(12)]
        # freeze_list = freeze_name[:freeze_n]
        # freeze_list += ['class_embedding', 'positional_embedding', 'conv1.weight', 'ln_pre']

        freeze_list = ['class_embedding', 'positional_embedding', 'ln', 'bias']

        for name, param in model.named_parameters():
            if any(keyword in name for keyword in freeze_list):
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif general == 'spatial':
        for name, param in model.named_parameters():
            # if 'Adapter' not in name and 't_general' not in name and 'fc_' not in name and 'ln_post' not in name \
            #         and 'resblocks.8.' not in name and 'resblocks.9.' not in name \
            #         and 'resblocks.10.' not in name and 'resblocks.11.' not in name:
            #     param.requires_grad = False
            # else:
            #     param.requires_grad = True

            if 'Adapter' not in name and 't_general' not in name and 'fc_' not in name and 'ln_post' not in name \
                    and 'resblocks.22.' not in name and 'resblocks.23.' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return model


def build_lite_mega(args, general=None):
    task_backbone = build_backbone(args)
    task_model = Spatiotemporal_Transformer(task_backbone, output='feature',
                                            pretrained='/home/liyc/Code/query2labels-main/new_output/train_val'
                                                 '/task/task_hier_loss/epoch_12_model_best.pth.tar')
    pretrained_pth = '/home/liyc/Code/query2labels-main/pretrained_weights/ViT-B-16.pt'

    if general == 'spatial':
        adapter_scale = 2.0
        mlp_adapter_ratio = 0.25
        general_model = CLIP_SpatialAdapter(pretrained=pretrained_pth, adapter_scale=adapter_scale,
                                            mlp_adapter_ratio=mlp_adapter_ratio)
    elif general == 'ft':
        general_model = CLIP_Only(pretrained=pretrained_pth)
    else:
        raise ValueError('no general model')
    model = Task_General(task_model, general_model, general, integrate='concat')
    msg=model.load_state_dict(torch.load('/home/liyc/Code/query2labels-main/new_output/train_val/general/'
                                     'freeze2_adapter8_ft2/epoch_06_model_best.pth.tar', map_location='cpu')['state_dict'], strict=False)
    print(f"Loaded model with msg: {msg}")
    pdb.set_trace()
    if general == 'aim':
        for name, param in model.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name \
                and 'Adapter' not in name and 't_general' not in name \
                    and 'fc_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif general == 'ft':
        for name, param in model.named_parameters():
            if 'model_task' not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif general == 'spatial':
        for name, param in model.named_parameters():
            # if 'model_task' and 'Adapter' not in name and 't_general' not in name and 'fc_' not in name and 'ln_post' not in name \
            #         and 'resblocks.10.' not in name and 'resblocks.11.' not in name:
            #     param.requires_grad = False
            # else:
            #     param.requires_grad = True

            if 'fc_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    #
    # pdb.set_trace()

    return model
