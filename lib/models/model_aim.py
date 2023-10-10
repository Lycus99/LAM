import pdb
from collections import OrderedDict
from typing import Tuple, Union
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#
#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask
#
#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
#
#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ResidualAttentionBlockAim(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 scale=1., num_tdapter=1, num_frames=10, mlp_adapter_ratio=0.25, drop_path=0.0):
        super().__init__()
        # clip
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # adapter
        self.num_frames = num_frames
        self.num_tadapter = num_tdapter  # tadapterd的数量，sth-sth v2中为2
        self.MLP_Adapter = Adapter(d_model, skip_connect=False, mlp_ratio=mlp_adapter_ratio)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        if num_tdapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape

        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)

        ## spatial adaptation
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))

        ## joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))

        # # clip
        # x = x + self.attention(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        # clip
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # clip
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlockSpatial(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, mlp_adapter_ratio=0.25,
                 drop_path=0.2, scale=1.):
        super().__init__()
        # clip
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.S_Adapter = Adapter(d_model)
        self.MLP_Adapter = Adapter(d_model, skip_connect=False, mlp_ratio=mlp_adapter_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # clip
        xs = self.S_Adapter(self.attention(self.ln_1(x)))
        x = x + self.drop_path(xs)

        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 num_frames=10, num_tadapter=1, scale=1., adapter=None, mlp_adapter_ratio=0.25):
        super().__init__()
        self.width = width
        self.layers = layers

        if adapter == 'Aim':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlockAim(width, heads, attn_mask,
                                scale=scale, num_tdapter=num_tadapter, num_frames=num_frames,
                                mlp_adapter_ratio=mlp_adapter_ratio, drop_path=0.2) for _ in range(layers)])
        elif adapter == 'Spatial':
            resblocks = []
            for i in range(layers):
                if i < 2 or i >= 22:
                    resblocks.append(ResidualAttentionBlock(width, heads, attn_mask))
                else:
                    resblocks.append(ResidualAttentionBlockSpatial(width, heads, attn_mask, scale=scale,
                                                                    mlp_adapter_ratio=mlp_adapter_ratio, drop_path=0.2))
            self.resblocks = nn.Sequential(*resblocks)
            # pdb.set_trace()
            # self.resblocks = nn.Sequential(*[ResidualAttentionBlockSpatial(width, heads, attn_mask, scale=scale,
            #                                                     mlp_adapter_ratio=mlp_adapter_ratio, drop_path=0.2)
            #                                  for _ in range(layers)])
        else:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask)
                                 for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class CLIP_Aim(nn.Module):
    # vision_width = 768
    # vision_layers = 12
    # vision_patch_size = 16
    # # grid_size = 14
    # image_resolution = 224
    # vision_heads = vision_width // 64

    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=8,
                 num_frames=10, num_tadapter=1, adapter_scale=0.5, pretrained=None, mlp_adapter_ratio=0.25):
        super().__init__()
        self.input_resolution = input_resolution
        # self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.transformer = Transformer(width, layers, heads, num_frames=num_frames, num_tadapter=num_tadapter,
                                       scale=adapter_scale, adapter='Aim', mlp_adapter_ratio=mlp_adapter_ratio)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            # self.apply(_init_weights)
            print(f'load model from: {self.pretrained}')
            clip_model = torch.load(self.pretrained, map_location='cpu')
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    def forward(self, x: torch.Tensor):
        # x: [bt, 3, 224, 224]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # pdb.set_trace()
        x = self.ln_post(x)
        x = x[self.num_frames-1::self.num_frames, 0]  # [bs, 768]

        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj
        return x


class CLIP_Only(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=8, pretrained=None):
        super().__init__()
        self.input_resolution = input_resolution
        # self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # self.num_frames = num_frames
        # self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            # self.apply(_init_weights)
            print(f'=> load model from: {self.pretrained}')
            clip_model = torch.load(self.pretrained, map_location='cpu')
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()

            for n, m in self.transformer.named_modules():
                if 'Adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

    def forward(self, x: torch.Tensor):
        # x: [bt, 3, 224, 224]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x[:, 0, :]  # [bs*t, 768]

        return x


class CLIP_SpatialAdapter(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=8, pretrained=None,
                 adapter_scale=0.5, mlp_adapter_ratio=0.25):
        super().__init__()
        self.input_resolution = input_resolution
        # self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # self.num_frames = num_frames
        # self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.transformer = Transformer(width, layers, heads, adapter='Spatial', scale=adapter_scale,
                                       mlp_adapter_ratio=mlp_adapter_ratio)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            # self.apply(_init_weights)
            print(f'=> load model from: {self.pretrained}')
            clip_model = torch.load(self.pretrained, map_location='cpu')
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


    def forward(self, x: torch.Tensor):
        # x: [bt, 3, 224, 224]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x[:, 0, :]  # [bs*t, 768]

        return x



if __name__ == '__main__':
    vision_width = 768
    vision_layers = 12
    vision_patch_size = 16
    grid_size = 14
    image_resolution = 224
    vision_heads = vision_width // 64
    vit_clip = CLIP_Aim(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        num_frames=10,
        num_tadapter=1,
        adapter_scale=0.5,
        pretrained='/home/liyc/Code/query2labels-main/pretrained_weights/ViT-B-16.pt'
    )
    x = torch.randn(200, 3, 224, 224)
    y = vit_clip(x)
    print(y.shape)