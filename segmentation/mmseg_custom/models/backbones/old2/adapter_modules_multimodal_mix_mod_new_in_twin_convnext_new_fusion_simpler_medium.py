import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath
# from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

_logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from mmcv.cnn import ConvModule
from mmcv_custom.cnn import ConvModule
# from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init)
from mmengine_custom.model import (BaseModule, ModuleList, caffe2_xavier_init)
# from mmseg_custom.registry import MODELS
# from mmseg_custom.utils import add_prefix
from einops import rearrange
from torch.nn import init, Sequential
import numbers
import numpy as np
import math
from .base.twin_convnext import TwinConvNeXt
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,
                 groups=1,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        scale = 1
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.scale2 = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1,groups=groups, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1,groups=groups, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        out = x + out * self.scale2
        return out
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class GFE(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,
                 groups = 1,):
        super(GFE, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,groups = groups)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x
    

class FFRM(BaseModule):
    """Fused Feature Recalibration Module in RoadFormer"""
    def __init__(self, in_chan, out_chan, norm=None):
        super(FFRM, self).__init__()
        self.conv_atten = ConvModule(in_chan, in_chan, kernel_size=1, bias=False, norm_cfg=norm)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
    def init_weights(self) -> None:
        """Initialize weights."""
        caffe2_xavier_init(self.conv_atten, bias=0)
    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))) 
        enhancefeat = torch.mul(x, atten)
        x = x + enhancefeat
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.SyncBatchNorm(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out
class CA(BaseModule):
    """Fused Feature Recalibration Module in RoadFormer with Coordinate Attention"""
    def __init__(self, in_chan, out_chan, norm=None):
        super(CA, self).__init__()
        self.coord_atten = CoordinateAttention(in_chan, in_chan)
        self.init_weights()
    def init_weights(self) -> None:
        """Initialize weights."""
        for m in self.coord_atten.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        atten = self.coord_atten(x)
        x = x + atten
        return x
class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
class GFFM(nn.Module):
    """Heterogeneous Feature Fusion Module in RoadFormer"""
    def __init__(self, feat_scale,dim):
        super().__init__()
        self.gammax = Scale(0)
        self.gammay = Scale(0)
        num_feats = feat_scale[0]*feat_scale[1]
        self.norm = nn.LayerNorm(num_feats)
    def forward(self, x):
        split_dim = x.size(1) // 2
        x, y = torch.split(x, (split_dim, split_dim), dim=1)
        batch_size, channels, height, width = x.size()
        qx = x.view(batch_size, channels, -1)
        kx = x.view(batch_size, channels, -1).permute(0, 2, 1)
        vx = x.view(batch_size, channels, -1)
        qy = y.view(batch_size, channels, -1)
        ky = y.view(batch_size, channels, -1).permute(0, 2, 1)
        vy = y.view(batch_size, channels, -1)
        energy_x = torch.bmm(qx, ky)
        energy_y = torch.bmm(qy, kx)
        attention_x = F.softmax(energy_x, dim=-1)
        attention_y = F.softmax(energy_y, dim=-1)
        outx = torch.bmm(attention_x, vy)
        outy = torch.bmm(attention_y, vx)
        outx = outx.view(batch_size, channels, height, width)
        outy = outy.view(batch_size, channels, height, width)
        outx = self.gammax(outx) + x
        outy = self.gammay(outy) + y
        outx = outx.view(batch_size, channels, -1)
        outy = outy.view(batch_size, channels, -1)
        out = torch.cat((outx, outy), dim=1)
        out = self.norm(out)
        out = out.view(batch_size, channels * 2, height, width)
        return out
class Scale2(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
    def forward(self, x, y):
        return x * self.scale1 + y * self.scale2
class MobileNetV2(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(MobileNetV2, self).__init__()   
        hidden_dim = in_channel * 2
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channel, 1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
    def forward(self, x):       
        x = self.bottleneckBlock(x) * self.scale + x
        return x
# @MODELS.register_module()
class RoadFormer2Neck(BaseModule):   
    def __init__(self,
                 in_channels=[128,256,512,512],
                 out_channels=[128,256,512,512],
                 layer = 4,
                 img_scale=(1024, 1024),
                 norm_cfg=dict(type='GN', num_groups=32)):
        super().__init__()
        assert isinstance(in_channels, list)
        assert len(in_channels) == len(out_channels)
        self.in_channels = in_channels
        self.in_channels1 = in_channels
        self.in_channels2 = [i//2 for i in in_channels]
        self.out_channels = out_channels
        self.num_input_levels = len(out_channels)     
        if layer is not None:
            self.layer = layer
        else:
            self.layer = self.num_input_levels
        enhance_blocks = ModuleList()
        for i in range(self.layer):            
            old_channel = in_channels[i]
            new_channel = out_channels[i]            
            enhance_block = FFRM(old_channel, new_channel, norm=norm_cfg)
            enhance_blocks.append(enhance_block)
        self.enhance_blocks = enhance_blocks
        self.img_scale = list(img_scale)
        self.global_feature_encoder_rgb = ModuleList([
            GFE(dim=ch // 2, num_heads=8, ffn_expansion_factor=2, qkv_bias=False,groups= 32)
            for ch in self.in_channels1
        ])        
        self.global_feature_encoder_sne = ModuleList([
            GFE(dim=ch // 2, num_heads=8, ffn_expansion_factor=2, qkv_bias=False,groups= 32)
            for ch in self.in_channels1
        ])       
        self.local_feature_encoder_rgb = ModuleList([
            MobileNetV2(in_channel = ch, out_channel = ch)
            for ch in self.in_channels2
        ])        
        self.local_feature_encoder_sne = ModuleList([
            MobileNetV2(in_channel = ch, out_channel = ch)
            for ch in self.in_channels2
        ])      
        ca_blocks = ModuleList()
        for i in range(self.num_input_levels):            
            old_channel = in_channels[i]
            new_channel = out_channels[i]            
            ca_block = CA(old_channel, new_channel, norm=norm_cfg)
            ca_blocks.append(ca_block)
        self.ca_blocks = ca_blocks
        feat_scales = []
        for i in range(self.num_input_levels):
            feat_scale = (self.img_scale[0]//2**(i+2), self.img_scale[1]//2**(i+2))
            feat_scales.append(feat_scale)
        fuse_blocks = ModuleList()
        scale_layers = ModuleList()
        for i in range(self.layer):
            fuse_block = GFFM(feat_scales[i],self.in_channels1[i])
            fuse_blocks.append(fuse_block)
            scale = Scale2()
            scale_layers.append(scale)
        self.fuse_blocks = fuse_blocks 
        self.scale_layers = scale_layers 
        self.detail_feature_extractions = ModuleList([
            Mlp(in_features=ch, ffn_expansion_factor=1,)  
            for ch in self.in_channels1[:self.layer]
        ])
    def forward(self, feats):
        assert len(feats) == len(self.in_channels1)
        assert len(feats) == len(self.in_channels2)
        feats_g = []
        feats_l = []        
        # feats_rgb_g = []
        # feats_sne_g = []
        # feats_rgb_l = []
        # feats_sne_l = []
        # newfeats = []
        # loss_decomp = []
        for i, feat in enumerate(feats):
            split_dim = self.in_channels[i] // 2
            feat_rgb, feat_sne = torch.split(feat, (split_dim, split_dim), dim=1)
            feat_rgb_g = self.global_feature_encoder_rgb[i](feat_rgb)
            feat_sne_g = self.global_feature_encoder_sne[i](feat_sne)
            feat_rgb_l = self.local_feature_encoder_rgb[i](feat_rgb)
            feat_sne_l = self.local_feature_encoder_sne[i](feat_sne)
            feats_g.append(torch.cat((feat_rgb_g, feat_sne_g), dim=1))
            feats_l.append(torch.cat((feat_rgb_l, feat_sne_l), dim=1))
        for i in range(self.layer):
            feats_g[i] = self.fuse_blocks[i](feats_g[i])
            feats_l[i] = self.detail_feature_extractions[i](feats_l[i])
            feats_g[i] = self.enhance_blocks[i](feats_g[i])
            feats[i] = self.scale_layers[i](feats_g[i],feats_l[i])
        for i in range(self.num_input_levels):
            feat = feats[i]
            feat_enhanced = self.ca_blocks[i](feat)
            feats[i] = feat_enhanced
        # losses = None
        return feats
    # , losses

def get_reference_points(spatial_shapes, device,type):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=type, device=device),#######mod torch.float32
            torch.linspace(0.5, W_ - 0.5, W_, dtype=type, device=device))#######mod torch.float32
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device,x.dtype)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device) 
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device,x.dtype)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
    
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query
        
        if self.with_cp and query.requires_grad:
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
            #     with record_function("model_inference"):
            query = cp.checkpoint(_inner_forward, query, feat)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            query = _inner_forward(query, feat)
            
        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
                # with record_function("model_inference"):
            query = cp.checkpoint(_inner_forward, query, feat)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            query = _inner_forward(query, feat)
            
        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, cls
    

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, in_channels=3):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
    
            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
# class SpatialPriorModulewithGammaSpatial(nn.Module):
#     def __init__(self, inplanes=64, embed_dim=384, with_cp=False, in_channels=3, num_mod=4):
#         super().__init__()
#         self.with_cp = with_cp
#         self.num_mod = num_mod

#         self.stem = nn.Sequential(*[
#             nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         ])
#         self.conv2 = nn.Sequential(*[
#             nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(2 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv3 = nn.Sequential(*[
#             nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False), 
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv4 = nn.Sequential(*[
#             nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.fc1 = nn.Conv2d(inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(2 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc3 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
#         # self.fc3 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
#         self.fc4 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
        
#         def _inner_forward(x):
#             c1 = self.stem(x)
#             c2 = self.conv2(c1)
#             c3 = self.conv3(c2)
#             c4 = self.conv4(c3)
#             c1 = self.fc1(c1)
#             c2 = self.fc2(c2)
#             c3 = self.fc3(c3)
#             c4 = self.fc4(c4)
#             mod_selector_gamma1=c1[:,-self.num_mod:,:,:] # bs, num_mod, H/4, W/4
#             mod_selector_gamma2=c2[:,-self.num_mod:,:,:] # bs, num_mod, H/8, W/8
#             mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
#             # mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
#             mod_selector_gamma4=c4[:,-self.num_mod:,:,:] # bs, num_mod, H/32, W/32
#             # mod_selector_gamma_inj=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
#             # mod_selector_gamma1=mod_selector_gamma1+F.interpolate(mod_selector_gamma4, size=c1.shape[-2:], mode='bilinear', align_corners=False)
#             # mod_selector_gamma2=mod_selector_gamma2+F.interpolate(mod_selector_gamma4, size=c2.shape[-2:], mode='bilinear', align_corners=False)
#             # mod_selector_gamma3=mod_selector_gamma3+F.interpolate(mod_selector_gamma4, size=c3.shape[-2:], mode='bilinear', align_corners=False) 
#             c1=c1[:,:-self.num_mod,:,:]
#             c2=c2[:,:-self.num_mod,:,:]
#             c3=c3[:,:-self.num_mod,:,:]
#             # c3=c3[:,:-self.num_mod,:,:]
#             c4=c4[:,:-self.num_mod,:,:]
#             bs, dim, _, _ = c1.shape
#             c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
#             c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
#             c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
#             c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
            
#             mod_selector_gamma1=mod_selector_gamma1.view(bs, self.num_mod, -1).transpose(1, 2)  # 4s bs,num_mod, HW/4^2 view is flattening the tensor transpose therefore bs, HW/4^2, num_mod
#             mod_selector_gamma2=mod_selector_gamma2.view(bs, self.num_mod, -1).transpose(1, 2) #8s
#             mod_selector_gamma3=mod_selector_gamma3.view(bs, self.num_mod, -1).transpose(1, 2) #16s
#             mod_selector_gamma4=mod_selector_gamma4.view(bs, self.num_mod, -1).transpose(1, 2) #32s
#             # mod_selector_gamma_inj=mod_selector_gamma_inj.view(bs, self.num_mod, -1).transpose(1, 2) #16s
            
#             return c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
#         # , mod_selector_gamma_inj
#             # return c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        
#         if self.with_cp and x.requires_grad:
#             outs = cp.checkpoint(_inner_forward, x)
#         else:
#             outs = _inner_forward(x)
#         return outs
    


# class ModalitySelector(nn.Module):
#     def __init__(self, inplanes=64, with_cp=False, in_channels=3, num_mod=4):
#         super().__init__()
#         self.with_cp = with_cp
#         self.num_mod = num_mod

#         self.stem = nn.Sequential(*[
#             nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.SyncBatchNorm(inplanes),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         ])
#         self.conv2 = nn.Sequential(*[
#             nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(2 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv3 = nn.Sequential(*[
#             nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False), 
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.conv4 = nn.Sequential(*[
#             nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.SyncBatchNorm(4 * inplanes),
#             nn.ReLU(inplace=True)
#         ])
#         self.fc1 = nn.Conv2d(inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(2 * inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc3 = nn.Conv2d(4 * inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
#         # self.fc3 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
#         self.fc4 = nn.Conv2d(4 * inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
        
#         def _inner_forward(x):
#             mod_selector_gamma1 = self.stem(x)
#             mod_selector_gamma2 = self.conv2(mod_selector_gamma1)
#             mod_selector_gamma3 = self.conv3(mod_selector_gamma2)
#             mod_selector_gamma4 = self.conv4(mod_selector_gamma3)
#             mod_selector_gamma1 = self.fc1(mod_selector_gamma1)
#             mod_selector_gamma2 = self.fc2(mod_selector_gamma2)
#             mod_selector_gamma3 = self.fc3(mod_selector_gamma3)
#             mod_selector_gamma4 = self.fc4(mod_selector_gamma4)
#             # mod_selector_gamma1=c1[:,-self.num_mod:,:,:] # bs, num_mod, H/4, W/4
#             # mod_selector_gamma2=c2[:,-self.num_mod:,:,:] # bs, num_mod, H/8, W/8
#             # mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
#             # # mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
#             # mod_selector_gamma4=c4[:,-self.num_mod:,:,:] # bs, num_mod, H/32, W/32
#             # # mod_selector_gamma_inj=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
#             # # mod_selector_gamma1=mod_selector_gamma1+F.interpolate(mod_selector_gamma4, size=c1.shape[-2:], mode='bilinear', align_corners=False)
#             # # mod_selector_gamma2=mod_selector_gamma2+F.interpolate(mod_selector_gamma4, size=c2.shape[-2:], mode='bilinear', align_corners=False)
#             # # mod_selector_gamma3=mod_selector_gamma3+F.interpolate(mod_selector_gamma4, size=c3.shape[-2:], mode='bilinear', align_corners=False) 
#             # c1=c1[:,:-self.num_mod,:,:]
#             # c2=c2[:,:-self.num_mod,:,:]
#             # c3=c3[:,:-self.num_mod,:,:]
#             # # c3=c3[:,:-self.num_mod,:,:]
#             # c4=c4[:,:-self.num_mod,:,:]
#             bs, dim, _, _ = mod_selector_gamma1.shape
#             mod_selector_gamma1 = mod_selector_gamma1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
#             mod_selector_gamma2 = mod_selector_gamma2.view(bs, dim, -1).transpose(1, 2)  # 8s
#             mod_selector_gamma3 = mod_selector_gamma3.view(bs, dim, -1).transpose(1, 2)  # 16s
#             mod_selector_gamma4 = mod_selector_gamma4.view(bs, dim, -1).transpose(1, 2)  # 32s
            
#             # mod_selector_gamma1=mod_selector_gamma1.view(bs, self.num_mod, -1).transpose(1, 2)  # 4s bs,num_mod, HW/4^2 view is flattening the tensor transpose therefore bs, HW/4^2, num_mod
#             # mod_selector_gamma2=mod_selector_gamma2.view(bs, self.num_mod, -1).transpose(1, 2) #8s
#             # mod_selector_gamma3=mod_selector_gamma3.view(bs, self.num_mod, -1).transpose(1, 2) #16s
#             # mod_selector_gamma4=mod_selector_gamma4.view(bs, self.num_mod, -1).transpose(1, 2) #32s
#             # mod_selector_gamma_inj=mod_selector_gamma_inj.view(bs, self.num_mod, -1).transpose(1, 2) #16s
            
#             return mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
#         # , mod_selector_gamma_inj
#             # return c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        
#         if self.with_cp and x.requires_grad:
#             outs = cp.checkpoint(_inner_forward, x)
#         else:
#             outs = _inner_forward(x)
#         return outs
    
# class SimpleFusion(nn.Module):
#     def __init__(self, in_channels):
#         super(SimpleFusion, self).__init__()
#         self.in_channels = in_channels

#     def forward(self, feats):
#         assert len(feats) == len(self.in_channels)
#         fused_feats = []
#         for feat in feats:
#             fused_feats.append(feat)
#         return fused_feats



class SpatialPriorModuleBimodal(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, in_channels=3, img_size=(1024,1024),arch='base',checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'):
        super().__init__()
        self.with_cp = with_cp
        self.inplanes=inplanes
        # inplanes=64
        # inplanes=48
        # arch='base'
        # arch='tiny'
        # checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'
        # checkpoint='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth'
        self.twin_conv=TwinConvNeXt(
                # arch='tiny',
                arch=arch,
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.4,#0
                 layer_scale_init_value=1.0,#1e-6
                 out_indices=[0, 1, 2, 3],
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=True,
                 init_cfg=
                 dict(
            type='Pretrained', 
            # checkpoint='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth',
            checkpoint=checkpoint,
            prefix='backbone.'),
                 )
        # self.stem = nn.Sequential(*[
        #     nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ])
        # self.stem.apply(self._init_weights)
        # self.conv2 = nn.Sequential(*[
        #     nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(2 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv2.apply(self._init_weights)
        # self.conv3 = nn.Sequential(*[
        #     nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv3.apply(self._init_weights)
        # self.conv4 = nn.Sequential(*[
        #     nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv4.apply(self._init_weights)
        self.fc1 = nn.Sequential(*[nn.Conv2d(inplanes*4, embed_dim//2, kernel_size=1, stride=1, padding=0, bias=True),nn.ReLU(inplace=True)])
        self.fc1.apply(self._init_weights)
        self.fc2 = nn.Sequential(*[nn.Conv2d(inplanes*8, embed_dim//2, kernel_size=1, stride=1, padding=0, bias=True),nn.ReLU(inplace=True)])
        self.fc2.apply(self._init_weights)
        self.fc3 = nn.Sequential(*[nn.Conv2d(inplanes*16, embed_dim//2, kernel_size=1, stride=1, padding=0, bias=True),nn.ReLU(inplace=True)])
        self.fc3.apply(self._init_weights)
        self.fc4 = nn.Sequential(*[nn.Conv2d(inplanes*32, embed_dim//2, kernel_size=1, stride=1, padding=0, bias=True),nn.ReLU(inplace=True)])
        self.fc4.apply(self._init_weights)
        self.fc1_ = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1_.apply(self._init_weights)
        self.fc2_ = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2_.apply(self._init_weights)
        self.fc3_ = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3_.apply(self._init_weights)
        self.fc4_ = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4_.apply(self._init_weights)
        # self.fc1_sum = nn.Conv2d(inplanes*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc1_sum.apply(self._init_weights)
        # self.fc2_sum = nn.Conv2d(inplanes*4, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2_sum.apply(self._init_weights)
        # self.fc3_sum = nn.Conv2d(inplanes*8, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc3_sum.apply(self._init_weights)
        # self.fc4_sum = nn.Conv2d(inplanes*16, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4_sum.apply(self._init_weights)
        
        # self.fc1_comb = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc1_comb.apply(self._init_weights)
        # self.fc2_comb = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2_comb.apply(self._init_weights)
        # self.fc3_comb = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc3_comb.apply(self._init_weights)
        # self.fc4_comb = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4_comb.apply(self._init_weights)
        
        # self.weight_sum_1 = nn.Sequential(*[nn.Conv2d(2*embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=True),nn.Softmax2d()])
        # self.weight_sum_2 = nn.Sequential(*[nn.Conv2d(2*embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=True),nn.Softmax2d()])
        # self.weight_sum_3 = nn.Sequential(*[nn.Conv2d(2*embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=True),nn.Softmax2d()])
        # self.weight_sum_4 = nn.Sequential(*[nn.Conv2d(2*embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=True),nn.Softmax2d()])
        # self.weight_sum_1.apply(self._init_weights)
        # self.weight_sum_2.apply(self._init_weights)
        # self.weight_sum_3.apply(self._init_weights)
        # self.weight_sum_4.apply(self._init_weights)
        
        # self.weight_conc_1 = nn.Sequential(*[nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),nn.Sigmoid()])
        # self.weight_conc_2 = nn.Sequential(*[nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),nn.Sigmoid()])
        # self.weight_conc_3 = nn.Sequential(*[nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),nn.Sigmoid()])
        # self.weight_conc_4 = nn.Sequential(*[nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),nn.Sigmoid()])
        
        # self.weight_conc_1.apply(self._init_weights)
        # self.weight_conc_2.apply(self._init_weights)
        # self.weight_conc_3.apply(self._init_weights)
        # self.weight_conc_4.apply(self._init_weights)
        # self.fc1 = nn.Conv2d(96*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc1.apply(self._init_weights)
        # self.fc2 = nn.Conv2d(192*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2.apply(self._init_weights)
        # self.fc3 = nn.Conv2d(2*384, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc3.apply(self._init_weights)
        # self.fc4 = nn.Conv2d(768*2, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4.apply(self._init_weights)
        # self.stem_other_mod = nn.Sequential(*[
        #     nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.SyncBatchNorm(inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ])
        # self.stem_other_mod.apply(self._init_weights)
        # self.conv2_other_mod = nn.Sequential(*[
        #     nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(2 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv2_other_mod.apply(self._init_weights)
        # self.conv3_other_mod = nn.Sequential(*[
        #     nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv3_other_mod.apply(self._init_weights)
        # self.conv4_other_mod = nn.Sequential(*[
        #     nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv4_other_mod.apply(self._init_weights)
        # self.smart_fusion = RoadFormer2Neck(in_channels=[96*2,192*2,2*384,768*2],
        #          out_channels=[96*2,192*2,2*384,768*2],
        #          layer = len([96*2,192*2,2*384,768*2]),
        #          img_scale=(img_size, img_size),
        #          norm_cfg=dict(type='GN', num_groups=32))
        # self.smart_fusion = RoadFormer2Neck(in_channels=[4*inplanes,8*inplanes,16*inplanes,inplanes*32],
        #          out_channels=[4*inplanes,8*inplanes,16*inplanes,inplanes*32],
        #          layer = len([4*inplanes,8*inplanes,16*inplanes,inplanes*32]),
        #          img_scale=(img_size, img_size),
        #          norm_cfg=dict(type='GN', num_groups=32))
        # print(self.smart_fusion)
        # self.simple_fusion=SimpleFusion(in_channels=[inplanes*4,inplanes*8,inplanes*16,inplanes*32],)
        # self.fc1_other_mod = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2_other_mod = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc3_other_mod = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4_other_mod = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
    def _init_weights(self, m):
        # if isinstance(m, nn.ParameterDict):
            # print(m)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x,x_other_mod):
        
        def _inner_forward(x,x_other_mod):
            # c1 = self.stem(x)
            # c2 = self.conv2(c1)
            # c3 = self.conv3(c2)
            # c4 = self.conv4(c3)
            # c1_other_mod=self.stem_other_mod(x_other_mod)
            # c2_other_mod=self.conv2_other_mod(c1_other_mod)
            # c3_other_mod=self.conv3_other_mod(c2_other_mod)
            # c4_other_mod=self.conv4_other_mod(c3_other_mod)
            c1,c2,c3,c4=self.twin_conv(x,x_other_mod)
            # c1_merged=torch.cat((c1,c1_other_mod),dim=1)
            # c2_merged=torch.cat((c2,c2_other_mod),dim=1)
            # c3_merged=torch.cat((c3,c3_other_mod),dim=1)
            # c4_merged=torch.cat((c4,c4_other_mod),dim=1)
            # c1,c2,c3,c4=self.smart_fusion([c1,c2,c3,c4])
            # c1,c2,c3,c4=self.simple_fusion([c1,c2,c3,c4])
            
            # c1_sum,c2_sum,c3_sum,c4_sum=c1[:,:2*self.inplanes]+c1[:,2*self.inplanes:],c2[:,:4*self.inplanes]+c2[:,4*self.inplanes:],c3[:,8*self.inplanes:]+c3[:,:8*self.inplanes],c4[:,:16*self.inplanes]+c4[:,16*self.inplanes:]
            
            # c1_sum=self.fc1_sum(c1_sum)
            # c2_sum=self.fc2_sum(c2_sum)
            # c3_sum=self.fc3_sum(c3_sum)
            # c4_sum=self.fc4_sum(c4_sum)
            # w1_sum,w2_sum,w3_sum,w4_sum=self.weight_sum_1(c1_sum),self.weight_sum_2(c2_sum),self.weight_sum_3(c3_sum),self.weight_sum_4(c4_sum)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
            c1 = self.fc1_(c1)
            c2 = self.fc2_(c2)
            c3 = self.fc3_(c3)
            c4 = self.fc4_(c4)
            # c1,c2,c3,c4=torch.cat((c1,c1_sum),dim=1),torch.cat((c2,c2_sum),dim=1),torch.cat((c3,c3_sum),dim=1),torch.cat((c4,c4_sum),dim=1)
            # c1=self.fc1_comb(c1)
            # c2=self.fc2_comb(c2)
            # c3=self.fc3_comb(c3)
            # c4=self.fc4_comb(c4)
            # w1_conc,w2_conc,w3_conc,w4_conc=self.weight_conc_1(c1),self.weight_conc_2(c2),self.weight_conc_3(c3),self.weight_conc_4(c4)
            # w1_conc,w2_conc,w3_conc,w4_conc=1-w1_sum,1-w2_sum,1-w3_sum,1-w4_sum
            # w1_sum,w2_sum,w3_sum,w4_sum=self.weight_sum_1(torch.cat((c1_sum,c1),dim=1)),self.weight_sum_2(torch.cat((c2_sum,c2),dim=1)),self.weight_sum_3(torch.cat((c3_sum,c3),dim=1)),self.weight_sum_4(torch.cat((c4_sum,c4),dim=1))
            # w1_conc,w2_conc,w3_conc,w4_conc=w1_sum[:,1].unsqueeze(1),w2_sum[:,1].unsqueeze(1),w3_sum[:,1].unsqueeze(1),w4_sum[:,1].unsqueeze(1)
            # c1=c1*w1_conc+c1_sum*w1_sum[:,0].unsqueeze(1)
            # c2=c2*w2_conc+c2_sum*w2_sum[:,0].unsqueeze(1)
            # c3=c3*w3_conc+c3_sum*w3_sum[:,0].unsqueeze(1)
            # c4=c4*w4_conc+c4_sum*w4_sum[:,0].unsqueeze(1)
            
    
            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x,x_other_mod)
        else:
            outs = _inner_forward(x,x_other_mod)
        return outs
