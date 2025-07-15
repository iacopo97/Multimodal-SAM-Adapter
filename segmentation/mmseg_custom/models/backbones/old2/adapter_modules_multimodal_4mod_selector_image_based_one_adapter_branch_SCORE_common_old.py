import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath
# from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F


_logger = logging.getLogger(__name__)


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
    
class SpatialPriorModuleGeneric(nn.Module):
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
        self.spatial_feature = SpatialFeatureGeneric(embed_dim=embed_dim, with_cp=with_cp)

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
            
            c5=self.spatial_feature(c1,c2,c3,c4)
            
            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
            
            # c5=c5.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4, c5
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
    
class SpatialFeatureGeneric(nn.Module):
    def __init__(self, embed_dim=64, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.conv = nn.Sequential(*[
            nn.ReLU(inplace=True), #add Sigmoid
            nn.Conv2d(4*embed_dim, embed_dim, kernel_size=1, stride=1, padding=0, bias=True),
        ])

    def forward(self, c1,c2,c3,c4):
        
        def _inner_forward(c1,c2,c3,c4):
            # c2_orig_shape=c2.shape
            # c3_orig_shape=c3.shape
            # c4_orig_shape=c4.shape
            c2=F.interpolate(c2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c3=F.interpolate(c3, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c4=F.interpolate(c4, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c=torch.cat([c1,c2,c3,c4],dim=1)
            spatial_feature_generic=self.conv(c)
            # mod_selector_gamma2=F.interpolate(mod_selector_gamma1, size=c2_orig_shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma3=F.interpolate(mod_selector_gamma1, size=c3_orig_shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma4=F.interpolate(mod_selector_gamma1, size=c4_orig_shape[-2:], mode='bilinear', align_corners=False)            
            return spatial_feature_generic
        # , mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        
        if self.with_cp and c1.requires_grad and c2.requires_grad and c3.requires_grad and c4.requires_grad:
            outs = cp.checkpoint(_inner_forward, c1,c2,c3,c4)
        else:
            outs = _inner_forward(c1,c2,c3,c4)
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
    
    
    
    

class SpatialPriorModuleModSelector(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, in_channels=3, num_mod=4):
        super().__init__()
        self.with_cp = with_cp
        self.num_mod = num_mod

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
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.mod_selector = ModalitySelector(embed_dim=embed_dim, with_cp=with_cp,num_mod=num_mod)
        
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
            mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4=self.mod_selector(c1,c2,c3,c4)
            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
            
            mod_selector_gamma1=mod_selector_gamma1.view(bs, self.num_mod, -1).transpose(1, 2)  # 4s bs,num_mod, HW/4^2 view is flattening the tensor transpose therefore bs, HW/4^2, num_mod
            mod_selector_gamma2=mod_selector_gamma2.view(bs, self.num_mod, -1).transpose(1, 2) #8s
            mod_selector_gamma3=mod_selector_gamma3.view(bs, self.num_mod, -1).transpose(1, 2) #16s
            mod_selector_gamma4=mod_selector_gamma4.view(bs, self.num_mod, -1).transpose(1, 2) #32s
            
            return mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4,c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
    


class ModalitySelector(nn.Module):
    def __init__(self, embed_dim=64, with_cp=False, num_mod=4):
        super().__init__()
        self.with_cp = with_cp
        self.num_mod = num_mod
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(4*embed_dim, num_mod, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True) #add Sigmoid
        ])

    def forward(self, c1,c2,c3,c4):
        
        def _inner_forward(c1,c2,c3,c4):
            # c2_orig_shape=c2.shape
            # c3_orig_shape=c3.shape
            # c4_orig_shape=c4.shape
            c2=F.interpolate(c2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c3=F.interpolate(c3, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c4=F.interpolate(c4, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c=torch.cat([c1,c2,c3,c4],dim=1)
            mod_selector_gamma1=self.conv1(c)
            # mod_selector_gamma2=F.interpolate(mod_selector_gamma1, size=c2_orig_shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma3=F.interpolate(mod_selector_gamma1, size=c3_orig_shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma4=F.interpolate(mod_selector_gamma1, size=c4_orig_shape[-2:], mode='bilinear', align_corners=False)            
            return mod_selector_gamma1
        # , mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        
        if self.with_cp and c1.requires_grad and c2.requires_grad and c3.requires_grad and c4.requires_grad:
            outs = cp.checkpoint(_inner_forward, c1,c2,c3,c4)
        else:
            outs = _inner_forward(c1,c2,c3,c4)
        return outs
    
    
class ModalityScore(nn.Module):
    def __init__(self, embed_dim=64, with_cp=False, num_mod=4, inter_ch=32):
        super().__init__()
        self.with_cp = with_cp
        self.num_mod = num_mod
        self.score = nn.Sequential(*[
            nn.Conv2d(4*embed_dim, num_mod, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Linear(embed_dim*num_mod,inter_ch),
            nn.GELU(),
            nn.Linear(inter_ch,num_mod),
            # nn.Conv2d(4*embed_dim, num_mod, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(dim=-1) #add Sigmoid
        ])

    def forward(self, c_merged):
        
        def _inner_forward(c_merged):
            mod_score=self.score(c_merged)
            # c2_orig_shape=c2.shape
            # c3_orig_shape=c3.shape
            # c4_orig_shape=c4.shape
            # c2=F.interpolate(c2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            # c3=F.interpolate(c3, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            # c4=F.interpolate(c4, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            # c=torch.cat([c1,c2,c3,c4],dim=1)
            # mod_selector_gamma1=self.conv1(c)
            # mod_selector_gamma2=F.interpolate(mod_selector_gamma1, size=c2_orig_shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma3=F.interpolate(mod_selector_gamma1, size=c3_orig_shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma4=F.interpolate(mod_selector_gamma1, size=c4_orig_shape[-2:], mode='bilinear', align_corners=False)            
            return mod_score
        # , mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        
        if self.with_cp and c_merged.requires_grad:
            outs = cp.checkpoint(_inner_forward, c_merged)
        else:
            outs = _inner_forward(c_merged)
        return outs
    
    
    
class SpatialPriorModuleModSelectorsinglechannel(nn.Module):
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
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.mod_selector = ModalitySelector(embed_dim=embed_dim, with_cp=with_cp,num_mod=1)
        
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
            mod_selector_gamma=self.mod_selector(c1,c2,c3,c4)
            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
            
            # mod_selector_gamma1=mod_selector_gamma1.view(bs, 1, -1).transpose(1, 2)  # 4s bs,num_mod, HW/4^2 view is flattening the tensor transpose therefore bs, HW/4^2, num_mod
            # mod_selector_gamma2=mod_selector_gamma2.view(bs, 1, -1).transpose(1, 2) #8s
            # mod_selector_gamma3=mod_selector_gamma3.view(bs, 1, -1).transpose(1, 2) #16s
            # mod_selector_gamma4=mod_selector_gamma4.view(bs, 1, -1).transpose(1, 2) #32s
            
            return mod_selector_gamma,c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
    