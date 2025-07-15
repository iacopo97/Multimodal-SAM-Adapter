import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
# from ops.modules import MSDeformAttn
import os
# if os.getcwd()=='/media/data2/icurti/projects/sina/ViT-Adapter/segmentation' or os.getcwd()=='/media/data2/icurti/projects/sina/ViT-Adapter/segmentation/mmseg_custom/models/backbones':
if '/segmentation' in os.getcwd():
    from ops.modules import MSDeformAttn
else:
    from ViTAdapter.segmentation.ops.modules import MSDeformAttn
from timm.models.layers import DropPath
 
# torch.autograd.set_detect_anomaly(True)
# from torch.profiler import profile, record_function, ProfilerActivity
_logger = logging.getLogger(__name__)

from .adapter_modules import get_reference_points, deform_inputs, ConvFFN, DWConv
# def get_reference_points(spatial_shapes, device):
#     reference_points_list = []
#     for lvl, (H_, W_) in enumerate(spatial_shapes):
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#             torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
#         #overall it is a meshgrid of two matrices  40x40 and 40x40 one that is going to increment along x directiona nd other one across y dimension
#         ref_y = ref_y.reshape(-1)[None] / H_
#         ref_x = ref_x.reshape(-1)[None] / W_ #normalization and reshaping(flattening)
#         ref = torch.stack((ref_x, ref_y), -1)
#         reference_points_list.append(ref)
#     reference_points = torch.cat(reference_points_list, 1)
#     reference_points = reference_points[:, :, None]
#     return reference_points


# def deform_inputs(x):
#     bs, c, h, w = x.shape
#     spatial_shapes = torch.as_tensor([(h // 8, w // 8),
#                                       (h // 16, w // 16),
#                                       (h // 32, w // 32)],
#                                      dtype=torch.long, device=x.device)
#     level_start_index = torch.cat((spatial_shapes.new_zeros(
#         (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#     reference_points = get_reference_points([(h // 16, w // 16)], x.device)
#     deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
#     spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
#     level_start_index = torch.cat((spatial_shapes.new_zeros(
#         (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#     reference_points = get_reference_points([(h // 8, w // 8),
#                                              (h // 16, w // 16),
#                                              (h // 32, w // 32)], x.device)
#     deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
#     return deform_inputs1, deform_inputs2


# class ConvFFN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None,
#                  act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim) #means each input channels is processed by its specific set of filters depthwise separable convolution

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         n = N // 21 #group convolutions
#         x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
#         x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
#         x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
#         x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
#         x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
#         x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
#         x = torch.cat([x1, x2, x3], dim=1)
#         return x

# class Extractor(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
#                  with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
#                  norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
#         super().__init__()
#         self.query_norm = norm_layer(dim)
#         self.feat_norm = norm_layer(dim)
#         self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
#                                  n_points=n_points, ratio=deform_ratio)
#         self.with_cffn = with_cffn
#         self.with_cp = with_cp
#         if with_cffn:
#             self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
#             self.ffn_norm = norm_layer(dim)
#             self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
#         def _inner_forward(query, feat):

#             attn = self.attn(self.query_norm(query), reference_points,
#                              self.feat_norm(feat), spatial_shapes,
#                              level_start_index, None)
#             query = query + attn
    
#             if self.with_cffn:
#                 query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
#             return query
        
#         if self.with_cp and query.requires_grad:
#             query = cp.checkpoint(_inner_forward, query, feat)
#         else:
#             query = _inner_forward(query, feat)
            
#         return query


class Injector_multimodal(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.,gamma_init_values=0, num_mod=3,modalities_name=['rgb','depth','event','lidar'],with_cp=False, drop_multimodal_path=0.5):
        super().__init__()
        self.with_cp = with_cp
        self.num_mod=num_mod
        self.query_norm = norm_layer(dim)
        self.modalities_name = modalities_name
        if "rgb" in self.modalities_name:
            self.feat_norm = norm_layer(dim)
            self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
            self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        if self.num_mod>1:
            self.feat_norm_other_modalities=nn.ModuleDict()
            self.attn_other_modalities=nn.ModuleDict()
            self.gamma_other_modalities=nn.ParameterDict()
            for i in range(1,len(self.modalities_name)):
                self.feat_norm_other_modalities.update({f"feat_norm_{self.modalities_name[i]}":norm_layer(dim)})
                self.attn_other_modalities.update({f"attn_{self.modalities_name[i]}":MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)})
                self.gamma_other_modalities.update({f"gamma_{self.modalities_name[i]}":nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)})
        self.drop_multimodal=DropPath(drop_multimodal_path) if drop_multimodal_path > 0. else nn.Identity()

    
    def forward(self, query, reference_points, feat, feat_other_modalities, spatial_shapes, level_start_index):
        # if len(query.shape)==4:
        #     query=query.flatten(1,2)
        # query_n=self.query_norm(query)
        # feat_n=self.feat_norm(feat)
        # feat_other_modalities_l=list()
        # for i in range(1,len(self.modalities_name)):
        #     feat_other_modalities_l.append(feat_other_modalities[f"c2_4_{self.modalities_name[i]}"].unsqueeze(0))
        # feat_other_modalities_tens=torch.cat(feat_other_modalities_l,0)
        def _inner_forward(query, feat, feat_other_modalities):
            # if 'rgb' in self.modalities_name:
            query_n=self.query_norm(query).type_as(query)
            feat_n=self.feat_norm(feat).type_as(feat)
            attn = self.attn(query_n, reference_points,
                                feat_n, spatial_shapes,
                                level_start_index, None)
            if self.num_mod>=1:#>1
                attn_cum=0
                # feat_n_other_modalities = self.feat_norm_other_modalities[f"feat_norm_{self.modalities_name[i]}"](feat_other_modalities)
                # attn_other_modalities = self.attn_other_modalities[f"attn_{self.modalities_name[i]}"](query_n, reference_points,
                #         feat_n_other_modalities, spatial_shapes,
                #         level_start_index, None)
                # attn_cum=self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}"]*attn_other_modalities)
                for i in range(1,len(self.modalities_name)):
                    # feat_n_other_modalities = self.feat_norm_other_modalities[f"feat_norm_{self.modalities_name[i]}"](feat_other_modalities[f"c2_4_{self.modalities_name[i]}"])
                    feat_n_other_modalities = self.feat_norm_other_modalities[f"feat_norm_{self.modalities_name[i]}"](feat_other_modalities[i-1]).type_as(feat_other_modalities[i-1])
                    attn_other_modalities = self.attn_other_modalities[f"attn_{self.modalities_name[i]}"](query_n, reference_points,
                            feat_n_other_modalities, spatial_shapes,
                            level_start_index, None)
                    attn_cum = attn_cum + self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}"].type_as(attn_other_modalities)*attn_other_modalities)
                # if 'rgb' in self.modalities_name:
                query=query+self.gamma.type_as(query)*attn+attn_cum
                    # query=query+self.gamma*attn+attn_other_modalities
                    # attn_cum=attn_cum+self.gamma*attn
                # else:              
                    # query=query+attn_cum
                    # attn_cum=attn_cum
            # elif self.num_mod==1 and 'rgb' in self.modalities_name: query=query+self.gamma*attn
            return query
            # return attn_cum
        
        if self.with_cp and query.requires_grad:
            # query = cp.checkpoint(_inner_forward, query, feat, feat_other_modalities_list)

            # query = cp.checkpoint(_inner_forward, query, feat, feat_other_modalities[f"c2_4_{self.modalities_name[1]}"])
            # query = cp.checkpoint(_inner_forward, query, feat, feat_other_modalities)
            # query = _inner_forward(query, feat, feat_other_modalities)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
            #     with record_function("model_inference"):
            query = cp.checkpoint(_inner_forward, query, feat, feat_other_modalities)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            # for i in range(1,len(self.modalities_name)):
            #     attn_cum = cp.checkpoint(_inner_forward, query_n, feat_n, feat_other_modalities[f"c2_4_{self.modalities_name[i]}"],i)
            #     query=query+attn_cum
            # attn_cum=cp.checkpoint(_inner_forward, query, feat, feat_other_modalities[f"c2_4_{self.modalities_name[1]}"])
            
        else:
            query = _inner_forward(query, feat, feat_other_modalities)
            # for i in range(1,len(self.modalities_name)):
            #     attn_cum = _inner_forward(query_n, feat_n, feat_other_modalities[f"c2_4_{self.modalities_name[i]}"],i)
            #     query=query+attn_cum
            
        return query


class Extractor_multimodal(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), num_mod=3,modalities_name=['rgb','depth','event','lidar'], gamma_init_values=0., with_cp=False, drop_multimodal_path=0.5):
        super().__init__()
        self.feat_norm = norm_layer(dim)
        # self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
        #                          n_points=n_points, ratio=deform_ratio)
        self.num_mod=num_mod
        self.modalities_name=modalities_name
        if 'rgb' in self.modalities_name:
            self.query_norm = norm_layer(dim)
            self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        if self.num_mod>1:
            self.query_norm_other_modalities=nn.ModuleDict()
            self.attn_other_modalities=nn.ModuleDict()
            self.gamma_other_modalities=nn.ParameterDict()
            for i in range(0,len(self.modalities_name)):
                if i != 0:
                    self.query_norm_other_modalities.update({f"query_norm_{self.modalities_name[i]}":norm_layer(dim)})
                    self.attn_other_modalities.update({f"attn_{self.modalities_name[i]}":MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                    n_points=n_points, ratio=deform_ratio)})
                for j in range(0,len(self.modalities_name)):
                    if self.modalities_name[i]!=self.modalities_name[j]:
                        self.gamma_other_modalities.update({f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}":nn.Parameter(gamma_init_values * torch.ones((dim)), requires_grad=True)})
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            new_dim=int(dim * cffn_ratio)
            if 'rgb' in self.modalities_name:
                self.ffn = ConvFFN(in_features=dim, hidden_features=new_dim, drop=drop)
                self.ffn_norm = norm_layer(dim)
            if self.num_mod>1:
                self.ffn_other_modalities=nn.ModuleDict()
                self.ffn_norm_other_modalities=nn.ModuleDict()
                for i in range(1,len(self.modalities_name)):
                    self.ffn_other_modalities.update({f"ffn_{self.modalities_name[i]}":ConvFFN(in_features=dim, hidden_features=new_dim, drop=drop)})
                    self.ffn_norm_other_modalities.update({f"ffn_norm_{self.modalities_name[i]}":norm_layer(dim)})
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_multimodal=DropPath(drop_multimodal_path) if drop_multimodal_path > 0. else nn.Identity()


    def forward(self, query, query_other_modalities,reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, query_other_modalities, feat):
            # query_other_modalities=torch.clone(query_other_modalities)
            # if 'rgb' in self.modalities_name:
            query_n=self.query_norm(query).type_as(query)
            feat_n=self.feat_norm(feat).type_as(feat)  
            attn = self.attn(query_n, reference_points,
                             feat_n, spatial_shapes,
                             level_start_index, None)
            if self.num_mod>1:
                for i in range(1,len(self.modalities_name)):
                    # query_n_other_modalities=self.query_norm_other_modalities[f"query_norm_{self.modalities_name[i]}"](query_other_modalities[f"c2_4_{self.modalities_name[i]}"])
                    if i==2:
                        prev_attn_other_modalities_t=torch.clone(attn_other_modalities_t).unsqueeze(0)
                    query_n_other_modalities=self.query_norm_other_modalities[f"query_norm_{self.modalities_name[i]}"](query_other_modalities[i-1]).type_as(query_other_modalities[i-1])    
                    attn_other_modalities_t = self.attn_other_modalities[f"attn_{self.modalities_name[i]}"](query_n_other_modalities, reference_points,
                             feat_n, spatial_shapes,
                             level_start_index, None)
                    # attn_cum.update({f"attn_mod_{self.modalities_name[i]}":attn_other_modalities_t})
                    if i>1:
                        attn_cum=torch.cat((prev_attn_other_modalities_t,attn_other_modalities_t.unsqueeze(0)),0)
                        prev_attn_other_modalities_t=torch.clone(attn_cum)
                cum_extraction_primary_modality=torch.zeros_like(attn)
                cum_extraction_other_modalities_t=torch.zeros_like(query_other_modalities)
                for i in range(0,len(self.modalities_name)):
                    for j in range(0,len(self.modalities_name)):
                        if self.modalities_name[i]!=self.modalities_name[j]:
                            if self.modalities_name[i]=='rgb':
                                cum_extraction_primary_modality = cum_extraction_primary_modality + attn + self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"].type_as(attn_cum[j-1])*attn_cum[j-1])
                                # query=query+ attn+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"]*attn_cum[j-1])
                                # query=query+ attn+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"]*attn_cum[f"attn_mod_{self.modalities_name[j]}"])
                            elif self.modalities_name[j]=='rgb':
                                cum_extraction_other_modalities_t[i-1] = cum_extraction_other_modalities_t[i-1] + attn_cum[i-1]+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"].type_as(attn)*attn)
                            #     # query_other_modalities[i-1]= query_other_modalities[i-1]+attn_cum[i-1]+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"]*attn)
                            #     # query_other_modalities[i-1]=query_other_modalities[i-1]+attn_cum[f"attn_mod_{self.modalities_name[i]}"]+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"]*attn)
                            #     # pass
                            else:
                                cum_extraction_other_modalities_t[i-1] = cum_extraction_other_modalities_t[i-1] + attn_cum[i-1]+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"].type_as(attn)*attn_cum[j-1])
                            #     # query_other_modalities[i-1]= query_other_modalities[i-1]+attn_cum[i-1]+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"]*attn_cum[j-1])
                            #     # query_other_modalities[i-1]=query_other_modalities[i-1]+attn_cum[f"attn_mod_{self.modalities_name[i]}"]+self.drop_multimodal(self.gamma_other_modalities[f"gamma_{self.modalities_name[i]}_{self.modalities_name[j]}"]*attn_cum[f"attn_mod_{self.modalities_name[j]}"])
                            #     # pass
                query = query + cum_extraction_primary_modality
                query_other_modalities = query_other_modalities + cum_extraction_other_modalities_t
                if self.with_cffn:
                    query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
                    cum_extraction_ffn_other_modalities_t = torch.zeros_like(query_other_modalities)
                    for i in range(1,len(self.modalities_name)):
                        cum_extraction_ffn_other_modalities_t[i-1] = cum_extraction_ffn_other_modalities_t[i-1] + self.drop_path(self.ffn_other_modalities[f"ffn_{self.modalities_name[i]}"](self.ffn_norm_other_modalities[f"ffn_norm_{self.modalities_name[i]}"](query_other_modalities[i-1]), H, W))
                    #     query_other_modalities[i-1] = query_other_modalities[i-1] + self.drop_path(self.ffn_other_modalities[f"ffn_{self.modalities_name[i]}"](self.ffn_norm_other_modalities[f"ffn_norm_{self.modalities_name[i]}"](query_other_modalities[i-1]), H, W))
                    query_other_modalities=query_other_modalities+cum_extraction_ffn_other_modalities_t
                return query, query_other_modalities
            elif self.num_mod==1:
                query = query + attn
    
                if self.with_cffn:
                    query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
                return query,None
            # return query, query_other_modalities
            # if self.with_cffn:
            #     query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            # return query
        
        if self.with_cp and query.requires_grad:
            
            # query, query_other_modalities = _inner_forward(query, query_other_modalities, feat)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
                # with record_function("model_inference"):
            query, query_other_modalities = cp.checkpoint(_inner_forward, query, query_other_modalities, feat)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            query, query_other_modalities = _inner_forward(query, query_other_modalities, feat)
            
        return query, query_other_modalities


# class Injector(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
#                  norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
#         super().__init__()
#         self.with_cp = with_cp
#         self.query_norm = norm_layer(dim)
#         self.feat_norm = norm_layer(dim)
#         self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
#                                  n_points=n_points, ratio=deform_ratio)
#         self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

#     def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
#         # if len(query.shape)==4:
#         #     query=query.flatten(1,2)
#         def _inner_forward(query, feat):

#             attn = self.attn(self.query_norm(query), reference_points,
#                              self.feat_norm(feat), spatial_shapes,
#                              level_start_index, None)
#             return query + self.gamma * attn
        
#         if self.with_cp and query.requires_grad:
#             query = cp.checkpoint(_inner_forward, query, feat)
#         else:
#             query = _inner_forward(query, feat)
            
#         return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,gamma_init_values=0.,
                 deform_ratio=1.0, num_mod=3, modalities_name=['rgb','depth','event','lidar'],  drop_multimodal_path=0.5, extra_extractor=False, with_cp=False):
        super().__init__()
        self.modalities_name=modalities_name
        self.num_mod=num_mod
        self.injector = Injector_multimodal(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values, gamma_init_values=gamma_init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,num_mod=self.num_mod, modalities_name=modalities_name,
                                 with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
        self.extractor = Extractor_multimodal(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,num_mod=self.num_mod, modalities_name=modalities_name, gamma_init_values=gamma_init_values, with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
        # self.injector_in = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
        #                          n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
        #                          with_cp=with_cp)
        # self.extractor_in = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
        #                            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
        #                            cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        # self.injector_xyz = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
        #                          n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
        #                          with_cp=with_cp)
        # self.extractor_xyz = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
        #                            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
        #                            cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor_multimodal(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, num_mod=self.num_mod, modalities_name=modalities_name, gamma_init_values=gamma_init_values, with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, c2_4_other_modalities):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, feat_other_modalities=c2_4_other_modalities, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c, c2_4_other_modalities = self.extractor(query=c, query_other_modalities=c2_4_other_modalities, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c, c2_4_other_modalities = extractor(query=c, query_other_modalities=c2_4_other_modalities, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, c2_4_other_modalities


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,gamma_init_values=0.,
                 deform_ratio=1.0, num_mod=3, modalities_name=['rgb','depth','event','lidar'],  drop_multimodal_path=0.5, extra_extractor=False, with_cp=False):
        super().__init__()
        self.modalities_name=modalities_name
        self.num_mod=num_mod
        # self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
        #                          n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
        #                          with_cp=with_cp)
        # self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
        #                            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
        #                            cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        self.injector = Injector_multimodal(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,gamma_init_values=gamma_init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,num_mod=self.num_mod, modalities_name=modalities_name,
                                 with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
        self.extractor = Extractor_multimodal(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,num_mod=self.num_mod, modalities_name=modalities_name, gamma_init_values=gamma_init_values, with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
        if extra_extractor:
            # self.extra_extractors = nn.Sequential(*[
            #     Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
            #               with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
            #               drop=drop, drop_path=drop_path, with_cp=with_cp)
            #     for _ in range(2)
            # ])
            self.extra_extractors = nn.Sequential(*[
                Extractor_multimodal(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, num_mod=self.num_mod, modalities_name=modalities_name, gamma_init_values=gamma_init_values, with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W, c2_4_other_modalities):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, feat_other_modalities=c2_4_other_modalities,spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c, c2_4_other_modalities = self.extractor(query=c, query_other_modalities=c2_4_other_modalities, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c, c2_4_other_modalities = extractor(query=c, query_other_modalities=c2_4_other_modalities, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, c2_4_other_modalities, cls
    

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