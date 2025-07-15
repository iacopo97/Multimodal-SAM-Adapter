import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath
# from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import numpy as np

_logger = logging.getLogger(__name__)

global DEBUG
DEBUG=True
# if DEBUG:
    # def get_attention_map(img, get_mask=False):
    #     x = transform(img)
    #     x.size()

    #     logits, att_mat = model(x.unsqueeze(0))

    #     att_mat = torch.stack(att_mat).squeeze(1)

    #     # Average the attention weights across all heads.
    #     att_mat = torch.mean(att_mat, dim=1)

    #     # To account for residual connections, we add an identity matrix to the
    #     # attention matrix and re-normalize the weights.
    #     residual_att = torch.eye(att_mat.size(1))
    #     aug_att_mat = att_mat + residual_att
    #     aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    #     # Recursively multiply the weight matrices
    #     joint_attentions = torch.zeros(aug_att_mat.size())
    #     joint_attentions[0] = aug_att_mat[0]

    #     for n in range(1, aug_att_mat.size(0)):
    #         joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #     v = joint_attentions[-1]
    #     grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    #     mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    #     if get_mask:
    #         result = cv2.resize(mask / mask.max(), img.size)
    #     else:        
    #         mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    #         result = (mask * img).astype("uint8")
        
    #     return result


    # # Assuming `attn_weights` and `attn_weights_other_modalities` are the attention weights extracted from the model
    # def visualize_attention(attn_weights, attn_weights_other_modalities=None):
    #     # Convert attention weights to numpy arrays
    #     attn_weights = attn_weights.detach().cpu().numpy()
    #     # Apply PCA to reduce from 1024 channels to 3 channels

    #     pca = PCA(n_components=3)
    #     if attn_weights_other_modalities is not None:
    #         attn_weights_other_modalities = attn_weights_other_modalities.detach().cpu().numpy()
    #         attn_weights_other_modalities_reduced = pca.fit_transform(attn_weights_other_modalities.reshape(-1, 1024)).reshape(64, 64, 3)
    #         attn_weights_other_modalities_reduced = (attn_weights_other_modalities_reduced - attn_weights_other_modalities_reduced.min()) / (attn_weights_other_modalities_reduced.max() - attn_weights_other_modalities_reduced.min())
            
    #     attn_weights_reduced = pca.fit_transform(attn_weights.reshape(-1, 1024)).reshape(64, 64, 3)
    #     attn_weights_reduced=(attn_weights_reduced - attn_weights_reduced.min()) / (attn_weights_reduced.max() - attn_weights_reduced.min())
    #     # Plot attention weights
    #     plt.figure(figsize=(10, 5))
        
    #     # Plot primary attention weights
    #     plt.subplot(1, 2, 1)
    #     # plt.imshow(attn_weights, cmap='viridis')
    #     # attn_weights=attn_weights.reshape(1,64,64,1024)
    #     plt.imsave('attention_rgb.png', attn_weights_reduced, cmap='magma',vmax=1,vmin=0)
    #     # plt.colorbar()
    #     plt.title('Attention Weights')

    #     if attn_weights_other_modalities is not None:
    #         # Plot other modalities attention weights
    #         # attn_weights_other_modalities=attn_weights_other_modalities_reduced.reshape(1,64,64,1024)
    #         plt.subplot(1, 2, 2)
    #         # plt.imshow(attn_weights_other_modalities, cmap='viridis')
    #         # plt.colorbar()
    #         plt.title('Other Modalities Attention Weights')

    #     # plt.show()
    #     plt.imsave('attention_mod.png', attn_weights_other_modalities_reduced, cmap='magma',vmax=1,vmin=0)
    # def get_attention_map(attn_weights, attn_weights_other_modalities=None,img, get_mask=False):
    #     # x = transform(img)
    #     # x.size()

    #     # logits, att_mat = model(x.unsqueeze(0))

    #     att_mat = torch.stack(att_mat).squeeze(1)

    #     # Average the attention weights across all heads.
    #     att_mat = torch.mean(att_mat, dim=1)

    #     # To account for residual connections, we add an identity matrix to the
    #     # attention matrix and re-normalize the weights.
    #     residual_att = torch.eye(att_mat.size(1))
    #     aug_att_mat = att_mat + residual_att
    #     aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    #     # Recursively multiply the weight matrices
    #     joint_attentions = torch.zeros(aug_att_mat.size())
    #     joint_attentions[0] = aug_att_mat[0]

    #     for n in range(1, aug_att_mat.size(0)):
    #         joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #     v = joint_attentions[-1]
    #     grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    #     mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    #     if get_mask:
    #         result = cv2.resize(mask / mask.max(), img.size)
    #     else:        
    #         mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    #         result = (mask * img).astype("uint8")
        
    #     return result

class ADA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0_0 = nn.Linear(dim, dim // 4)
        self.conv0_1 = nn.Linear(dim, dim // 4)
        self.conv = nn.Linear(dim // 4, dim)

    def forward(self, p, x):
        p = self.conv0_0(p)
        x = self.conv0_1(x)
        p1 = p + x
        p1 = self.conv(p1)
        return p1

# class PatchEmbed(nn.Module):
#     """
#     Image to Patch Embedding.
#     """

#     def __init__(
#         self,
#         img_size: int=224,
#         kernel_size: Tuple[int, int] = (16, 16),
#         stride: Tuple[int, int] = (16, 16),
#         padding: Tuple[int, int] = (0, 0),
#         in_chans: int = 3,
#         embed_dim: int = 768,
#     ) -> None:
#         """
#         Args:
#             kernel_size (Tuple): kernel size of the projection layer.
#             stride (Tuple): stride of the projection layer.
#             padding (Tuple): padding size of the projection layer.
#             in_chans (int): Number of input image channels.
#             embed_dim (int): Patch embedding dimension.
#         """
#         super().__init__()
#         #############MY MOD##################
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(kernel_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         # patch_size=kernel_size
#         self.proj = nn.Conv2d(
#             in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.proj(x)
#         Hp, Wp = x.shape[2], x.shape[3]
        
#         # B C H W -> B H W C
#         x = x.permute(0, 2, 3, 1)
#         ##MY mOD######
#         x=x.flatten(1,2)
#         ###############
#         return x,Hp,Wp



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
class Injector_bimodal_ADA(nn.Module):
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
            self.query_norm_other_mod=norm_layer(dim)
            self.feat_norm_other_modalities=norm_layer(dim)
            self.ADA=ADA(dim)
            # self.attn_other_modalities=MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                #  n_points=n_points, ratio=deform_ratio)
            self.gamma_other_modalities=nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        # self.drop_multimodal=DropPath(drop_multimodal_path) if drop_multimodal_path > 0. else nn.Identity()

    
    def forward(self, query, reference_points, feat, feat_other_modalities, spatial_shapes, level_start_index):
        def _inner_forward(query, feat, feat_other_modalities):
            # if 'rgb' in self.modalities_name:
            query_n=self.query_norm(query).type_as(query)
            feat_n=self.feat_norm(feat).type_as(feat)
            attn = self.attn(query_n, reference_points,
                                feat_n, spatial_shapes,
                                level_start_index, None)
            query=query+self.gamma*attn
            if self.num_mod>1:#>1
                feat_other_modalities = self.feat_norm_other_modalities(feat_other_modalities).type_as(feat_other_modalities)
                # attn_other_modalities = self.attn_other_modalities(self.query_norm_other_mod(query), reference_points,
                #             feat_other_modalities, spatial_shapes,
                #             level_start_index, None)
                prompted=self.ADA(self.query_norm_other_mod(query),feat_other_modalities)
                query = query + self.gamma_other_modalities*prompted
            # else:
            #     query=query+ self.gamma*attn
            # if DEBUG:
            #     visualize_attention(attn,attn_other_modalities)
            return query,prompted
            # return attn_cum
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, feat_other_modalities)
        else:
            query = _inner_forward(query, feat, feat_other_modalities)
            
        return query
class Injector_bimodal(nn.Module):
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
            self.query_norm_other_mod=norm_layer(dim)
            self.feat_norm_other_modalities=norm_layer(dim)
            self.attn_other_modalities=MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
            self.gamma_other_modalities=nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        # self.drop_multimodal=DropPath(drop_multimodal_path) if drop_multimodal_path > 0. else nn.Identity()

    
    def forward(self, query, reference_points, feat, feat_other_modalities, spatial_shapes, level_start_index):
        def _inner_forward(query, feat, feat_other_modalities):
            # if 'rgb' in self.modalities_name:
            query_n=self.query_norm(query).type_as(query)
            feat_n=self.feat_norm(feat).type_as(feat)
            attn = self.attn(query_n, reference_points,
                                feat_n, spatial_shapes,
                                level_start_index, None)
            query=query+self.gamma*attn
            if self.num_mod>1:#>1
                feat_other_modalities = self.feat_norm_other_modalities(feat_other_modalities).type_as(feat_other_modalities)
                attn_other_modalities = self.attn_other_modalities(self.query_norm_other_mod(query), reference_points,
                            feat_other_modalities, spatial_shapes,
                            level_start_index, None)
                query = query + self.gamma_other_modalities*attn_other_modalities
            # else:
            #     query=query+ self.gamma*attn
            # if DEBUG:
            #     visualize_attention(attn,attn_other_modalities)
            return query
            # return attn_cum
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, feat_other_modalities)
        else:
            query = _inner_forward(query, feat, feat_other_modalities)
            
        return query


class Extractor_bimodal(nn.Module):
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
            self.query_norm_other_modalities=norm_layer(dim)
            self.attn_other_modalities=MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            new_dim=int(dim * cffn_ratio)
            if 'rgb' in self.modalities_name:
                self.ffn = ConvFFN(in_features=dim, hidden_features=new_dim, drop=drop)
                self.ffn_norm = norm_layer(dim)
            if self.num_mod>1:
                # self.ffn_other_modalities=nn.ModuleDict()
                # self.ffn_norm_other_modalities=nn.ModuleDict()
                self.ffn_other_modalities = ConvFFN(in_features=dim, hidden_features=new_dim, drop=drop)
                self.ffn_norm_other_modalities = norm_layer(dim)
                # for i in range(1,len(self.modalities_name)):
                #     self.ffn_other_modalities.update({f"ffn_{self.modalities_name[i]}":ConvFFN(in_features=dim, hidden_features=new_dim, drop=drop)})
                #     self.ffn_norm_other_modalities.update({f"ffn_norm_{self.modalities_name[i]}":norm_layer(dim)})
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_multimodal = DropPath(drop_multimodal_path) if drop_multimodal_path > 0. else nn.Identity()


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
                query_n_other_modalities = self.query_norm_other_modalities(query_other_modalities)
                attn_other_modalities_t = self.attn_other_modalities(query_n_other_modalities, reference_points,
                             feat_n, spatial_shapes,
                             level_start_index, None)
                query = query + attn
                query_other_modalities = query_other_modalities + attn_other_modalities_t
                if self.with_cffn:
                    query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
                    # cum_extraction_ffn_other_modalities_t = torch.zeros_like(query_other_modalities)
                    # for i in range(1,len(self.modalities_name)):
                    #     cum_extraction_ffn_other_modalities_t[i-1] = cum_extraction_ffn_other_modalities_t[i-1] + self.drop_path(self.ffn_other_modalities[f"ffn_{self.modalities_name[i]}"](self.ffn_norm_other_modalities[f"ffn_norm_{self.modalities_name[i]}"](query_other_modalities[i-1]), H, W))
                    # #     query_other_modalities[i-1] = query_other_modalities[i-1] + self.drop_path(self.ffn_other_modalities[f"ffn_{self.modalities_name[i]}"](self.ffn_norm_other_modalities[f"ffn_norm_{self.modalities_name[i]}"](query_other_modalities[i-1]), H, W))
                    query_other_modalities = query_other_modalities + self.drop_path(self.ffn_other_modalities(self.ffn_norm_other_modalities(query_other_modalities), H, W))
                return query, query_other_modalities
            elif self.num_mod==1:
                query = query + attn
                if self.with_cffn:
                    query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
                return query,None
        
        if self.with_cp and query.requires_grad:
            
            # query, query_other_modalities = _inner_forward(query, query_other_modalities, feat)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
                # with record_function("model_inference"):
            query, query_other_modalities = cp.checkpoint(_inner_forward, query, query_other_modalities, feat)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            query, query_other_modalities = _inner_forward(query, query_other_modalities, feat)
            
        return query, query_other_modalities


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,gamma_init_values=0.,
                 deform_ratio=1.0, num_mod=3, modalities_name=['rgb','depth','event','lidar'],  drop_multimodal_path=0.5, extra_extractor=False, with_cp=False):
        super().__init__()
        self.modalities_name=modalities_name
        self.num_mod=num_mod
        self.injector = Injector_bimodal_ADA(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values, gamma_init_values=gamma_init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,num_mod=self.num_mod, modalities_name=modalities_name,
                                 with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
        # self.extractor = Extractor_bimodal(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
        #                            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
        #                            cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,num_mod=self.num_mod, modalities_name=modalities_name, gamma_init_values=gamma_init_values, with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
        # self.injector_in = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
        #                          n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
        #                          with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        # self.injector_xyz = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
        #                          n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
        #                          with_cp=with_cp)
        # self.extractor_xyz = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
        #                            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
        #                            cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
                # Extractor_bimodal(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                #           with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                #           drop=drop, drop_path=drop_path, num_mod=self.num_mod, modalities_name=modalities_name, gamma_init_values=gamma_init_values, with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, x_other_modalities):
        x,x_other_modalities = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, feat_other_modalities=x_other_modalities, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        # c, c2_4_other_modalities = self.extractor(query=c, query_other_modalities=c2_4_other_modalities, reference_points=deform_inputs2[0],
        #                    feat=x, spatial_shapes=deform_inputs2[1],
        #                    level_start_index=deform_inputs2[2], H=H, W=W)
        c= self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
                # c, c2_4_other_modalities = extractor(query=c, query_other_modalities=c2_4_other_modalities, reference_points=deform_inputs2[0],
                #               feat=x, spatial_shapes=deform_inputs2[1],
                #               level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, x_other_modalities


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
        self.injector = Injector_bimodal(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,gamma_init_values=gamma_init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,num_mod=self.num_mod, modalities_name=modalities_name,
                                 with_cp=with_cp, drop_multimodal_path=drop_multimodal_path)
        self.extractor = Extractor_bimodal(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
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
                Extractor_bimodal(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
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
    

# class InteractionBlock(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                  drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
#                  deform_ratio=1.0, extra_extractor=False, with_cp=False):
#         super().__init__()

#         self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
#                                  n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
#                                  with_cp=False)
#         self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
#                                    norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
#                                    cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=False)
#         if extra_extractor:
#             self.extra_extractors = nn.Sequential(*[
#                 Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
#                           with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
#                           drop=drop, drop_path=drop_path, with_cp=False)
#                 for _ in range(2)
#             ])
#         else:
#             self.extra_extractors = None

#     def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
#         x = self.injector(query=x, reference_points=deform_inputs1[0],
#                           feat=c, spatial_shapes=deform_inputs1[1],
#                           level_start_index=deform_inputs1[2])
#         for idx, blk in enumerate(blocks):
#             x = blk(x, H, W)
#         c = self.extractor(query=c, reference_points=deform_inputs2[0],
#                            feat=x, spatial_shapes=deform_inputs2[1],
#                            level_start_index=deform_inputs2[2], H=H, W=W)
#         if self.extra_extractors is not None:
#             for extractor in self.extra_extractors:
#                 c = extractor(query=c, reference_points=deform_inputs2[0],
#                               feat=x, spatial_shapes=deform_inputs2[1],
#                               level_start_index=deform_inputs2[2], H=H, W=W)
#         return x, c


# class InteractionBlockWithCls(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                  drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
#                  deform_ratio=1.0, extra_extractor=False, with_cp=False):
#         super().__init__()

#         self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
#                                  n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
#                                  with_cp=with_cp)
#         self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
#                                    norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
#                                    cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
#         if extra_extractor:
#             self.extra_extractors = nn.Sequential(*[
#                 Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
#                           with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
#                           drop=drop, drop_path=drop_path, with_cp=with_cp)
#                 for _ in range(2)
#             ])
#         else:
#             self.extra_extractors = None

#     def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
#         x = self.injector(query=x, reference_points=deform_inputs1[0],
#                           feat=c, spatial_shapes=deform_inputs1[1],
#                           level_start_index=deform_inputs1[2])
#         x = torch.cat((cls, x), dim=1)
#         for idx, blk in enumerate(blocks):
#             x = blk(x, H, W)
#         cls, x = x[:, :1, ], x[:, 1:, ]
#         c = self.extractor(query=c, reference_points=deform_inputs2[0],
#                            feat=x, spatial_shapes=deform_inputs2[1],
#                            level_start_index=deform_inputs2[2], H=H, W=W)
#         if self.extra_extractors is not None:
#             for extractor in self.extra_extractors:
#                 c = extractor(query=c, reference_points=deform_inputs2[0],
#                               feat=x, spatial_shapes=deform_inputs2[1],
#                               level_start_index=deform_inputs2[2], H=H, W=W)
#         return x, c, cls
    

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
class SpatialPriorModulewithGammaSpatial(nn.Module):
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
        self.fc1 = nn.Conv2d(inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
        # self.fc3 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)

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
            mod_selector_gamma1=c1[:,-self.num_mod:,:,:] # bs, num_mod, H/4, W/4
            mod_selector_gamma2=c2[:,-self.num_mod:,:,:] # bs, num_mod, H/8, W/8
            mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
            # mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
            mod_selector_gamma4=c4[:,-self.num_mod:,:,:] # bs, num_mod, H/32, W/32
            # mod_selector_gamma_inj=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
            # mod_selector_gamma1=mod_selector_gamma1+F.interpolate(mod_selector_gamma4, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma2=mod_selector_gamma2+F.interpolate(mod_selector_gamma4, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            # mod_selector_gamma3=mod_selector_gamma3+F.interpolate(mod_selector_gamma4, size=c3.shape[-2:], mode='bilinear', align_corners=False) 
            c1=c1[:,:-self.num_mod,:,:]
            c2=c2[:,:-self.num_mod,:,:]
            c3=c3[:,:-self.num_mod,:,:]
            # c3=c3[:,:-self.num_mod,:,:]
            c4=c4[:,:-self.num_mod,:,:]
            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
            
            mod_selector_gamma1=mod_selector_gamma1.view(bs, self.num_mod, -1).transpose(1, 2)  # 4s bs,num_mod, HW/4^2 view is flattening the tensor transpose therefore bs, HW/4^2, num_mod
            mod_selector_gamma2=mod_selector_gamma2.view(bs, self.num_mod, -1).transpose(1, 2) #8s
            mod_selector_gamma3=mod_selector_gamma3.view(bs, self.num_mod, -1).transpose(1, 2) #16s
            mod_selector_gamma4=mod_selector_gamma4.view(bs, self.num_mod, -1).transpose(1, 2) #32s
            # mod_selector_gamma_inj=mod_selector_gamma_inj.view(bs, self.num_mod, -1).transpose(1, 2) #16s
            
            return c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        # , mod_selector_gamma_inj
            # return c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
    


class ModalitySelector(nn.Module):
    def __init__(self, inplanes=64, with_cp=False, in_channels=3, num_mod=4):
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
        self.fc1 = nn.Conv2d(inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
        # self.fc3 = nn.Conv2d(4 * inplanes, embed_dim+self.num_mod, kernel_size=1, stride=1, padding=0, bias=True) #needed for creating the spatial gamma for injector
        self.fc4 = nn.Conv2d(4 * inplanes, self.num_mod, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            mod_selector_gamma1 = self.stem(x)
            mod_selector_gamma2 = self.conv2(mod_selector_gamma1)
            mod_selector_gamma3 = self.conv3(mod_selector_gamma2)
            mod_selector_gamma4 = self.conv4(mod_selector_gamma3)
            mod_selector_gamma1 = self.fc1(mod_selector_gamma1)
            mod_selector_gamma2 = self.fc2(mod_selector_gamma2)
            mod_selector_gamma3 = self.fc3(mod_selector_gamma3)
            mod_selector_gamma4 = self.fc4(mod_selector_gamma4)
            # mod_selector_gamma1=c1[:,-self.num_mod:,:,:] # bs, num_mod, H/4, W/4
            # mod_selector_gamma2=c2[:,-self.num_mod:,:,:] # bs, num_mod, H/8, W/8
            # mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
            # # mod_selector_gamma3=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
            # mod_selector_gamma4=c4[:,-self.num_mod:,:,:] # bs, num_mod, H/32, W/32
            # # mod_selector_gamma_inj=c3[:,-self.num_mod:,:,:] # bs, num_mod, H/16, W/16
            # # mod_selector_gamma1=mod_selector_gamma1+F.interpolate(mod_selector_gamma4, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            # # mod_selector_gamma2=mod_selector_gamma2+F.interpolate(mod_selector_gamma4, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            # # mod_selector_gamma3=mod_selector_gamma3+F.interpolate(mod_selector_gamma4, size=c3.shape[-2:], mode='bilinear', align_corners=False) 
            # c1=c1[:,:-self.num_mod,:,:]
            # c2=c2[:,:-self.num_mod,:,:]
            # c3=c3[:,:-self.num_mod,:,:]
            # # c3=c3[:,:-self.num_mod,:,:]
            # c4=c4[:,:-self.num_mod,:,:]
            bs, dim, _, _ = mod_selector_gamma1.shape
            mod_selector_gamma1 = mod_selector_gamma1.view(bs, dim, -1).transpose(1, 2)  # 4s 2,1024, HW/4^2 view is flattening the tensor transpose therefore 2, HW/4^2, 1024
            mod_selector_gamma2 = mod_selector_gamma2.view(bs, dim, -1).transpose(1, 2)  # 8s
            mod_selector_gamma3 = mod_selector_gamma3.view(bs, dim, -1).transpose(1, 2)  # 16s
            mod_selector_gamma4 = mod_selector_gamma4.view(bs, dim, -1).transpose(1, 2)  # 32s
            
            # mod_selector_gamma1=mod_selector_gamma1.view(bs, self.num_mod, -1).transpose(1, 2)  # 4s bs,num_mod, HW/4^2 view is flattening the tensor transpose therefore bs, HW/4^2, num_mod
            # mod_selector_gamma2=mod_selector_gamma2.view(bs, self.num_mod, -1).transpose(1, 2) #8s
            # mod_selector_gamma3=mod_selector_gamma3.view(bs, self.num_mod, -1).transpose(1, 2) #16s
            # mod_selector_gamma4=mod_selector_gamma4.view(bs, self.num_mod, -1).transpose(1, 2) #32s
            # mod_selector_gamma_inj=mod_selector_gamma_inj.view(bs, self.num_mod, -1).transpose(1, 2) #16s
            
            return mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        # , mod_selector_gamma_inj
            # return c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs