# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
# from ops.modules import MSDeformAttn
import os
# if os.getcwd()=='/media/data2/icurti/projects/sina/ViT-Adapter/segmentation':
if '/segmentation' in os.getcwd():
    from ops.modules import MSDeformAttn
else:
    from ViTAdapter.segmentation.ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

# from .base.vit import TIMMVisionTransformer
from .base.image_encoder import ImageEncoderViT
from .adapter_modules_multimodal_4mod_selector_image_based_one_adapter_branch import SpatialPriorModule, SpatialPriorModulewithGammaSpatial, InteractionBlock, deform_inputs
from functools import partial
from timm.models.layers import DropPath
# torch.autograd.set_detect_anomaly(True)
#ONLY STOCHASTIC DEPTH and at the end and in the injectors and extractors
_logger = logging.getLogger(__name__)
#  backbone=dict( 
#         _delete_=True,
#         type='SAMAdapter',
#         img_size=crop_size[0],
#         # pretrain_size=640,
#         patch_size=16,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         mlp_ratio=4,
#         drop_path_rate=0.4,
#         conv_inplane=64,
#         n_points=4,
#         deform_num_heads=16,
#         cffn_ratio=0.25,
#         deform_ratio=0.5,
#         with_cp=True,  # set with_cp=True to save memory
#         interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
#         window_attn=[False] * 24,
#         window_size=[None] * 24
#         ), 
# COUNTER=0
@BACKBONES.register_module()
class SAMAdapterbimodal4modselectoronebranch(ImageEncoderViT):
    def __init__(self, pretrain_size=1024 #1024
                 , num_heads=12, conv_inplane=64, n_points=4,#um_heads12
                 modalities_name=['rgb','depth','lidar','event'], modalities_ch=[3,3,3,1],
                 deform_num_heads=6, init_values=0.,gamma_init_values=0., interaction_indexes=None, with_cffn=True,#def6
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,#deform_ratio1.0
                 use_extra_extractor=True, with_cp=True, drop_path_rate=0.4,drop_rate=0., drop_multimodal_path=0.2, *args, **kwargs):
        
        super().__init__(num_heads=num_heads, pretrained=pretrained, #,
                         with_cp=with_cp,*args,**kwargs) #with_cp=with_cp *args, **kwargs
        
        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        self.num_mod=len(modalities_name)
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch
        # self.gamma_init_values=gamma_init_values
        if ('rgb' in self.modalities_name) and self.num_mod>1:
            self.in_ch_im=self.modalities_ch[self.modalities_name.index('rgb')]
            # self.spm_gamma_spatial = SpatialPriorModulewithGammaSpatial(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, in_channels=self.in_ch_im, num_mod=self.num_mod)
            self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, in_channels=self.in_ch_im)
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
            ##INITIALIZE THEM###
            self.up.apply(self._init_weights)
            self.spm.apply(self._init_weights)
            # self.spm_gamma_spatial.apply(self._init_weights)
            ###################
        elif ('rgb' in self.modalities_name) and self.num_mod==1:
            self.in_ch_im=self.modalities_ch[self.modalities_name.index('rgb')]
            self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, in_channels=self.in_ch_im)
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
            ##INITIALIZE THEM###
            self.up.apply(self._init_weights)
            self.spm.apply(self._init_weights)
            ###################
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.drop_path_rate=drop_path_rate
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.multimodal_drop_path=DropPath(drop_multimodal_path) if drop_multimodal_path > 0. else nn.Identity() #DropPath is the stochastic Depth, so it is pruning a modality instead of another turning all the elements in zeros
        self.norm_layer=partial(nn.LayerNorm, eps=1e-6)
        if self.num_mod>1:
        #     self.spm_other_modalities=nn.ModuleDict()
        #     # self.up_other_modalities=nn.ModuleDict()
        #     # self.gamma_1_other_modalities=nn.ParameterDict()
        #     # self.gamma_2_other_modalities=nn.ParameterDict()
        #     for i in range(1,len(self.modalities_name)):
        #         self.spm_other_modalities.update({f"spm_{self.modalities_name[i]}":SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, in_channels=self.modalities_ch[i])})
        #         # self.up_other_modalities.update({f"up_{self.modalities_name[i]}":nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)})
        #         # self.gamma_1_other_modalities.update({f"gamma_1_{self.modalities_name[i]}":nn.Parameter(gamma_init_values * torch.ones((embed_dim)), requires_grad=True)})
        #         # self.gamma_2_other_modalities.update({f"gamma_2_{self.modalities_name[i]}":nn.Parameter(gamma_init_values * torch.ones((embed_dim)), requires_grad=True)})
            self.spm_other_modalities=SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, in_channels=self.modalities_ch[1])
            self.relu_mod1=nn.ReLU(inplace=True)
            self.relu_mod2=nn.ReLU(inplace=True)
            self.relu_mod3=nn.ReLU(inplace=True)
            self.relu_mod4=nn.ReLU(inplace=True)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values,drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.norm1 = nn.SyncBatchNorm(embed_dim) #in the model.statedict the norm layer will be called backbone.norm.1
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)
        
        # self.norm_mod1 = partial(nn.LayerNorm, eps=1e-6)(self.num_mod)
        # self.norm_mod2 = partial(nn.LayerNorm, eps=1e-6)(self.num_mod)
        # self.norm_mod3 = partial(nn.LayerNorm, eps=1e-6)(self.num_mod)
        # self.norm_mod4 = partial(nn.LayerNorm, eps=1e-6)(self.num_mod)

        if self.num_mod>1:
            # for i in range(1,len(self.modalities_name)):
                # self.up_other_modalities[f"up_{self.modalities_name[i]}"].apply(self._init_weights)
                # self.spm_other_modalities[f"spm_{self.modalities_name[i]}"].apply(self._init_weights)
            self.spm_other_modalities.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

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

    def _get_pos_embed1(self, pos_embed, H, W):
        # pos_embed = pos_embed.reshape(
        #     1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed=pos_embed.permute(0,3,1,2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)
        # .\
            # reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        # global COUNTER
        c2 = c2 + self.level_embed[0].type_as(c2)
        # COUNTER+=1
        c3 = c3 + self.level_embed[1].type_as(c3)
        # COUNTER+=1
        c4 = c4 + self.level_embed[2].type_as(c4) 
        # COUNTER+=1
        # if COUNTER==3*self.num_mod:
        #     COUNTER=0
        return c2, c3, c4

    def forward(self, x):
        try:
            x_other_modalities=x[:,self.in_ch_im:]
        except:
            pass
        # modalities_ch_cumsum=torch.cumsum(torch.tensor(self.modalities_ch),0)-self.in_ch_im
        
        x=x[:,:self.in_ch_im]
        if ('rgb' in self.modalities_name):
            # if self.num_mod>1:
            #     # SPM forward
            #     c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4 = self.spm(x)
            #     # c1, c2, c3, c4, mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4 = self.spm_gamma_spatial(x)
            #     # mod_selector_gamma=torch.cat([mod_selector_gamma2, mod_selector_gamma3, mod_selector_gamma4], dim=1) #bs, (HW/8^2+HW/16^2+HW/32^2), num_mod
            #     # mod_selector_gamma1=self.norm_mod1(mod_selector_gamma1)
            #     # mod_selector_gamma2=self.norm_mod2(mod_selector_gamma2)
            #     # mod_selector_gamma3=self.norm_mod3(mod_selector_gamma3)
            #     #smod_selector_gamma2,mod_selector_gamma3,mod_selector_gamma4=self._add_level_embed(mod_selector_gamma2,mod_selector_gamma3,mod_selector_gamma4)
            #     # mod_selector_gamma1=F.normalize(mod_selector_gamma1, p=2, dim=1)
            #     # mod_selector_gamma2=F.normalize(mod_selector_gamma2, p=2, dim=1)
            #     # mod_selector_gamma3=F.normalize(mod_selector_gamma3, p=2, dim=1)
            #     # mod_selector_gamma4=F.normalize(mod_selector_gamma4, p=2, dim=1)
            #     # mod_selector_gamma4=self.norm_mod4(mod_selector_gamma4)
            #     # mod_selector_gamma1=(mod_selector_gamma1)/(mod_selector_gamma1.sum(dim=-1,keepdim=True)+1e-12)
            #     # mod_selector_gamma2=(mod_selector_gamma2)/(mod_selector_gamma2.sum(dim=-1,keepdim=True)+1e-12)
            #     # mod_selector_gamma3=(mod_selector_gamma3)/(mod_selector_gamma3.sum(dim=-1,keepdim=True)+1e-12)
            #     # mod_selector_gamma4=(mod_selector_gamma4)/(mod_selector_gamma4.sum(dim=-1,keepdim=True)+1e-12)
            #     mod_selector_gamma1=self.relu_mod1(mod_selector_gamma1)
            #     mod_selector_gamma2=self.relu_mod2(mod_selector_gamma2)
            #     mod_selector_gamma3=self.relu_mod3(mod_selector_gamma3)
            #     mod_selector_gamma4=self.relu_mod4(mod_selector_gamma4)
            #     mod_selector_gamma1=(mod_selector_gamma1)/torch.max(mod_selector_gamma1.sum(dim=-1,keepdim=True), 1e-12*torch.ones_like(mod_selector_gamma1.sum(dim=-1,keepdim=True)))
            #     mod_selector_gamma2=(mod_selector_gamma2)/torch.max(mod_selector_gamma2.sum(dim=-1,keepdim=True), 1e-12*torch.ones_like(mod_selector_gamma2.sum(dim=-1,keepdim=True)))
            #     mod_selector_gamma3=(mod_selector_gamma3)/torch.max(mod_selector_gamma3.sum(dim=-1,keepdim=True), 1e-12*torch.ones_like(mod_selector_gamma3.sum(dim=-1,keepdim=True)))
            #     mod_selector_gamma4=(mod_selector_gamma4)/torch.max(mod_selector_gamma4.sum(dim=-1,keepdim=True), 1e-12*torch.ones_like(mod_selector_gamma4.sum(dim=-1,keepdim=True)))
            #     # mod_selector_gamma1=(mod_selector_gamma1)/(mod_selector_gamma1.sum(dim=1,keepdim=True)+1e-12)
            #     # mod_selector_gamma2=(mod_selector_gamma2)/(mod_selector_gamma2.sum(dim=1,keepdim=True)+1e-12)
            #     # mod_selector_gamma3=(mod_selector_gamma3)/(mod_selector_gamma3.sum(dim=1,keepdim=True)+1e-12)
            #     # mod_selector_gamma4=(mod_selector_gamma4)/(mod_selector_gamma4.sum(dim=1,keepdim=True)+1e-12)
            #     # mod_selector_gamma1=self.relu_mod1(mod_selector_gamma1)
            #     # mod_selector_gamma2=self.relu_mod2(mod_selector_gamma2)
            #     # mod_selector_gamma3=self.relu_mod3(mod_selector_gamma3)
            #     # mod_selector_gamma4=self.relu_mod4(mod_selector_gamma4)
            #     # mod_selector_gamma1=(mod_selector_gamma1-mod_selector_gamma1.min(dim=1,keepdim=True).values)/(mod_selector_gamma1.max(dim=1,keepdim=True).values-mod_selector_gamma1.min(dim=1,keepdim=True).values+1e-12)
            #     # mod_selector_gamma2=(mod_selector_gamma2-mod_selector_gamma2.min(dim=1,keepdim=True).values)/(mod_selector_gamma2.max(dim=1,keepdim=True).values-mod_selector_gamma2.min(dim=1,keepdim=True).values+1e-12)
            #     # mod_selector_gamma3=(mod_selector_gamma3-mod_selector_gamma3.min(dim=1,keepdim=True).values)/(mod_selector_gamma3.max(dim=1,keepdim=True).values-mod_selector_gamma3.min(dim=1,keepdim=True).values+1e-12)
            #     # mod_selector_gamma4=(mod_selector_gamma4-mod_selector_gamma4.min(dim=1,keepdim=True).values)/(mod_selector_gamma4.max(dim=1,keepdim=True).values-mod_selector_gamma4.min(dim=1,keepdim=True).values+1e-12)
            #     # sum_mod_selector_gamma1=mod_selector_gamma1.sum(dim=1,keepdim=True)
            #     # sum_mod_selector_gamma2=mod_selector_gamma2.sum(dim=1,keepdim=True)
            #     # sum_mod_selector_gamma3=mod_selector_gamma3.sum(dim=1,keepdim=True)
            #     # sum_mod_selector_gamma4=mod_selector_gamma4.sum(dim=1,keepdim=True)
            #     # mod_selector_gamma1=mod_selector_gamma1/sum_mod_selector_gamma1
            #     # mod_selector_gamma2=mod_selector_gamma2/sum_mod_selector_gamma2
            #     # mod_selector_gamma3=mod_selector_gamma3/sum_mod_selector_gamma3
            #     # mod_selector_gamma4=mod_selector_gamma4/sum_mod_selector_gamma4
            #     # mod_selector_gamma1=F.normalize(mod_selector_gamma1, p=2, dim=1)
            #     # mod_selector_gamma2=F.normalize(mod_selector_gamma2, p=2, dim=1)
            #     # mod_selector_gamma3=F.normalize(mod_selector_gamma3, p=2, dim=1)
            #     # mod_selector_gamma4=F.normalize(mod_selector_gamma4, p=2, dim=1)
            #     c1, c2, c3, c4 = mod_selector_gamma1[:,:,0].unsqueeze(-1)*c1, mod_selector_gamma2[:,:,0].unsqueeze(-1)*c2, mod_selector_gamma3[:,:,0].unsqueeze(-1)*c3, mod_selector_gamma4[:,:,0].unsqueeze(-1)*c4 
            # else:
            c1, c2, c3, c4 = self.spm(x)
            c2, c3, c4 = self._add_level_embed(c2, c3, c4)
            c = torch.cat([c2, c3, c4], dim=1)
        
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        # mod_selector_gamma_inj=mod_selector_gamma3
        # mod_selector_gamma_ext=torch.clone(mod_selector_gamma)
        # mod_selector_gamma_ext=torch.clone(mod_selector_gamma)
        if self.num_mod>1:
            # c2_4_other_modalities={}
            # c1_other_modalities={}
            # c2_4_other_modalities_cumulative=torch.zeros_like(c)
            # c1_other_modalities_cumulative=torch.zeros_like(c1)
            # for i in range(1,len(self.modalities_name)):
                # SPM forward
                # if i==2:
                #     prev_c2_4_other_modalities=torch.clone(c_other_modalities_t).unsqueeze(0)
                #     prev_c1_other_modalities=torch.clone(c1_other_modalities_t).unsqueeze(0)
                # c1_other_modalities_t, c2_other_modalities_t, c3_other_modalities_t, c4_other_modalities_t = self.spm_other_modalities[f"spm_{self.modalities_name[i]}"](x_other_modalities[:,modalities_ch_cumsum[i-1]:modalities_ch_cumsum[i]])
                # c1_other_modalities_t, c2_other_modalities_t, c3_other_modalities_t, c4_other_modalities_t =  mod_selector_gamma1[:,:,i].unsqueeze(-1)*c1_other_modalities_t, mod_selector_gamma2[:,:,i].unsqueeze(-1)*c2_other_modalities_t, mod_selector_gamma3[:,:,i].unsqueeze(-1)*c3_other_modalities_t, mod_selector_gamma4[:,:,i].unsqueeze(-1)*c4_other_modalities_t
                # c2_other_modalities_t, c3_other_modalities_t, c4_other_modalities_t = self._add_level_embed(c2_other_modalities_t, c3_other_modalities_t, c4_other_modalities_t)
                # c_other_modalities_t = torch.cat([c2_other_modalities_t, c3_other_modalities_t, c4_other_modalities_t], dim=1)
                # # if i>1:
                # #     c2_4_other_modalities=torch.cat([prev_c2_4_other_modalities,c_other_modalities_t.unsqueeze(0)],dim=0)
                # #     c1_other_modalities=torch.cat([prev_c1_other_modalities,c1_other_modalities_t.unsqueeze(0)],dim=0)
                # #     prev_c2_4_other_modalities=torch.clone(c2_4_other_modalities)
                # #     prev_c1_other_modalities=torch.clone(c1_other_modalities)
                # # c2_4_other_modalities.update({f"c2_4_{self.modalities_name[i]}":c_other_modalities_t})
                # # c1_other_modalities.update({f"c1_{self.modalities_name[i]}":c1_other_modalities_t}) 
                # c2_4_other_modalities_cumulative=c2_4_other_modalities_cumulative+self.multimodal_drop_path(c_other_modalities_t)
                # c1_other_modalities_cumulative=c1_other_modalities_cumulative+self.multimodal_drop_path(c1_other_modalities_t)
            c1_other_modalities, c2_other_modalities, c3_other_modalities, c4_other_modalities = self.spm_other_modalities(x_other_modalities)
            # c1_other_modalities, c2_other_modalities, c3_other_modalities, c4_other_modalities =  mod_selector_gamma1[:,:,1].unsqueeze(-1)*c1_other_modalities, mod_selector_gamma2[:,:,1].unsqueeze(-1)*c2_other_modalities, mod_selector_gamma3[:,:,1].unsqueeze(-1)*c3_other_modalities, mod_selector_gamma4[:,:,1].unsqueeze(-1)*c4_other_modalities
            c2_other_modalities, c3_other_modalities, c4_other_modalities = self._add_level_embed(c2_other_modalities, c3_other_modalities, c4_other_modalities)
            c2_4_other_modalities = torch.cat([c2_other_modalities, c3_other_modalities, c4_other_modalities], dim=1)
            # c=c+c2_4_other_modalities_cumulative
            # c1=c1+c1_other_modalities_cumulative
            c=c+c2_4_other_modalities
            c1=c1+c1_other_modalities
            
        
        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        # x=x.flatten(2)
        # x=x.flatten(1,2)
        bs, n, dim = x.shape
        # if self.pos_embed is not None:
        #     pos_embed = self._get_pos_embed(self.pos_embed, H, W)
        #     x = x + pos_embed
        # pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        pos_embed=self._get_pos_embed1(self.pos_embed, H, W)
        #pos_embed=pos_embed.flatten(1,2)
        x = self.pos_drop(x + pos_embed)
        # x=x+pos_embed

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            # if self.num_mod>1:
            x,c=layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                    deform_inputs1, deform_inputs2, H, W)
            # elif ('rgb' in self.modalities_name) and (self.num_mod==1):
            #     x,c=layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
            #                 deform_inputs1, deform_inputs2, H, W)
            # c+=c2_4_other_modalities_cumulative
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # if ('rgb' in self.modalities_name) and (self.num_mod==1):
        #     c1=c1.transpose(1, 2).view(bs, dim, 4*H, 4*W).contiguous()
        #     c2 = c[:, 0:c2.size(1), :]
        #     c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        #     c4 = c[:, c2.size(1) + c3.size(1):, :]

        #     c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        #     c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        #     c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        #     c1 = self.up(c2) + c1
        # elif self.num_mod>=1:
        #     c2_4_other_modalities_cumulative=torch.zeros_like(c)
        #     c1_other_modalities_cumulative=torch.zeros_like(c1)
        #     for i in range(1,len(self.modalities_name)):
        #         # c2_4_other_modalities_cumulative+=self.multimodal_drop_path(self.gamma_1_other_modalities[f"gamma_1_{self.modalities_name[i]}"]*c2_4_other_modalities[f"c2_4_{self.modalities_name[i]}"])
        #         # c1_other_modalities_cumulative+=self.multimodal_drop_path(self.gamma_2_other_modalities[f"gamma_2_{self.modalities_name[i]}"]*c1_other_modalities[f"c1_{self.modalities_name[i]}"])
        #         # c2_4_other_modalities_cumulative+=self.multimodal_drop_path(self.gamma_1_other_modalities[f"gamma_1_{self.modalities_name[i]}"].type_as(c2_4_other_modalities[i-1])*c2_4_other_modalities[i-1])
        #         # c1_other_modalities_cumulative+=self.multimodal_drop_path(self.gamma_2_other_modalities[f"gamma_2_{self.modalities_name[i]}"].type_as(c1_other_modalities[i-1])*c1_other_modalities[i-1])
        #         c2_4_other_modalities_cumulative+=self.multimodal_drop_path(c2_4_other_modalities[i-1])
        #         c1_other_modalities_cumulative+=self.multimodal_drop_path(mod_selector_gamma1[:,:,i].unsqueeze(-1)*c1_other_modalities[i-1])
            # c=c+c2_4_other_modalities_cumulative
            # c1=c1+c1_other_modalities_cumulative
        c1=c1.transpose(1, 2).view(bs, dim, 4*H, 4*W).contiguous() #H and W are H/16 and W/16
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False).type_as(x1)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False).type_as(x2)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False).type_as(x4)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        # if self.num_mod>1:
        #     mod_selector_gamma1=mod_selector_gamma1.transpose(1, 2).view(bs, self.num_mod, 4*H, 4*W).contiguous() #H and W are H/16 and W/16
        #     # mod_selector_gamma2 = mod_selector_gamma[:, 0:mod_selector_gamma2.size(1), :]
        #     # mod_selector_gamma3 = mod_selector_gamma[:, mod_selector_gamma2.size(1):mod_selector_gamma2.size(1) + mod_selector_gamma3.size(1), :]
        #     # mod_selector_gamma4 = mod_selector_gamma[:, mod_selector_gamma2.size(1) + mod_selector_gamma3.size(1):, :]
        #     mod_selector_gamma2 = mod_selector_gamma2.transpose(1, 2).view(bs, self.num_mod, H * 2, W * 2).contiguous()
        #     mod_selector_gamma3 = mod_selector_gamma3.transpose(1, 2).view(bs, self.num_mod, H, W).contiguous()
        #     # mod_selector_gamma_inj = mod_selector_gamma_inj.transpose(1, 2).view(bs, self.num_mod, H, W).contiguous()
        #     mod_selector_gamma4 = mod_selector_gamma4.transpose(1, 2).view(bs, self.num_mod, H // 2, W // 2).contiguous()
        #     return [f1, f2, f3, f4], mod_selector_gamma1, mod_selector_gamma2, mod_selector_gamma3,mod_selector_gamma4
        # else:
        return [f1, f2, f3, f4],None
    # ,mod_selector_gamma_inj
