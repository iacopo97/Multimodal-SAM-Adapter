# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from functools import partial
import os
# if os.getcwd()=='/media/data2/icurti/projects/sina/ViT-Adapter/segmentation':
if '/segmentation' in os.getcwd():
    from mmcv_custom import load_checkpoint
else:
    from ViTAdapter.segmentation.mmcv_custom import load_checkpoint
# from ViTAdapter.segmentation.mmcv_custom import load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
# from .common import LayerNorm2d, MLPBlock
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch.utils.checkpoint as cp
import numpy as np
import matplotlib.pyplot as plt
import cv2
global DEBUG
DEBUG=True
if DEBUG:
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
    def get_attention_map(img, attn_weights, attn_weights_other_modalities=None,get_mask=False):
        # x = transform(img)
        # x.size()

        # logits, att_mat = model(x.unsqueeze(0))

        # att_mat = torch.stack(att_mat).squeeze(1)

        # Average the attention weights across all heads.
        att_mat = attn_weights
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        if get_mask:
            result = cv2.resize(mask / mask.max(), img.size)
        else:        
            mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
            result = (mask * img).astype("uint8")
        
        return result
    
    
    def plot_attention_map(original_img, att_map):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map Last Layer')
        _ = ax1.imshow(original_img)
        _ = ax2.imshow(att_map)


def fix_params(input_module: torch.nn.Module):
    if isinstance(input_module,nn.Parameter):
        input_module.requires_grad = False
    else:
        for name, param in input_module.named_parameters():
            param.requires_grad = False


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# # From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# # Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
# class LayerNorm2d(nn.Module):
#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
@BACKBONES.register_module(force=True)
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024, #1024 768
        depth: int = 24, #24 12
        num_heads: int = 16, #12
        mlp_ratio: float = 4.0,
        # out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),#nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True, #True
        use_rel_pos: bool = True, #False
        rel_pos_zero_init: bool = True,
        window_size: int = 14, #0
        global_attn_indexes: Tuple[int, ...] = [5, 11, 17, 23],#(),#=[7, 15, 23, 31]
        ############MY_MOD###########
        pretrained=None,
        with_cp=False,
        pretrained_size=1024,
        fix=False
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            # self.pos_embed = nn.Parameter(
            #     torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            # )
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pretrained_size // patch_size, pretrained_size // patch_size, embed_dim)
            )
            # self.num_tokens=1
            # num_patches=img_size//patch_size
            # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        # else:
        #     self.pos_embed= None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            # if (i+1)%4 ==0:
            #     with_cp=False
            # else:
            #     with_cp=True
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                # input_size=(img_size // patch_size, img_size // patch_size), with_cp=with_cp,
                input_size=(pretrained_size // patch_size, pretrained_size // patch_size), with_cp=with_cp)#bool((i)%4))bool((i)%2) bool((i)%2)
            self.blocks.append(block)

        # self.neck = nn.Sequential(
        #     nn.Conv2d(
        #         embed_dim,
        #         out_chans,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        #     nn.Conv2d(
        #         out_chans,
        #         out_chans,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        # )
        #############My MOD###############
        self.apply(self._init_weights)
        self.init_weights(pretrained)
        if fix:
            fix_params(self.patch_embed)
            fix_params(self.pos_embed)
            fix_params(self.blocks)

        # self.fix_init_weight()
    
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # pretrained = 'pretrained/beit_large_patch16_512_pt22k_ft22kto1k.pth'
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        with_cp=True
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        ######MY MOD###########
        self.with_cp=with_cp
        #######################
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor, H : int, W:int) -> torch.Tensor:
        if len(x.shape)==3:
            x=x.unflatten(1,(H,W))
        # shortcut = x
        # x = self.norm1(x)
        # # Window partition
        # if self.window_size > 0:
        #     H, W = x.shape[1], x.shape[2]
        #     x, pad_hw = window_partition(x, self.window_size)

        # x = self.attn(x)
        # # Reverse window partition
        # if self.window_size > 0:
        #     x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # x = shortcut + x
        # x = x + self.mlp(self.norm2(x))
        def _inner_forward(x):
            shortcut = x
            x = self.norm1(x)
            # Window partition
            if self.window_size > 0:
                # H, W = H,W
                # if len(x.shape)==3:
                #     x=x.unflatten(1,(H,W))
                x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x)
            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))

            x = shortcut + x
            x = x + self.mlp(self.norm2(x))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        x=x.flatten(1,2)
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ##MY MOD###################
        # B,N,C=x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # # q, k, v with shape (B * nHead, H * W, C)
        # q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        # attn = (q * self.scale) @ k.transpose(-2, -1)

        # if self.use_rel_pos:
        #     attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        # attn = attn.softmax(dim=-1)
        # x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        # x = self.proj(x)
        # ###########################


        B, H, W, _ = x.shape
        # if len(x.shape())==3:
        #     B, H, _ = x.shape
        # B,H,W,_ =x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        img_size: int=224,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        #############MY MOD##################
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(kernel_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # patch_size=kernel_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        ##MY mOD######
        x=x.flatten(1,2)
        ###############
        return x,Hp,Wp