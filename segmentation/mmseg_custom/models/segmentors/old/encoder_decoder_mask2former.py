# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
# from torch.profiler import profile, record_function, ProfilerActivity
from mmcv.runner import force_fp32, auto_fp16




@SEGMENTORS.register_module()
class EncoderDecoderMask2Former(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoderMask2Former, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        decode_head.update(train_cfg=train_cfg)
        decode_head.update(test_cfg=test_cfg)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        # x = self.backbone(img)
        x,*_=self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def extract_feat_test(self, img):
        """Extract features from images."""
        # x = self.backbone(img)
        # x,mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj=self.backbone(img)
        x=self.backbone(img)
        if len(x)>=2:
            if self.with_neck:
                x[0] = self.neck(x[0])
            # return x
        else:
            if self.with_neck:
                x = self.neck(x)
            # return x,mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # with profile(activities=[
                # ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            # with record_function("model_inference"):
        x = self.extract_feat(img)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    def encode_decode_test(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x,mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj = self.extract_feat_test(img)
        x= self.extract_feat_test(img)
        if len(x)>=2:
            out = self._decode_head_forward_test(x[0], img_metas)
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            return out,x[1:]
        else:
            out = self._decode_head_forward_test(x, img_metas)
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # return out, mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj
            return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg,
                                   **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg, **kwargs)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit
    
    # @force_fp32(apply_to=('img', ))
    # @auto_fp16(apply_to=('img', ))
    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # with profile(activities=[
                # ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
            # with record_function("model_inference"):
        x = self.extract_feat(img)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      **kwargs)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds
    def slide_inference_mod_sel(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        num_mod=len([key for key in img_meta[0].keys() if 'norm_cfg' in key])
        mod_sels = img.new_zeros((batch_size, num_mod, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit_t = self.encode_decode_test(crop_img, img_meta)
                crop_seg_logit=crop_seg_logit_t[0]
                if crop_seg_logit_t[1] != (None,):
                    mod_selector=crop_seg_logit_t[1][0]
                    mod_selector=resize(
                        mod_selector,
                        size=(w_crop,h_crop),
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False)
                    mod_sels += F.pad(mod_selector,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2))) 
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if crop_seg_logit_t[1] != (None,):    
            mod_sels = mod_sels / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            if crop_seg_logit_t[1] != (None,):    
                mod_sels = mod_sels / count_mat
                mod_sels = resize(
                    mod_sels,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            else:
                mod_sels = (None,)
        return preds, mod_sels

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit
    
    def whole_inference_dim(self, img, img_meta, rescale, dim):
        """Inference with full image."""

        # seg_logit, mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj = self.encode_decode_test(img, img_meta)
        seg_logit= self.encode_decode_test(img, img_meta)
        if len(seg_logit)>1:
            if rescale:
                # support dynamic shape for onnx
                # if torch.onnx.is_in_onnx_export():
                #     size = img.shape[2:]
                # else:
                #     size = img_meta[0]['ori_shape'][:2]
                seg_logit_t = resize(
                    seg_logit[0],
                    size=dim,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                return seg_logit_t,seg_logit[1]
        else:
            if rescale:
                # support dynamic shape for onnx
                # if torch.onnx.is_in_onnx_export():
                #     size = img.shape[2:]
                # else:
                #     size = img_meta[0]['ori_shape'][:2]
                seg_logit = resize(
                    seg_logit,
                    size=dim,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)

                return seg_logit
    
    def whole_inference_dim_cut(self, img, img_meta, rescale, dim,cut_dim):
        """Inference with full image."""

        # seg_logit, mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj = self.encode_decode_test(img, img_meta)
        seg_logit= self.encode_decode_test(img, img_meta)
        if len(seg_logit)>1:
            if rescale:
                # support dynamic shape for onnx
                # if torch.onnx.is_in_onnx_export():
                #     size = img.shape[2:]
                # else:
                #     size = img_meta[0]['ori_shape'][:2]
                seg_logit_t = resize(
                    seg_logit[0],
                    size=dim,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                return seg_logit_t,seg_logit[1]
            else:
                seg_logit_t = seg_logit[0]
                return seg_logit_t[:,:,:cut_dim[1],:cut_dim[0]],seg_logit[1]
        else:
            if rescale:
                # support dynamic shape for onnx
                # if torch.onnx.is_in_onnx_export():
                #     size = img.shape[2:]
                # else:
                #     size = img_meta[0]['ori_shape'][:2]
                seg_logit = resize(
                    seg_logit,
                    size=dim,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)

                return seg_logit
            else:
                return seg_logit[:,:,:cut_dim[1],:cut_dim[0]]
    

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole', 'whole_dim' ,'slide_mod_sel', 'whole_dim_cut']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        elif self.test_cfg.mode == 'slide_mod_sel':
            seg_logit = self.slide_inference_mod_sel(img, img_meta, rescale)
        elif self.test_cfg.mode == 'whole':
            seg_logit = self.whole_inference(img, img_meta, rescale)
        elif self.test_cfg.mode == 'whole_dim_cut':
            seg_logit = self.whole_inference_dim_cut(img, img_meta, rescale, self.test_cfg.dim,self.test_cfg.cut_dim)
        else:
            seg_logit = self.whole_inference_dim(img, img_meta, rescale, self.test_cfg.dim)
        #if len(seg_logit)==4:    
        
        if isinstance(seg_logit, tuple):
            output = F.softmax(seg_logit[0], dim=1)
            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))
            return output, seg_logit[1]
        else:
            output = F.softmax(seg_logit, dim=1)
            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))
            return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        # seg_logit, mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj = self.inference(img, img_meta, rescale)
        seg_logit= self.inference(img, img_meta, rescale)
        
        if isinstance(seg_logit, tuple):
            seg_pred = seg_logit[0].argmax(dim=1)
            if torch.onnx.is_in_onnx_export():
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
                return seg_pred
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            if seg_logit[1] != (None,):
                for i in range(0, len(seg_logit[1])):
                    cur_mod_gamma = seg_logit[1][i]
                    cur_mod_gamma = cur_mod_gamma.cpu().numpy()
                    seg_pred.append(cur_mod_gamma)
            # mod_gamma_spatial_1 = mod_gamma_spatial_1.cpu().numpy()
            # mod_gamma_spatial_2 = mod_gamma_spatial_2.cpu().numpy()
            # mod_gamma_spatial_3 = mod_gamma_spatial_3.cpu().numpy()
            # mod_gamma_spatial_4 = mod_gamma_spatial_4.cpu().numpy()
            # mod_gamma_spatial_inj = mod_gamma_spatial_inj.cpu().numpy()
            # unravel batch dim
            # seg_pred = list(seg_pred)
            # return seg_pred,  mod_gamma_spatial_1,mod_gamma_spatial_2,mod_gamma_spatial_3,mod_gamma_spatial_4,mod_gamma_spatial_inj
            return seg_pred
        else:
            seg_pred = seg_logit.argmax(dim=1)
            if torch.onnx.is_in_onnx_export():
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
                return seg_pred
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            return seg_pred
    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        if isinstance(seg_logit, tuple):
            # seg_logit = seg_logit[0]
            seg_logit=list(seg_logit)
            for i in range(1, len(imgs)):
                cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
                seg_logit[0] += cur_seg_logit[0]
                seg_logit[1] += cur_seg_logit[1]
            seg_logit[0] /= len(imgs)
            seg_logit[1] /= len(imgs)
            seg_pred = seg_logit[0].argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            if seg_logit[1] != (None,):
                for i in range(0, len(seg_logit[1])):
                    cur_mod_gamma = seg_logit[1][i]
                    cur_mod_gamma = cur_mod_gamma.cpu().numpy()
                    seg_pred.append(cur_mod_gamma)
            # seg_pred.append(seg_logit[1].cpu().numpy())
        else:
            for i in range(1, len(imgs)):
                cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
                seg_logit += cur_seg_logit
            seg_logit /= len(imgs)
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
        return seg_pred
