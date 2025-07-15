import mmcv
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES
import cv2
from numpy import random
from typing import List, Tuple, Union
import numbers


@PIPELINES.register_module()
class RandomCropGen(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            cond=True
            while cond==True:
                for count in range(10):
                    if count==0:
                        crop_bbox = crop_bbox
                    else:
                        crop_bbox = crop_bbox_new
                    seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                    labels, cnt = np.unique(seg_temp, return_counts=True)
                    cnt = cnt[labels != self.ignore_index]
                    if len(cnt) > 1 and np.max(cnt) / np.sum(
                            cnt) < self.cat_max_ratio:
                        cond=False
                        break
                    crop_bbox_new = self.get_crop_bbox(img)
                # if len(labels)==2 and( labels[0]==self.ignore_index or labels[1]==self.ignore_index):
                #     print('here2')
                #     print(cnt)
                # if len(labels)==1 and labels[0]==self.ignore_index:
                #     print('here1')
                if (len(labels)>=2):
                    cond=False
                # if (len(labels)==2) and (self.ignore_index in labels):
                #     if(cnt[0]>2):
                #         print(cnt)
                #         cond=False
                # if self.ignore_index not in labels:
                #     cond=False

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class SETR_Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 crop_size=None,
                 setr_multi_scale=False):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.crop_size = crop_size
        self.setr_multi_scale = setr_multi_scale

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """
        
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None
    
    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""

        if self.keep_ratio:
            if self.setr_multi_scale:
                if min(results['scale']) < self.crop_size[0]:
                    new_short = self.crop_size[0]
                else:
                    new_short = min(results['scale'])

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(results['img'],
                                               results['scale'],
                                               return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(results['img'],
                                                  results['scale'],
                                                  return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(results[key],
                                        results['scale'],
                                        interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(results[key],
                                       results['scale'],
                                       interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

@PIPELINES.register_module()
class SETR_Resize_multimodal(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 crop_size=None,
                 setr_multi_scale=False,
                 modalities_name=['RGB','HHA'],
                 modalities_ch=[3,3]):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.crop_size = crop_size
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch
        self.modalities_ch_cumsum=np.cumsum(np.array(self.modalities_ch),0)
        self.setr_multi_scale = setr_multi_scale

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """
        
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None
    
    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx
        
        
    
    def _resize_multimodal(self, results):
        """Resize multimodal images with ``results['scale']``."""
        mm_matrix=[]
        for index in range(0,len(self.modalities_name)):
            if self.modalities_name[index]=="rgb":
                prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=self.modalities_ch_cumsum[index-1]
            if self.keep_ratio:
                if self.setr_multi_scale:
                    if min(results['scale']) < self.crop_size[0]:
                        new_short = self.crop_size[0]
                    else:
                        new_short = min(results['scale'])

                    h, w = results['img'].shape[:2]
                    if h > w:
                        new_h, new_w = new_short * h / w, new_short
                    else:
                        new_h, new_w = new_short, new_short * w / h
                    results['scale'] = (new_h, new_w)
                # img_t=results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch[i]]
                img_t, scale_factor = mmcv.imrescale(
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img_t.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img_t, w_scale, h_scale = mmcv.imresize(
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
            if len(img_t.shape)==2:
                img_t=img_t[...,np.newaxis]
            mm_matrix.append(img_t)
        mm_matrix = np.concatenate(mm_matrix, axis=2)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = mm_matrix
        results['img_shape'] = mm_matrix.shape
        results['pad_shape'] = mm_matrix.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio
    
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""

        if self.keep_ratio:
            if self.setr_multi_scale:
                if min(results['scale']) < self.crop_size[0]:
                    new_short = self.crop_size[0]
                else:
                    new_short = min(results['scale'])

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(results['img'],
                                               results['scale'],
                                               return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(results['img'],
                                                  results['scale'],
                                                  return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(results[key],
                                        results['scale'],
                                        interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(results[key],
                                       results['scale'],
                                       interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        # self._resize_img(results)
        self._resize_multimodal(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio}),'
                     f'modalities_name={self.modalities_name},')
        return repr_str


@PIPELINES.register_module()
class PadShortSide(object):
    """Pad the image & mask.

    Pad to the minimum size that is equal or larger than a number.
    Added keys are "pad_shape", "pad_fixed_size",

    Args:
        size (int, optional): Fixed padding size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """
    def __init__(self, size=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        h, w = results['img'].shape[:2]
        new_h = max(h, self.size)
        new_w = max(w, self.size)
        padded_img = mmcv.impad(results['img'],
                                shape=(new_h, new_w),
                                pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        # results['unpad_shape'] = (h, w)

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key],
                                      shape=results['pad_shape'][:2],
                                      pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        h, w = results['img'].shape[:2]
        if h >= self.size and w >= self.size:  # 短边比窗口大，跳过
            pass
        else:
            self._pad_img(results)
            self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, pad_val={self.pad_val})'
        return repr_str




@PIPELINES.register_module()
class RandomResizedCrop:
    # def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
    #     """Resize the input image to the given size.
    #     """
    #     self.size = size
    #     self.scale = scale
    #     self.seg_fill = seg_fill

    # def __call__(self, sample: list) -> list:
    #     # img, mask = sample['img'], sample['mask']
    #     H, W = sample['img'].shape[1:]
    #     tH, tW = self.size

    #     # get the scale
    #     ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
    #     # ratio = random.uniform(min(self.scale), max(self.scale))
    #     scale = int(tH*ratio), int(tW*4*ratio)
    #     # scale the image 
    #     scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
    #     nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
    #     # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
    #     for k, v in sample.items():
    #         if k == 'mask':                
    #             sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
    #         else:
    #             sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

    #     # random crop
    #     margin_h = max(sample['img'].shape[1] - tH, 0)
    #     margin_w = max(sample['img'].shape[2] - tW, 0)
    #     y1 = random.randint(0, margin_h+1)
    #     x1 = random.randint(0, margin_w+1)
    #     y2 = y1 + tH
    #     x2 = x1 + tW
    #     for k, v in sample.items():
    #         sample[k] = v[:, y1:y2, x1:x2]

    #     # pad the image
    #     if sample['img'].shape[1:] != self.size:
    #         padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
    #         for k, v in sample.items():
    #             if k == 'mask':                
    #                 sample[k] = TF.pad(v, padding, fill=self.seg_fill)
    #             else:
    #                 sample[k] = TF.pad(v, padding, fill=0)

    #     return sample
    
    def __init__(self, crop_size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0, modalities_name=None,modalities_ch=None,cat_max_ratio=1.,ignore_index=255) -> None:
        """Resize the input image to the given size.
        """
        self.crop_size = crop_size
        self.scale = scale
        self.seg_fill = seg_fill
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch
        self.cat_max_ratio=cat_max_ratio
        self.ignore_index=ignore_index

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img
    # def _resize_img(self, img, size):
    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        # center= True if np.random.rand() < self.prob else False
        # if center:
        #     cv,cu=tuple(ti/2 for ti in img.shape[:2])
        #     assert (cu+(self.crop_size[1]/2))<img.shape[1] and (cv+(self.crop_size[0]/2))<img.shape[0]
        #     crop_y1, crop_y2 = int(cv) - int(self.crop_size[0]/2), int(cv) + int(self.crop_size[0]/2)
        #     crop_x1, crop_x2 = int(cu) - int(self.crop_size[1]/2), int(cu) + int(self.crop_size[1]/2)
            
        # else:
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2
    
    def get_random_size(self,scale,crop_size,H,W):
        # get the scale
        tH, tW = crop_size
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        return nH,nW
    
    def resize_multimodal(self,modalities_ch_cumsum,img,nH,nW):
        resized_matrix=[]
        for i in range(0,len(self.modalities_name)):
            if self.modalities_name[i]=="rgb":
                prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=modalities_ch_cumsum[i-1]
            resized_matrix_t=cv2.resize(img[:,:,prev_modalities_ch_cumsum:modalities_ch_cumsum[i]], (nW, nH), interpolation=cv2.INTER_LINEAR)
            if len(resized_matrix_t.shape)==2:
                resized_matrix_t=resized_matrix_t[...,np.newaxis]
            resized_matrix.append(resized_matrix_t)
        return np.concatenate(resized_matrix, axis=2)
    
    def resize_seg(self,seg,nH,nW):
        return cv2.resize(seg, (nW, nH), interpolation=cv2.INTER_NEAREST)
    
    def __call__(self, results):
        # img, mask = sample['img'], sample['mask']
        # H, W = sample['img'].shape[1:]
        H,W = results['img'].shape[:2]
        # tH, tW = self.crop_size
        crop_size=self.crop_size
        # get the scale
        # ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # # ratio = random.uniform(min(self.scale), max(self.scale))
        # scale = int(tH*ratio), int(tW*4*ratio)
        # # scale the image 
        # scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        # for k, v in results.items():
            # if k == 'mask':                
                # results[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            # else:
        nH,nW=self.get_random_size(self.scale,crop_size,H,W)
        # for k, v in results.items():
        modalities_ch_cumsum=np.cumsum(np.array(self.modalities_ch),0)
        # resized_matrix=[]
        # for i in range(0,len(self.modalities_name)):
        #     if self.modalities_name[i]=="rgb":
        #         prev_modalities_ch_cumsum=0
        #     else:
        #         prev_modalities_ch_cumsum=modalities_ch_cumsum[i-1]
        #     resized_matrix_t=cv2.resize(results['img'][:,:,prev_modalities_ch_cumsum:modalities_ch_cumsum[i]], (nW, nH), interpolation=cv2.INTER_LINEAR)
        #     if len(resized_matrix_t.shape)==2:
        #         resized_matrix_t=resized_matrix_t[...,np.newaxis]
        #     resized_matrix.append(resized_matrix_t)
        # results['img']=np.concatenate(resized_matrix, axis=2)
        results['img']=self.resize_multimodal(modalities_ch_cumsum,results['img'],nH,nW)
        for k in results.get('seg_fields',[]):
            # results[k] = cv2.resize(results[k], (nW, nH), interpolation=cv2.INTER_NEAREST)
            results[k]=self.resize_seg(results[k],nH,nW)
        # random crop
        # margin_h = max(results['img'].shape[0] - tH, 0)
        # margin_w = max(results['img'].shape[1] - tW, 0)
        # y1 = random.randint(0, margin_h+1)
        # x1 = random.randint(0, margin_w+1)
        # y2 = y1 + tH
        # x2 = x1 + tW
        img=results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)
        
        # results['img'] = results['img'][y1:y2, x1:x2,:]
        results['img']=self.crop(img,crop_bbox)
        for k in results.get('seg_fields',[]):
            # results[k] = cv2.resize(results[k], (nW, nH), interpolation=cv2.INTER_NEAREST)
            # results[k] = results[k][y1:y2, x1:x2]
            results[k]=self.crop(results[k],crop_bbox)

        # # pad the image
        # if results['img'].shape[:2] != self.crop_size:
        #     # padding = [0, 0, tW - results['img'].shape[2], tH - results['img'].shape[1]]
        #     # for k, v in results.items():
        #         # if k == 'mask':                
        #     # results['img'] = TF.pad(v, padding, fill=0)
        #     results['img']=impad(results['img'],shape=self.crop_size, pad_val=0)
        #         # else:
        #     # results[k] = TF.pad(v, padding, fill=0)
        #     for k in results.get('seg_fields'):
        #         # results[k] = TF.pad(results[k], padding, fill=self.seg_fill)
        #         results[k] = impad(results[k], shape=self.crop_size, pad_val=self.seg_fill)
        results['img_shape'] = results['img'].shape
        return results
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, scale={self.scale}, ' \
                    f'seg_fill={self.seg_fill},'\
                    f'cat_max_ratio={self.cat_max_ratio},'\
                    f'ignore_index={self.ignore_index})'
        return repr_str
    
@PIPELINES.register_module()
class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5, modalities_name=None,modalities_ch=None) -> None:
        self.kernel_size = kernel_size
        self.p = p
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch
        self.modalities_ch_cumsum=np.cumsum(np.array(modalities_ch),0)
        if 'rgb' in modalities_name:
            self.in_ch_im=modalities_ch[modalities_name.index('rgb')]

    # def __call__(self, sample: list) -> list:
    #     if random.random() < self.p:
    #         sample['img'] = TF.gaussian_blur(sample['img'], self.kernel_size)
    #         # img = TF.gaussian_blur(img, self.kernel_size)
    #     return sample
    def __call__(self, results):
        if self.in_ch_im:
            if random.random() < self.p: #(0,0) kernel size for mcubes
                if self.kernel_size==(0,0) or self.kernel_size==0:
                    radius=random.random()
                else:
                    radius=0
                # sample['img'] = TF.gaussian_blur(sample['img'], self.kernel_size)
                # results['img'][:,:,:self.in_ch_im] = TF.gaussian_blur(results['img'][:,:,:self.in_ch_im], self.kernel_size)
                results['img'][:,:,:self.in_ch_im] = cv2.GaussianBlur(results['img'][:,:,:self.in_ch_im], self.kernel_size,radius)
                if 'NIR' in self.modalities_name:
                    results['img'][:,:,self.modalities_ch_cumsum[self.modalities_name.index('NIR')-1]:self.modalities_ch_cumsum[self.modalities_name.index('NIR')]] = cv2.GaussianBlur(results['img'][:,:,self.modalities_ch_cumsum[self.modalities_name.index('NIR')-1]:self.modalities_ch_cumsum[self.modalities_name.index('NIR')]], self.kernel_size,radius)
                # img = TF.gaussian_blur(img, self.kernel_size)
        return results
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(kernel_size={self.kernel_size}, p={self.p},)'
        return repr_str
    

@PIPELINES.register_module()
class Resize_multimodal(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 seg_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 modalities_name=None,
                 modalities_ch=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
                self.seg_scale = seg_scale
            else:
                self.img_scale = [img_scale]
                self.seg_scale = [seg_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch
        self.modalities_ch_cumsum=np.cumsum(np.array(self.modalities_ch),0)

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    # def _resize_img(self, results):
    #     """Resize images with ``results['scale']``."""
    #     if self.keep_ratio:
    #         img, scale_factor = mmcv.imrescale(
    #             results['img'], results['scale'], return_scale=True)
    #         # the w_scale and h_scale has minor difference
    #         # a real fix should be done in the mmcv.imrescale in the future
    #         new_h, new_w = img.shape[:2]
    #         h, w = results['img'].shape[:2]
    #         w_scale = new_w / w
    #         h_scale = new_h / h
    #     else:
    #         img, w_scale, h_scale = mmcv.imresize(
    #             results['img'], results['scale'], return_scale=True)
    #     scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
    #                             dtype=np.float32)
    #     results['img'] = img
    #     results['img_shape'] = img.shape
    #     results['pad_shape'] = img.shape  # in case that there is no padding
    #     results['scale_factor'] = scale_factor
    #     results['keep_ratio'] = self.keep_ratio

    def _resize_multimodal(self, results):
        """Resize multimodal images with ``results['scale']``."""
        mm_matrix=[]
        for index in range(0,len(self.modalities_name)):
            if self.modalities_name[index]=="rgb":
                prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=self.modalities_ch_cumsum[index-1]
            if self.keep_ratio:
                # img_t=results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch[i]]
                img_t, scale_factor = mmcv.imrescale(
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img_t.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img_t, w_scale, h_scale = mmcv.imresize(
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
            if len(img_t.shape)==2:
                img_t=img_t[...,np.newaxis]
            mm_matrix.append(img_t)
        mm_matrix = np.concatenate(mm_matrix, axis=2)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = mm_matrix
        results['img_shape'] = mm_matrix.shape
        results['pad_shape'] = mm_matrix.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        if self.seg_scale is None or self.seg_scale==[None]:
            for key in results.get('seg_fields', []):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key], results['scale'], interpolation='nearest')
                else:
                    gt_seg = mmcv.imresize(
                        results[key], results['scale'], interpolation='nearest')
                results[key] = gt_seg
        else:
            for key in results.get('seg_fields', []):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key], self.seg_scale[0], interpolation='nearest')
                else:
                    gt_seg = mmcv.imresize(
                        results[key], self.seg_scale[0], interpolation='nearest')
                results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        # self._resize_img(results)
        try:
            self._resize_multimodal(results)
        except:
            pass
        try:
            self._resize_seg(results)
        except:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'modalities_name={self.modalities_name}, '
                     f'modalities_ch={self.modalities_ch}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str
    
# @PIPELINES.register_module()
# class RandomScaleCrop_multimodal(object):
#     """Resize images & seg.

#     This transform resizes the input image to some scale. If the input dict
#     contains the key "scale", then the scale in the input dict is used,
#     otherwise the specified scale in the init method is used.

#     ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
#     (multi-scale). There are 4 multiscale modes:

#     - ``ratio_range is not None``:
#     1. When img_scale is None, img_scale is the shape of image in results
#         (img_scale = results['img'].shape[:2]) and the image is resized based
#         on the original size. (mode 1)
#     2. When img_scale is a tuple (single-scale), randomly sample a ratio from
#         the ratio range and multiply it with the image scale. (mode 2)

#     - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
#     scale from the a range. (mode 3)

#     - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
#     scale from multiple scales. (mode 4)

#     Args:
#         img_scale (tuple or list[tuple]): Images scales for resizing.
#             Default:None.
#         multiscale_mode (str): Either "range" or "value".
#             Default: 'range'
#         ratio_range (tuple[float]): (min_ratio, max_ratio).
#             Default: None
#         keep_ratio (bool): Whether to keep the aspect ratio when resizing the
#             image. Default: True
#     """

#     def __init__(self,
#                  img_scale=None,
#                  seg_scale=None,
#                  multiscale_mode='range',
#                  ratio_range=None,
#                  keep_ratio=True,
#                  modalities_name=None,
#                  modalities_ch=None,
#                  base_size=None,
#                  crop_size=None):
#         # if img_scale is None:
#         #     self.img_scale = None
#         # else:
#         #     if isinstance(img_scale, list):
#         #         self.img_scale = img_scale
#         #         self.seg_scale = seg_scale
#         #     else:
#         #         self.img_scale = [img_scale]
#         #         self.seg_scale = [seg_scale]
#         #     assert mmcv.is_list_of(self.img_scale, tuple)

#         # if ratio_range is not None:
#         #     # mode 1: given img_scale=None and a range of image ratio
#         #     # mode 2: given a scale and a range of image ratio
#         #     assert self.img_scale is None or len(self.img_scale) == 1
#         # else:
#         #     # mode 3 and 4: given multiple scales or a range of scales
#         #     assert multiscale_mode in ['value', 'range']

#         self.multiscale_mode = multiscale_mode
#         self.ratio_range = ratio_range
#         self.keep_ratio = keep_ratio
#         self.modalities_name=modalities_name
#         self.modalities_ch=modalities_ch
#         self.base_size=base_size
#         self.crop_size=crop_size
#         self.modalities_ch_cumsum=np.cumsum(np.array(self.modalities_ch),0)

#     # @staticmethod
#     # def random_select(img_scales):
#     #     """Randomly select an img_scale from given candidates.

#     #     Args:
#     #         img_scales (list[tuple]): Images scales for selection.

#     #     Returns:
#     #         (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
#     #             where ``img_scale`` is the selected image scale and
#     #             ``scale_idx`` is the selected index in the given candidates.
#     #     """

#     #     assert mmcv.is_list_of(img_scales, tuple)
#     #     scale_idx = np.random.randint(len(img_scales))
#     #     img_scale = img_scales[scale_idx]
#     #     return img_scale, scale_idx

#     # @staticmethod
#     # def random_sample(img_scales):
#     #     """Randomly sample an img_scale when ``multiscale_mode=='range'``.

#     #     Args:
#     #         img_scales (list[tuple]): Images scale range for sampling.
#     #             There must be two tuples in img_scales, which specify the lower
#     #             and upper bound of image scales.

#     #     Returns:
#     #         (tuple, None): Returns a tuple ``(img_scale, None)``, where
#     #             ``img_scale`` is sampled scale and None is just a placeholder
#     #             to be consistent with :func:`random_select`.
#     #     """

#     #     assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
#     #     img_scale_long = [max(s) for s in img_scales]
#     #     img_scale_short = [min(s) for s in img_scales]
#     #     long_edge = np.random.randint(
#     #         min(img_scale_long),
#     #         max(img_scale_long) + 1)
#     #     short_edge = np.random.randint(
#     #         min(img_scale_short),
#     #         max(img_scale_short) + 1)
#     #     img_scale = (long_edge, short_edge)
#     #     return img_scale, None

#     # @staticmethod
#     # def random_sample_ratio(img_scale, ratio_range):
#     #     """Randomly sample an img_scale when ``ratio_range`` is specified.

#     #     A ratio will be randomly sampled from the range specified by
#     #     ``ratio_range``. Then it would be multiplied with ``img_scale`` to
#     #     generate sampled scale.

#     #     Args:
#     #         img_scale (tuple): Images scale base to multiply with ratio.
#     #         ratio_range (tuple[float]): The minimum and maximum ratio to scale
#     #             the ``img_scale``.

#     #     Returns:
#     #         (tuple, None): Returns a tuple ``(scale, None)``, where
#     #             ``scale`` is sampled ratio multiplied with ``img_scale`` and
#     #             None is just a placeholder to be consistent with
#     #             :func:`random_select`.
#     #     """

#     #     assert isinstance(img_scale, tuple) and len(img_scale) == 2
#     #     min_ratio, max_ratio = ratio_range
#     #     assert min_ratio <= max_ratio
#     #     ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
#     #     scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
#     #     return scale, None

#     # def _random_scale(self, results):
#     #     """Randomly sample an img_scale according to ``ratio_range`` and
#     #     ``multiscale_mode``.

#     #     If ``ratio_range`` is specified, a ratio will be sampled and be
#     #     multiplied with ``img_scale``.
#     #     If multiple scales are specified by ``img_scale``, a scale will be
#     #     sampled according to ``multiscale_mode``.
#     #     Otherwise, single scale will be used.

#     #     Args:
#     #         results (dict): Result dict from :obj:`dataset`.

#     #     Returns:
#     #         dict: Two new keys 'scale` and 'scale_idx` are added into
#     #             ``results``, which would be used by subsequent pipelines.
#     #     """

#     #     if self.ratio_range is not None:
#     #         if self.img_scale is None:
#     #             h, w = results['img'].shape[:2]
#     #             scale, scale_idx = self.random_sample_ratio((w, h),
#     #                                                         self.ratio_range)
#     #         else:
#     #             scale, scale_idx = self.random_sample_ratio(
#     #                 self.img_scale[0], self.ratio_range)
#     #     elif len(self.img_scale) == 1:
#     #         scale, scale_idx = self.img_scale[0], 0
#     #     elif self.multiscale_mode == 'range':
#     #         scale, scale_idx = self.random_sample(self.img_scale)
#     #     elif self.multiscale_mode == 'value':
#     #         scale, scale_idx = self.random_select(self.img_scale)
#     #     else:
#     #         raise NotImplementedError

#     #     results['scale'] = scale
#     #     results['scale_idx'] = scale_idx

#     # # def _resize_img(self, results):
#     # #     """Resize images with ``results['scale']``."""
#     # #     if self.keep_ratio:
#     # #         img, scale_factor = mmcv.imrescale(
#     # #             results['img'], results['scale'], return_scale=True)
#     # #         # the w_scale and h_scale has minor difference
#     # #         # a real fix should be done in the mmcv.imrescale in the future
#     # #         new_h, new_w = img.shape[:2]
#     # #         h, w = results['img'].shape[:2]
#     # #         w_scale = new_w / w
#     # #         h_scale = new_h / h
#     # #     else:
#     # #         img, w_scale, h_scale = mmcv.imresize(
#     # #             results['img'], results['scale'], return_scale=True)
#     # #     scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
#     # #                             dtype=np.float32)
#     # #     results['img'] = img
#     # #     results['img_shape'] = img.shape
#     # #     results['pad_shape'] = img.shape  # in case that there is no padding
#     # #     results['scale_factor'] = scale_factor
#     # #     results['keep_ratio'] = self.keep_ratio

#     # def _resize_multimodal(self, results):
#     #     """Resize multimodal images with ``results['scale']``."""
#     #     mm_matrix=[]
#     #     for index in range(0,len(self.modalities_name)):
#     #         if self.modalities_name[index]=="rgb":
#     #             prev_modalities_ch_cumsum=0
#     #         else:
#     #             prev_modalities_ch_cumsum=self.modalities_ch_cumsum[index-1]
#     #         if self.keep_ratio:
#     #             # img_t=results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch[i]]
#     #             img_t, scale_factor = mmcv.imrescale(
#     #                 results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
#     #             # the w_scale and h_scale has minor difference
#     #             # a real fix should be done in the mmcv.imrescale in the future
#     #             new_h, new_w = img_t.shape[:2]
#     #             h, w = results['img'].shape[:2]
#     #             w_scale = new_w / w
#     #             h_scale = new_h / h
#     #         else:
#     #             img_t, w_scale, h_scale = mmcv.imresize(
#     #                 results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
#     #         if len(img_t.shape)==2:
#     #             img_t=img_t[...,np.newaxis]
#     #         mm_matrix.append(img_t)
#     #     mm_matrix = np.concatenate(mm_matrix, axis=2)
#     #     scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
#     #                             dtype=np.float32)
#     #     results['img'] = mm_matrix
#     #     results['img_shape'] = mm_matrix.shape
#     #     results['pad_shape'] = mm_matrix.shape  # in case that there is no padding
#     #     results['scale_factor'] = scale_factor
#     #     results['keep_ratio'] = self.keep_ratio

#     # def _resize_seg(self, results):
#     #     """Resize semantic segmentation map with ``results['scale']``."""
#     #     if self.seg_scale is None or self.seg_scale==[None]:
#     #         for key in results.get('seg_fields', []):
#     #             if self.keep_ratio:
#     #                 gt_seg = mmcv.imrescale(
#     #                     results[key], results['scale'], interpolation='nearest')
#     #             else:
#     #                 gt_seg = mmcv.imresize(
#     #                     results[key], results['scale'], interpolation='nearest')
#     #             results[key] = gt_seg
#     #     else:
#     #         for key in results.get('seg_fields', []):
#     #             if self.keep_ratio:
#     #                 gt_seg = mmcv.imrescale(
#     #                     results[key], self.seg_scale[0], interpolation='nearest')
#     #             else:
#     #                 gt_seg = mmcv.imresize(
#     #                     results[key], self.seg_scale[0], interpolation='nearest')
#     #             results[key] = gt_seg
    
#     def _resize_multimodal(self, results):
#     #     """Resize multimodal images with ``results['scale']``."""
    
#     def __call__(self, results):
#         """Call function to resize images, bounding boxes, masks, semantic
#         segmentation map.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
#                 'keep_ratio' keys are added into result dict.
#         """
#         short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
#         h, w = results['img'].shape[:2]
#         if h > w:
#             ow = short_size
#             oh = int(1.0 * h * ow / w)
#         else:
#             oh = short_size
#             ow = int(1.0 * w * oh / h)

#         # pad crop
#         if short_size < self.crop_size:
#             padh = self.crop_size - oh if oh < self.crop_size else 0
#             padw = self.crop_size - ow if ow < self.crop_size else 0
            
#         # random crop crop_size
#         # w, h = img.size
#         h, w = img.shape[:2]

#         # x1 = random.randint(0, w - self.crop_size)
#         # y1 = random.randint(0, h - self.crop_size)
#         x1 = random.randint(0, max(0, ow - self.crop_size))
#         y1 = random.randint(0, max(0, oh - self.crop_size))

#         # u_map = sample['u_map']
#         # v_map = sample['v_map']
#         # u_map    = cv2.resize(u_map,(ow,oh))
#         # v_map    = cv2.resize(v_map,(ow,oh))
#         self._resize_multimodal(results)
#         self._crop_multimodal(results)
        
#         self._resize_seg(results)
#         self._crop_seg(results)
#         aolp     = cv2.resize(aolp ,(ow,oh))
#         dolp     = cv2.resize(dolp ,(ow,oh))
#         # SS     = cv2.resize(SS ,(ow,oh))
#         img      = cv2.resize(img  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
#         # mask     = cv2.resize(mask ,(ow,oh), interpolation=cv2.INTER_NEAREST)
#         nir      = cv2.resize(nir  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
#         # nir_mask = cv2.resize(nir_mask  ,(ow,oh), interpolation=cv2.INTER_NEAREST)
#         if short_size < self.crop_size:
#             # u_map_ = np.zeros((oh+padh,ow+padw))
#             # u_map_[:oh,:ow] = u_map
#             # u_map = u_map_
#             # v_map_ = np.zeros((oh+padh,ow+padw))
#             # v_map_[:oh,:ow] = v_map
#             # v_map = v_map_
#             aolp_ = np.zeros((oh+padh,ow+padw,3))
#             aolp_[:oh,:ow] = aolp
#             aolp = aolp_
#             dolp_ = np.zeros((oh+padh,ow+padw,3))
#             dolp_[:oh,:ow] = dolp
#             dolp = dolp_

#             img_ = np.zeros((oh+padh,ow+padw,3))
#             img_[:oh,:ow] = img
#             img = img_
#             SS_ = np.zeros((oh+padh,ow+padw))
#             SS_[:oh,:ow] = SS
#             SS = SS_
#             mask_ = np.full((oh+padh,ow+padw),self.fill)
#             mask_[:oh,:ow] = mask
#             mask = mask_
#             nir_ = np.zeros((oh+padh,ow+padw,3))
#             nir_[:oh,:ow] = nir
#             nir = nir_
#             nir_mask_ = np.zeros((oh+padh,ow+padw))
#             nir_mask_[:oh,:ow] = nir_mask
#             nir_mask = nir_mask_

#         u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         aolp  =  aolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         dolp  =  dolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         img   =   img[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         mask  =  mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         nir   =   nir[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         SS   =   SS[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         nir_mask = nir_mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        
#         # if 'scale' not in results:
#         #     self._random_scale(results)
#         # self._resize_img(results)
#         # try:
#         #     self._resize_multimodal(results)
#         # except:
#         #     pass
#         # try:
#         #     self._resize_seg(results)
#         # except:
#         #     pass
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += (f'(img_scale={self.img_scale}, '
#                      f'multiscale_mode={self.multiscale_mode}, '
#                      f'ratio_range={self.ratio_range}, '
#                      f'modalities_name={self.modalities_name}, '
#                      f'modalities_ch={self.modalities_ch}, '
#                      f'keep_ratio={self.keep_ratio})')
#         return repr_str
    
@PIPELINES.register_module()
class FixScaleCrop(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 seg_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 modalities_name=None,
                 modalities_ch=None,
                 base_size=None,
                 crop_size=None):
        # if img_scale is None:
        #     self.img_scale = None
        # else:
        #     if isinstance(img_scale, list):
        #         self.img_scale = img_scale
        #         self.seg_scale = seg_scale
        #     else:
        #         self.img_scale = [img_scale]
        #         self.seg_scale = [seg_scale]
        #     assert mmcv.is_list_of(self.img_scale, tuple)

        # if ratio_range is not None:
        #     # mode 1: given img_scale=None and a range of image ratio
        #     # mode 2: given a scale and a range of image ratio
        #     assert self.img_scale is None or len(self.img_scale) == 1
        # else:
        #     # mode 3 and 4: given multiple scales or a range of scales
        #     assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch
        self.base_size=base_size
        self.crop_size=crop_size
        self.modalities_ch_cumsum=np.cumsum(np.array(self.modalities_ch),0)

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    # def _resize_img(self, results):
    #     """Resize images with ``results['scale']``."""
    #     if self.keep_ratio:
    #         img, scale_factor = mmcv.imrescale(
    #             results['img'], results['scale'], return_scale=True)
    #         # the w_scale and h_scale has minor difference
    #         # a real fix should be done in the mmcv.imrescale in the future
    #         new_h, new_w = img.shape[:2]
    #         h, w = results['img'].shape[:2]
    #         w_scale = new_w / w
    #         h_scale = new_h / h
    #     else:
    #         img, w_scale, h_scale = mmcv.imresize(
    #             results['img'], results['scale'], return_scale=True)
    #     scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
    #                             dtype=np.float32)
    #     results['img'] = img
    #     results['img_shape'] = img.shape
    #     results['pad_shape'] = img.shape  # in case that there is no padding
    #     results['scale_factor'] = scale_factor
    #     results['keep_ratio'] = self.keep_ratio

    def _resize_multimodal(self, results):
        """Resize multimodal images with ``results['scale']``."""
        mm_matrix=[]
        for index in range(0,len(self.modalities_name)):
            if self.modalities_name[index]=="rgb":
                prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=self.modalities_ch_cumsum[index-1]
            if self.keep_ratio:
                # img_t=results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch[i]]
                img_t, scale_factor = mmcv.imrescale(
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img_t.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img_t, w_scale, h_scale = mmcv.imresize(
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[index]], results['scale'], return_scale=True)
            if len(img_t.shape)==2:
                img_t=img_t[...,np.newaxis]
            mm_matrix.append(img_t)
        mm_matrix = np.concatenate(mm_matrix, axis=2)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = mm_matrix
        results['img_shape'] = mm_matrix.shape
        results['pad_shape'] = mm_matrix.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        if self.seg_scale is None or self.seg_scale==[None]:
            for key in results.get('seg_fields', []):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key], results['scale'], interpolation='nearest')
                else:
                    gt_seg = mmcv.imresize(
                        results[key], results['scale'], interpolation='nearest')
                results[key] = gt_seg
        else:
            for key in results.get('seg_fields', []):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key], self.seg_scale[0], interpolation='nearest')
                else:
                    gt_seg = mmcv.imresize(
                        results[key], self.seg_scale[0], interpolation='nearest')
                results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        # self._resize_img(results)
        try:
            self._resize_multimodal(results)
        except:
            pass
        try:
            self._resize_seg(results)
        except:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'modalities_name={self.modalities_name}, '
                     f'modalities_ch={self.modalities_ch}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

@PIPELINES.register_module()
class Resize_4_test(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 seg_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
            self.seg_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
                self.seg_scale = seg_scale
            else:
                self.img_scale = [img_scale]
                self.seg_scale = [seg_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        # results['ori_shape'][:-1]=self.seg_scale[0]
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], self.seg_scale[0], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], self.seg_scale[0], interpolation='nearest')
            results[key] = gt_seg


    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        try:
            self._resize_img(results)
        except:
            pass
        try:
            self._resize_seg(results)
        except:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'seg_scale={self.seg_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str
    

@PIPELINES.register_module()
class Shift(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 x_trans,
                 y_trans,
                 prob=0.5,                                                                        
                 pad_val=0,
                 seg_pad_val=0
                #  center=None,
                #  auto_bound=False
                ):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        # if isinstance(degree, (float, int)):
        #     assert degree > 0, f'degree {degree} should be positive'
        #     self.degree = (-degree, degree)
        # else:
        #     self.degree = degree
        # assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
        #                               f'tuple of (min, max)'
        self.x_trans = x_trans
        self.y_trans = y_trans
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # self.center = center
        # self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        # rotate = True if np.random.rand() < self.prob else False
        # degree = np.random.uniform(min(*self.degree), max(*self.degree))
        translate= True if np.random.rand() < self.prob else False
        if translate:
            direction_x= np.random.uniform(-1,1)
            direction_y= np.random.uniform(-1,1)
            x_trans = direction_x*self.x_trans
            y_trans = direction_y*self.y_trans
            # if rotate:
            #     # rotate image
            #     results['img'] = mmcv.imrotate(
            #         results['img'],
            #         angle=degree,
            #         border_value=self.pal_val,
            #         center=self.center,
            #         auto_bound=self.auto_bound)

            #     # rotate segs
            #     for key in results.get('seg_fields', []):
            #         results[key] = mmcv.imrotate(
            #             results[key],
            #             angle=degree,
            #             border_value=self.seg_pad_val,
            #             center=self.center,
            #             auto_bound=self.auto_bound,
            #             interpolation='nearest')
            results['img'] = mmcv.imtranslate(
                img=results['img'],
                offset=x_trans,
                direction='horizontal',
                border_value=self.pad_val,
                interpolation='bilinear')
            results['img'] = mmcv.imtranslate(
                img=results['img'],
                offset=y_trans,
                direction='vertical',
                border_value=self.pad_val,
                interpolation='bilinear')
            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imtranslate(
                    img=results[key],
                    offset=x_trans,
                    direction='horizontal',
                    border_value=self.seg_pad_val,
                    interpolation='bilinear')
                results[key] = mmcv.imtranslate(
                    img=results[key],
                    offset=y_trans,
                    direction='vertical',
                    border_value=self.seg_pad_val,
                    interpolation='bilinear')
                
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        # repr_str += f'(prob={self.prob}, ' \
        #             f'degree={self.degree}, ' \
        #             f'pad_val={self.pal_val}, ' \
        #             f'seg_pad_val={self.seg_pad_val}, ' \
        #             f'center={self.center}, ' \
        #             f'auto_bound={self.auto_bound})'
        repr_str += f'(prob={self.prob}, ' \
                    f'x_trans={self.x_trans}, ' \
                    f'y_trans={self.y_trans}, ' \
                    f'pad_val={self.pad_val}, ' \
                    f'seg_pad_val={self.seg_pad_val})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortion_multimodal(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 modalities_name=None,
                 modalities_ch=None):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        if 'rgb' in modalities_name:
            self.in_ch_im=modalities_ch[modalities_name.index('rgb')]

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if self.in_ch_im:
            img = results['img'][:,:,:self.in_ch_im]
            # random brightness
            img = self.brightness(img)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                img = self.contrast(img)

            # random saturation
            img = self.saturation(img)

            # random hue
            img = self.hue(img)

            # random contrast
            if mode == 0:
                img = self.contrast(img)

            results['img'][:,:,:self.in_ch_im] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class Shift_multimodal(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 x_trans,
                 y_trans,
                 prob=0.5,                                                                        
                 pad_val=0,
                 seg_pad_val=255,
                 modalities_name = None,
                 modalities_ch = None,
                #  center=None,
                #  auto_bound=False
                ):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        # if isinstance(degree, (float, int)):
        #     assert degree > 0, f'degree {degree} should be positive'
        #     self.degree = (-degree, degree)
        # else:
        #     self.degree = degree
        # assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
        #                               f'tuple of (min, max)'
        self.x_trans = x_trans
        self.y_trans = y_trans
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.modalities_name = modalities_name
        self.modalities_ch = modalities_ch
        # self.center = center
        # self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        # rotate = True if np.random.rand() < self.prob else False
        # degree = np.random.uniform(min(*self.degree), max(*self.degree))
        translate= True if np.random.rand() < self.prob else False
        if translate:
            direction_x= np.random.uniform(-1,1)
            direction_y= np.random.uniform(-1,1)
            x_trans = direction_x*self.x_trans
            y_trans = direction_y*self.y_trans
            # if rotate:
            #     # rotate image
            #     results['img'] = mmcv.imrotate(
            #         results['img'],
            #         angle=degree,
            #         border_value=self.pal_val,
            #         center=self.center,
            #         auto_bound=self.auto_bound)

            #     # rotate segs
            #     for key in results.get('seg_fields', []):
            #         results[key] = mmcv.imrotate(
            #             results[key],
            #             angle=degree,
            #             border_value=self.seg_pad_val,
            #             center=self.center,
            #             auto_bound=self.auto_bound,
            #             interpolation='nearest')
            im_trans=[]
            modalities_ch_cumsum=np.cumsum(np.array(self.modalities_ch),0)
            for i in range(len(self.modalities_name)):
                if i==0:
                    prev_modalities_ch_cumsum=0
                else:
                    prev_modalities_ch_cumsum=modalities_ch_cumsum[i-1]
                im_trans_t_x=mmcv.imtranslate(
                    img=results['img'][:,:,prev_modalities_ch_cumsum:modalities_ch_cumsum[i]],
                    offset=x_trans,
                    direction='horizontal',
                    border_value=self.pad_val,
                    interpolation='bilinear')
                im_trans_t_y=mmcv.imtranslate(
                    img=im_trans_t_x,
                    offset=y_trans,
                    direction='vertical',
                    border_value=self.pad_val,
                    interpolation='bilinear')
                if len(im_trans_t_y.shape)==2:
                    im_trans_t_y=im_trans_t_y[...,np.newaxis]
                im_trans.append(im_trans_t_y)
                # im_trans.update({f"y_{self.modalities_name[i]}":im_trans_t_y})

            results['img']=np.concatenate(im_trans,axis=2)


            # if results['img'].shape[2]==self.in_ch_im:
            #     results['img'] = mmcv.imtranslate(
            #         img=results['img'],
            #         offset=x_trans,
            #         direction='horizontal',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     results['img'] = mmcv.imtranslate(
            #         img=results['img'],
            #         offset=y_trans,
            #         direction='vertical',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            # elif results['img'].shape[2]==self.in_ch_im+self.in_ch_in:
            #     img_trasx = mmcv.imtranslate(
            #         img=results['img'][:,:,:self.in_ch_im],
            #         offset=x_trans,
            #         direction='horizontal',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     img_trasy = mmcv.imtranslate(
            #         img=img_trasx,
            #         offset=y_trans,
            #         direction='vertical',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     in_img_trasx = mmcv.imtranslate(
            #         img=results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in],
            #         offset=x_trans,
            #         direction='horizontal',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     in_img_trasy = mmcv.imtranslate(
            #         img=in_img_trasx,
            #         offset=y_trans,
            #         direction='vertical',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     in_img_trasy=in_img_trasy[...,np.newaxis]
            #     if len(in_img_trasy.shape)==2:
            #         in_img_trasy=in_img_trasy[...,np.newaxis]
            #     results['img']=np.concatenate((img_trasy,in_img_trasy),axis=2)
            # elif results['img'].shape[2]>self.in_ch_im+self.in_ch_in:
            #     img_trasx = mmcv.imtranslate(
            #         img=results['img'][:,:,:self.in_ch_im],
            #         offset=x_trans,
            #         direction='horizontal',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     img_trasy = mmcv.imtranslate(
            #         img=img_trasx,
            #         offset=y_trans,
            #         direction='vertical',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     in_img_trasx = mmcv.imtranslate(
            #         img=results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in],
            #         offset=x_trans,
            #         direction='horizontal',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     in_img_trasy = mmcv.imtranslate(
            #         img=in_img_trasx,
            #         offset=y_trans,
            #         direction='vertical',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     xyz_img_trasx = mmcv.imtranslate(
            #         img=results['img'][:,:,self.in_ch_im+self.in_ch_in:],
            #         offset=x_trans,
            #         direction='horizontal',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     xyz_img_trasy = mmcv.imtranslate(
            #         img=xyz_img_trasx,
            #         offset=y_trans,
            #         direction='vertical',
            #         border_value=self.pad_val,
            #         interpolation='bilinear')
            #     if len(in_img_trasy.shape)==2:
            #         in_img_trasy=in_img_trasy[...,np.newaxis]
            #     if len(xyz_img_trasy.shape)==2:
            #         xyz_img_trasy=xyz_img_trasy[...,np.newaxis]
            #     results['img']=np.concatenate((img_trasy,in_img_trasy,xyz_img_trasy),axis=2)
            # translate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imtranslate(
                    img=results[key],
                    offset=x_trans,
                    direction='horizontal',
                    border_value=self.seg_pad_val,
                    interpolation='nearest')#'nearest'
                results[key] = mmcv.imtranslate(
                    img=results[key],
                    offset=y_trans,
                    direction='vertical',
                    border_value=self.seg_pad_val,
                    interpolation='nearest')#'nearest' when we have y_trans or x_trans i.e 3.6 it is interpolated the value of that pixel 
                # if np.max(np.unique(results[key]))>25 and np.max(np.unique(results[key]))!=self.seg_pad_val:
                #     print('error with unique values',np.unique(results[key]))
        return results
    

@PIPELINES.register_module()
class Normalize_multimodal_Muses(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    # def __init__(self, mean, std, to_rgb, mean_in, std_in,mean_xyz, std_xyz, in_ch_im=3,in_ch_in=1,in_ch_xyz=1):
    def __init__(self, mean, std, to_rgb, modalities_name, modalities_ch,norm_by_max=False):
        # self.in_ch_im=in_ch_im
        # self.in_ch_in=in_ch_in
        # self.in_ch_xyz=in_ch_xyz
        self.modalities_name = modalities_name
        self.modalities_ch = modalities_ch
        self.modalities_ch_cumsum=np.cumsum(np.array(modalities_ch))
        self.norm_by_max=norm_by_max
        # self.to_rgb = to_rgb
        self.mean = dict()
        self.std = dict()
        self.to_rgb = dict()
        for i in range(0,len(modalities_name)):
            if i==0:
                    prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=self.modalities_ch_cumsum[i-1]
            self.mean.update({f"mean_{self.modalities_name[i]}":np.array(mean[prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]], dtype=np.float32)})
            self.std.update({f"std_{self.modalities_name[i]}":np.array(std[prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]], dtype=np.float32)})
            self.to_rgb.update({f"to_rgb_{self.modalities_name[i]}":to_rgb[i]})
        # self.mean = np.array(mean, dtype=np.float32)
        # self.std = np.array(std, dtype=np.float32)
        # self.mean_in = np.array(mean_in, dtype=np.float32)
        # self.std_in = np.array(std_in, dtype=np.float32)
        # self.mean_xyz = np.array(mean_xyz, dtype=np.float32)
        # self.std_xyz = np.array(std_xyz, dtype=np.float32)


    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # try:
        #     if results['img_info']['in'] is not None:
        #         ind_in=3
        #         try: 
        #             if results['img_info']['x'] is not None:
        #                 ind_x=4
        #         except:
        #             pass
        # except:
        #     ind_in=3
        #     ind_x=3

        # results['img'][:,:,:self.in_ch_im] = mmcv.imnormalize(results['img'][:,:,:self.in_ch_im], self.mean, self.std,
                                        #   self.to_rgb)
        #normalize intensity map
        # try:
        #     if self.in_ch_in==3:
        #         results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in] = mmcv.imnormalize(results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in], self.mean_in, self.std_in, True)
        #     else:    
        #         results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in] = mmcv.imnormalize(results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in], self.mean_in, self.std_in, False)
        # except:
        #     pass
        # #normalize x  map  
        # try:
        #     results['img'][:,:,self.in_ch_im+self.in_ch_in:self.in_ch_im+self.in_ch_in+self.in_ch_xyz] = mmcv.imnormalize(results['img'][:,:,self.in_ch_im+self.in_ch_in:self.in_ch_im+self.in_ch_in+self.in_ch_xyz], self.mean_xyz, self.std_xyz, False)
        # except:
        #     pass
        #normalize y  map
        for i in range(0,len(self.modalities_name)):
            if i==0:
                prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=self.modalities_ch_cumsum[i-1]
            if self.modalities_name[i]=='rgb':
                if self.norm_by_max==False:
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] = mmcv.imnormalize(results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]], self.mean[f"mean_{self.modalities_name[i]}"], self.std[f"std_{self.modalities_name[i]}"], self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
                else:
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] = mmcv.imnormalize(results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]]/255, self.mean[f"mean_{self.modalities_name[i]}"], self.std[f"std_{self.modalities_name[i]}"], self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
            else:
                if self.norm_by_max==False:
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] = mmcv.imnormalize(results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] , self.mean[f"mean_{self.modalities_name[i]}"], self.std[f"std_{self.modalities_name[i]}"], self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
                else:
                    results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] = mmcv.imnormalize(results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] , self.mean[f"mean_{self.modalities_name[i]}"], self.std[f"std_{self.modalities_name[i]}"], self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
            if self.modalities_name[i]=='rgb':
                results['img_norm_cfg'] = dict(mean=self.mean[f"mean_{self.modalities_name[i]}"], std=self.std[f"std_{self.modalities_name[i]}"], to_rgb=self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
            else:
                results[f'{self.modalities_name[i]}_norm_cfg'] = dict(mean=self.mean[f"mean_{self.modalities_name[i]}"], std=self.std[f"std_{self.modalities_name[i]}"], to_rgb=self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])

        # results['img_norm_cfg'] = dict(
        #     mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        # results['in_norm_cfg'] = dict(
        #     mean=self.mean_in, std=self.std_in, to_rgb=False)
        # results['xyz_norm_cfg'] = dict(
        #     mean=self.mean_xyz, std=self.std_xyz, to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean[f"mean_{self.modalities_name[0]}"]}, std={self.std[f"std_{self.modalities_name[0]}"]}, to_rgb=' \
                    f'{self.to_rgb[f"to_rgb_{self.modalities_name[0]}"]})'
        for i in range(1,len(self.modalities_name)): 
            repr_str += f'(mean_{self.modalities_name[i]}={self.mean[f"mean_{self.modalities_name[i]}"]}, std_{self.modalities_name[i]}={self.std[f"std_{self.modalities_name[0]}"]},to_rgb_{self.modalities_name[i]}=' \
                    f'{self.to_rgb[f"to_rgb_{self.modalities_name[i]}"]})'
        return repr_str
@PIPELINES.register_module()
class Normalize_multimodal(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    # def __init__(self, mean, std, to_rgb, mean_in, std_in,mean_xyz, std_xyz, in_ch_im=3,in_ch_in=1,in_ch_xyz=1):
    def __init__(self, mean, std, to_rgb, modalities_name, modalities_ch,norm_by_max=False):
        # self.in_ch_im=in_ch_im
        # self.in_ch_in=in_ch_in
        # self.in_ch_xyz=in_ch_xyz
        self.modalities_name = modalities_name
        self.modalities_ch = modalities_ch
        self.modalities_ch_cumsum=np.cumsum(np.array(modalities_ch))
        self.norm_by_max=norm_by_max
        # self.to_rgb = to_rgb
        self.mean = dict()
        self.std = dict()
        self.to_rgb = dict()
        for i in range(0,len(modalities_name)):
            if i==0:
                    prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=self.modalities_ch_cumsum[i-1]
            self.mean.update({f"mean_{self.modalities_name[i]}":np.array(mean[prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]], dtype=np.float32)})
            self.std.update({f"std_{self.modalities_name[i]}":np.array(std[prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]], dtype=np.float32)})
            self.to_rgb.update({f"to_rgb_{self.modalities_name[i]}":to_rgb[i]})
        # self.mean = np.array(mean, dtype=np.float32)
        # self.std = np.array(std, dtype=np.float32)
        # self.mean_in = np.array(mean_in, dtype=np.float32)
        # self.std_in = np.array(std_in, dtype=np.float32)
        # self.mean_xyz = np.array(mean_xyz, dtype=np.float32)
        # self.std_xyz = np.array(std_xyz, dtype=np.float32)


    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # try:
        #     if results['img_info']['in'] is not None:
        #         ind_in=3
        #         try: 
        #             if results['img_info']['x'] is not None:
        #                 ind_x=4
        #         except:
        #             pass
        # except:
        #     ind_in=3
        #     ind_x=3

        # results['img'][:,:,:self.in_ch_im] = mmcv.imnormalize(results['img'][:,:,:self.in_ch_im], self.mean, self.std,
                                        #   self.to_rgb)
        #normalize intensity map
        # try:
        #     if self.in_ch_in==3:
        #         results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in] = mmcv.imnormalize(results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in], self.mean_in, self.std_in, True)
        #     else:    
        #         results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in] = mmcv.imnormalize(results['img'][:,:,self.in_ch_im:self.in_ch_im+self.in_ch_in], self.mean_in, self.std_in, False)
        # except:
        #     pass
        # #normalize x  map  
        # try:
        #     results['img'][:,:,self.in_ch_im+self.in_ch_in:self.in_ch_im+self.in_ch_in+self.in_ch_xyz] = mmcv.imnormalize(results['img'][:,:,self.in_ch_im+self.in_ch_in:self.in_ch_im+self.in_ch_in+self.in_ch_xyz], self.mean_xyz, self.std_xyz, False)
        # except:
        #     pass
        #normalize y  map
        for i in range(0,len(self.modalities_name)):
            if i==0:
                prev_modalities_ch_cumsum=0
            else:
                prev_modalities_ch_cumsum=self.modalities_ch_cumsum[i-1]
            if self.norm_by_max==False:
                results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] = mmcv.imnormalize(results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]], self.mean[f"mean_{self.modalities_name[i]}"], self.std[f"std_{self.modalities_name[i]}"], self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
            else:
                results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]] = mmcv.imnormalize(results['img'][:,:,prev_modalities_ch_cumsum:self.modalities_ch_cumsum[i]]/255, self.mean[f"mean_{self.modalities_name[i]}"], self.std[f"std_{self.modalities_name[i]}"], self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
            if self.modalities_name[i]=='rgb':
                results['img_norm_cfg'] = dict(mean=self.mean[f"mean_{self.modalities_name[i]}"], std=self.std[f"std_{self.modalities_name[i]}"], to_rgb=self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])
            else:
                results[f'{self.modalities_name[i]}_norm_cfg'] = dict(mean=self.mean[f"mean_{self.modalities_name[i]}"], std=self.std[f"std_{self.modalities_name[i]}"], to_rgb=self.to_rgb[f"to_rgb_{self.modalities_name[i]}"])

        # results['img_norm_cfg'] = dict(
        #     mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        # results['in_norm_cfg'] = dict(
        #     mean=self.mean_in, std=self.std_in, to_rgb=False)
        # results['xyz_norm_cfg'] = dict(
        #     mean=self.mean_xyz, std=self.std_xyz, to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean[f"mean_{self.modalities_name[0]}"]}, std={self.std[f"std_{self.modalities_name[0]}"]}, to_rgb=' \
                    f'{self.to_rgb[f"to_rgb_{self.modalities_name[0]}"]})'
        for i in range(1,len(self.modalities_name)): 
            repr_str += f'(mean_{self.modalities_name[i]}={self.mean[f"mean_{self.modalities_name[i]}"]}, std_{self.modalities_name[i]}={self.std[f"std_{self.modalities_name[0]}"]},to_rgb_{self.modalities_name[i]}=' \
                    f'{self.to_rgb[f"to_rgb_{self.modalities_name[i]}"]})'
        return repr_str
@PIPELINES.register_module()
class CropRect(object):
    """Down crop crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, box_crop, cat_max_ratio=1., ignore_index=None):
        assert box_crop[0] > 0 and box_crop[1] > 0 and box_crop[2] >0 and box_crop[3] >0
        self.box_crop = box_crop
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    # def get_crop_bbox(self, img):
    #     """get a crop bounding box."""
    #     # margin_h = max(img.shape[0] - self.crop_size[0], 0)
    #     # margin_w = max(img.shape[1] - self.crop_size[1], 0)
    #     # offset_h = np.random.randint(0, margin_h + 1)
    #     # offset_w = np.random.randint(0, margin_w + 1)
    #     # center_w
    #     # center_h = img.shape[0]/2
    #     # crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
    #     # crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
    #     v_mid,u_mid=np.asarray(img.shape[:2])/2
    #     u_mid=int(u_mid)
    #     v_mid=int(v_mid)
    #     for u in range(u_mid):
    #         if np.all(img[v_mid,u_mid-u]==np.zeros((3,))):
    #             u_min=u_mid-u
    #             break
    #     for v in range(v_mid):
    #         if np.all(img[v_mid-v,u_mid]==np.zeros((3,))):
    #             v_min=v_mid-v
    #             break
    #     for u in range(u_mid):
    #         if np.all(img[v_mid,u_mid+u]==np.zeros((3,))):
    #             u_max=u_mid+u
    #             break
    #     for v in range(v_mid):
    #         if np.all(img[v_mid-v,u_mid]==np.zeros((3,))):
    #             v_max=v_mid+v
    #             break
    #     crop_img=img[v_min:v_max, u_min:u_max]
    #     print(crop_img.shape)

        # return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img
             , crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_y1=int(crop_y1*img.shape[0])
        crop_y2=int(crop_y2*img.shape[0])
        crop_x1=int(crop_x1*img.shape[1])
        crop_x2=int(crop_x2*img.shape[1])
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        # img=Image.fromarray(img)
        # img=center_crop(img,output_size=self.crop_size)
        return np.asarray(img)

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        if results.get('img') is not None:
            img = results['img']
        # crop_bbox = self.get_crop_bbox(img)
        # if self.cat_max_ratio < 1.:
        #     # Repeat 10 times
        #     for _ in range(10):
        #         seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
        #         labels, cnt = np.unique(seg_temp, return_counts=True)
        #         cnt = cnt[labels != self.ignore_index]
        #         if len(cnt) > 1 and np.max(cnt) / np.sum(
        #                 cnt) < self.cat_max_ratio:
        #             break
        #         crop_bbox = self.get_crop_bbox(img)

        # crop the image
        # img = self.crop(img, crop_bbox)
            img=self.crop(img,self.box_crop)
        # img=self.crop(img)
            img_shape = img.shape
            results['img'] = img
            results['img_shape'] = img_shape
            results['flip']=None
            results['flip_direction']=None

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key]
                                    #   , crop_bbox)
                                    ,self.box_crop)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(box_crop={self.box_crop})'
 
@PIPELINES.register_module()
class Pad_multimodal(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=0,
                 ):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # self.modalities_name = modalities_name
        # self.modalities_ch = modalities_ch
        # self.in_ch_im=in_ch_im
        # self.in_ch_in=in_ch_in
        # self.in_ch_xyz=in_ch_xyz
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            # try:
            results[key] = impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)
            # except:
            #     results[key] = impad(
            #         results[key],
            #         shape=self.size,
            #         pad_val=self.seg_pad_val)
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        try:
            self._pad_img(results)
        except:
            pass
        try:
            self._pad_seg(results)
        except:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str
    

def impad(img,
          *,
          shape=None,
          padding=None,
          pad_val=0,
          padding_mode='constant'):
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    # border_type = {
    #     'constant': cv2.BORDER_CONSTANT,
    #     'edge': cv2.BORDER_REPLICATE,
    #     'reflect': cv2.BORDER_REFLECT_101,
    #     'symmetric': cv2.BORDER_REFLECT
    # }
    # img = cv2.copyMakeBorder(
    #     img,
    #     padding[1],
    #     padding[3],
    #     padding[0],
    #     padding[2],
    #     border_type[padding_mode],
    #     value=pad_val)
    if len(img.shape)==2:
        img = np.pad(img,((padding[1],padding[3]),(padding[0],padding[2])),mode=padding_mode,constant_values=pad_val)
    else:
        img=np.pad(img,((padding[1],padding[3]),(padding[0],padding[2]),(0,0)),mode=padding_mode,constant_values=pad_val)

    return img


def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)



# from mmcv_custom.builder import TRANSFORMS
# from mmcv_custom.base import BaseTransform
from typing import Dict, List, Optional, Sequence, Tuple, Union
# from mmcv_custom.transforms.utils import cache_randomness
from mmengine_custom.utils import is_seq_of

@PIPELINES.register_module()
# @TRANSFORMS.register_module()
class RandomChoiceResize(object):
    """Resize images & bbox & mask from a list of multiple scales.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. Resize scale will be randomly
    selected from ``scales``.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if `scale` is a list of tuple, the target scale is sampled from the list
      uniformally.
    - if `scale` is a tuple, the target scale will be set to the tuple.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Added Keys:

    - scale
    - scale_factor
    - scale_idx
    - keep_ratio


    Args:
        scales (Union[list, Tuple]): Images scales for resizing.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv_custom.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        scales: Sequence[Union[int, Tuple]],
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:
        super().__init__()
        if isinstance(scales, list):
            self.scales = scales
        else:
            self.scales = [scales]
        assert is_seq_of(self.scales, (tuple, int))

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Resize object
        # self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})
        self.resize = PIPELINES.build({'scale': [(0,0)], **self.resize_cfg})

    # @cache_randomness
    @staticmethod
    def _random_select(img_scales) -> Tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(img_scales))
        scale = img_scales[scale_idx]
        return scale, scale_idx
    
    # @staticmethod
    # def random_select(img_scales):
    #     """Randomly select an img_scale from given candidates.

    #     Args:
    #         img_scales (list[tuple]): Images scales for selection.

    #     Returns:
    #         (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
    #             where ``img_scale`` is the selected image scale and
    #             ``scale_idx`` is the selected index in the given candidates.
    #     """

    #     assert mmcv.is_list_of(img_scales, tuple)
    #     scale_idx = np.random.randint(len(img_scales))
    #     img_scale = img_scales[scale_idx]
    #     return img_scale, scale_idx

    def __call__(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        target_scale, scale_idx = self._random_select(self.scales)
        self.resize.scale = target_scale
        results = self.resize(results)
        results['scale_idx'] = scale_idx
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scales={self.scales}'
        repr_str += f', resize_cfg={self.resize_cfg})'
        return repr_str
# @TRANSFORMS.register_module()
# 
@PIPELINES.register_module()
class ResizeShortestEdge(object):
    """Resize the image and mask while keeping the aspect ratio unchanged.

    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py#L130 # noqa:E501
    Copyright (c) Facebook, Inc. and its affiliates.
    Licensed under the Apache-2.0 License

    This transform attempts to scale the shorter edge to the given
    `scale`, as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer
    edge does not exceed `max_size`.

    Required Keys:

    - img
    - gt_seg_map (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map (optional))

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio


    Args:
        scale (Union[int, Tuple[int, int]]): The target short edge length.
            If it's tuple, will select the min value as the short edge length.
        max_size (int): The maximum allowed longest edge length.
    """

    def __init__(self, scale: Union[int, Tuple[int, int]],
                 max_size: int) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size

        # Create a empty Resize object
        # self.resize = TRANSFORMS.build({
        #     'type': 'Resize',
        #     'scale': 0,
        #     'keep_ratio': True
        # })
        # self.resize = Resize_multimodal(img_scale=0,keep_ratio=True)
        self.resize=PIPELINES.build({'type': 'Resize',
            'img_scale': [(0,0)],
            'keep_ratio': True})

    def _get_output_shape(self, img, short_edge_length) -> Tuple[int, int]:
        """Compute the target image shape with the given `short_edge_length`.

        Args:
            img (np.ndarray): The input image.
            short_edge_length (Union[int, Tuple[int, int]]): The target short
                edge length. If it's tuple, will select the min value as the
                short edge length.
        """
        h, w = img.shape[:2]
        if isinstance(short_edge_length, int):
            size = short_edge_length * 1.0
        elif isinstance(short_edge_length, tuple):
            size = min(short_edge_length) * 1.0
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h *= scale
            new_w *= scale

        new_h = int(new_h + 0.5)
        new_w = int(new_w + 0.5)
        return (new_w, new_h)

    def __call__(self, results: Dict) -> Dict:
        self.resize.img_scale = [self._get_output_shape(results['img'], self.scale)]
        return self.resize(results)






