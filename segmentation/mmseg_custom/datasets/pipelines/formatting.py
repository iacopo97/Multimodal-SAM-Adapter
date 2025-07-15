# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module(force=True)
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(to_tensor(
                results['gt_semantic_seg'][None, ...].astype(np.int64)),
                                            stack=True)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(to_tensor(results['gt_masks']))
        if 'gt_labels' in results:
            results['gt_labels'] = DC(to_tensor(results['gt_labels']))

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ToMask(object):
    """Transfer gt_semantic_seg to binary mask and generate gt_labels."""
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def __call__(self, results):
        gt_semantic_seg = results['gt_semantic_seg']
        gt_labels = np.unique(gt_semantic_seg)
        # remove ignored region
        gt_labels = gt_labels[gt_labels != self.ignore_index]

        gt_masks = []
        for class_id in gt_labels:
            gt_masks.append(gt_semantic_seg == class_id)

        if len(gt_masks) == 0:
            # Some image does not have annotation (all ignored)
            gt_masks = np.empty((0, ) + results['pad_shape'][:-1], dtype=np.int64)
            gt_labels = np.empty((0, ),  dtype=np.int64)
        else:
            gt_masks = np.asarray(gt_masks, dtype=np.int64)
            gt_labels = np.asarray(gt_labels, dtype=np.int64)

        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(ignore_index={self.ignore_index})'
    

@PIPELINES.register_module()
class Collectmod(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_semantic_seg".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding
        
        - "ori_img_shape": image shape at beginning
        - "in_shape": in shape
        - "x_shape": x shape
        - "y_shape": y shape
        - "z_shape": z shape
        - 'in_shape_ext': in shape extended
        - 'x_shape_ext': x shape extended
        - 'y_shape_ext': y shape extended
        - 'z_shape_ext': z shape extended

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: (``filename``, ``ori_filename``, ``ori_shape``,
            ``img_shape``, ``pad_shape``, ``scale_factor``, ``flip``,
            ``flip_direction``, ``img_norm_cfg``)
    """

    def __init__(self,
                 keys,
                 meta_keys=['filename', 'ori_filename', 'ori_shape', 
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'ori_img_shape', 'in_shape', 'x_shape', 'y_shape', 'z_shape', 'in_shape_ext', 'x_shape_ext', 'y_shape_ext', 'z_shape_ext'], modalities_name=['rgb','depth','event','lidar'], modalities_ch=[3,3,3,1]):
        self.keys = keys
        self.meta_keys = meta_keys
        for i in range(1,len(modalities_name)):
            self.meta_keys.append(f'{modalities_name[i]}_shape')
            self.meta_keys.append(f'{modalities_name[i]}_shape_ext')
            self.meta_keys.append(f'{modalities_name[i]}_norm_cfg')
        self.meta_keys= tuple(self.meta_keys)

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            try:
                img_meta[key] = results[key]
            except:
                pass
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'

