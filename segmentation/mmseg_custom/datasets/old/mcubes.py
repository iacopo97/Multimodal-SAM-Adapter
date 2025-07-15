from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
# from .pipelines.transform import LoadBinAnn
from mmseg.utils import get_root_logger
from mmcv.utils import print_log
from mmseg.datasets.pipelines import Compose
import numpy as np
from collections import OrderedDict
from prettytable import PrettyTable
import os.path as osp
from mmseg.datasets.pipelines import Compose,LoadAnnotations
import torch




import mmcv
from mmseg_custom.apis.evaluation import pre_eval_to_metrics_dict



# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset
# from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
# from .pipelines.transform import LoadBinAnn
# from mmseg.utils import get_root_logger
# from mmcv.utils import print_log
# from mmseg.datasets.pipelines import Compose
# import numpy as np
# from collections import OrderedDict
# from prettytable import PrettyTable
# import os.path as osp





# import mmcv

@DATASETS.register_module()
class MCubes(CustomDataset):
#     ├── MCubeS
# │   ├── polL_color
# │   ├── polL_aolp
# │   ├── polL_dolp
# │   ├── NIR_warped
# │   └── SS
    """
    num_classes: 20
    """
    CLASSES = ('asphalt','concrete','metal','road_marking','fabric','glass','plaster','plastic','rubber','sand',
    'gravel','ceramic','cobblestone','brick','grass','wood','leaf','water','human','sky')

    PALETTE = [[ 44, 160,  44],
                [ 31, 119, 180],
                [255, 127,  14],
                [214,  39,  40],
                [140,  86,  75],
                [127, 127, 127],
                [188, 189,  34],
                [255, 152, 150],
                [ 23, 190, 207],
                [174, 199, 232],
                [196, 156, 148],
                [197, 176, 213],
                [247, 182, 210],
                [199, 199, 199],
                [219, 219, 141],
                [158, 218, 229],
                [ 57,  59, 121],
                [107, 110, 207],
                [156, 158, 222],
                [ 99, 121,  57]]
    

    def __init__(self,
                pipeline,
                img_dir,
                img_suffix='.png',
                ann_dir=None,
                seg_map_suffix='.png',
                split=None,
                data_root=None,
                test_mode=False,
                ignore_index=255,
                reduce_zero_label=False,
                classes=None,
                palette=None,
                gt_seg_map_loader_cfg=None,
                mod_dir=None,
                mod_suffix=None,
                # ev_dir=None,
                # ev_suffix='_event_front.png',
                # lid_dir=None,
                # lid_suffix='_lidar_front.png',
                # depth_dir=None,
                # depth_suffix='_depth_front.png',
                modalities_name=None,
                modalities_ch=None,
                # z_dir=None,
                # z_suffix='_Z.tiff',
                **kwargs):
        self.modalities_name = modalities_name
        self.modalities_ch = modalities_ch
        self.mod_dir=mod_dir
        self.mod_suffix=mod_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if len(modalities_name)>1:
                self.mod_dir_dict=dict()
                self.mod_suffix_dict=dict()
                for i in range(1,len(modalities_name)):
                    if not (self.mod_dir[i-1] is None or osp.isabs(self.mod_dir[i-1])):
                        self.mod_dir_dict.update({f"{modalities_name[i]}_dir":osp.join(self.data_root, self.mod_dir[i-1])})
                        self.mod_suffix_dict.update({f"{modalities_name[i]}_suffix":self.mod_suffix[i-1]})


        # load annotations
        if len(modalities_name)>1:
            self.img_infos = self.load_annotations_modalities(self.img_dir, self.img_suffix, self.mod_dir_dict, self.mod_suffix_dict, self.modalities_name,
                                                self.ann_dir,self.seg_map_suffix, self.split)
        else:
            self.img_infos = self.load_annotations(self.img_dir, self.img_suffix, self.ann_dir,
                                                self.seg_map_suffix, self.split)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if len(self.modalities_name)>1:
            for i in range(1,len(self.modalities_name)):
                results[f"{self.modalities_name[i]}_prefix"] = self.mod_dir_dict[f"{self.modalities_name[i]}_dir"]
            # results['event_prefix'] = self.ev_dir
            # results['lidar_prefix'] = self.lid_dir
            # results['depth_prefix'] = self.depth_dir
        if self.custom_classes:
            results['label_map'] = self.label_map 
    
    # def _get_file_names(self, split):
    #     assert split in ['train', 'val']
    #     source = os.path.join(self.root, 'test.txt') if split == 'val' else os.path.join(self.root, 'train.txt')
    #     file_names = []
    #     with open(source) as f:
    #         files = f.readlines()
    #     for item in files:
    #         file_name = item.strip()
    #         if ' ' in file_name:
    #             file_name = file_name.split(' ')[0]
    #         file_names.append(file_name)
    #     return file_names
    
    def load_annotations_modalities(self, img_dir, img_suffix,mod_dir_dict, mod_suffix_dict,modalities_name, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory. 

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            # assert split in ['train', 'val']
            # source = os.path.join(self.data_root, 'test.txt') if split == 'val' else os.path.join(self.data_root, 'train.txt')
            # with open(source) as f:
            #     for line in f:
            #         img_name = line.strip()
            #         img_info = dict(filename=img_name + img_suffix)
            #         if ann_dir is not None:
            #             seg_map = img_name + seg_map_suffix
            #             img_info['ann'] = dict(seg_map=seg_map)
            #         if len(modalities_name)>1:
            #             for i in range(1,len(modalities_name)):
            #                 if mod_dir_dict[f"{modalities_name[i]}_dir"] is not None:
            #                     mod_file = img_name + mod_suffix_dict[f"{modalities_name[i]}_suffix"]
            #                     img_info[modalities_name[i]] = dict({f"{modalities_name[i]}_file":mod_file})

            #         img_infos.append(img_info)
            assert split in ['train', 'val','test']
            if split == 'test':
                source = osp.join(self.data_root, 'list_folder/test.txt')
            elif split == 'val':
                source = osp.join(self.data_root, 'list_folder/val.txt')
            elif split == 'train':
                source = osp.join(self.data_root, 'list_folder/train.txt')
            with open(source) as f:
                files=f.readlines()
                # img_names =[]
                for line in files:
                    img_name = line.strip()
                    # if '\t' in img_name:
                    #     img_name = img_name.split('\t')[0].split('/')[-1]
                    # img_names.append(img_name)
                    img_info = dict(filename=img_name+img_suffix)
                    if ann_dir is not None:
                        # seg_map = img_name.replace(img_dir.split('/')[-1],ann_dir.split('/')[-1]).replace(img_suffix,seg_map_suffix)
                        seg_map = img_name.replace(img_suffix,seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map+seg_map_suffix)
                    if len(modalities_name)>1:
                        for i in range(1,len(modalities_name)):
                            if mod_dir_dict[f"{modalities_name[i]}_dir"] is not None:
                                mod_file = img_name.replace(img_suffix,mod_suffix_dict[f"{modalities_name[i]}_suffix"]) #replace(img_dir.split('/')[-1],mod_dir_dict[f"{modalities_name[i]}_dir"].split('/')[-1])
                                # mod_file = img_name + mod_suffix_dict[f"{modalities_name[i]}_suffix"]
                                img_info[modalities_name[i]] = dict({f"{modalities_name[i]}_file":mod_file+mod_suffix_dict[f"{modalities_name[i]}_suffix"]})

                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if len(modalities_name)>1:
                    for i in range(1,len(modalities_name)):
                        if mod_dir_dict[f"{modalities_name[i]}_dir"] is not None:
                            mod_file = img.replace(img_suffix, mod_suffix_dict[f"{modalities_name[i]}_suffix"])
                            img_info[modalities_name[i]] = dict({f"{modalities_name[i]}_file":mod_file})
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
    
    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        for index in range(len(self.pipeline.transforms)):
            if "MultiScaleFlipAug" in self.pipeline.transforms[index].__repr__():
                break
        self.gt_seg_map_loader = [LoadAnnotations()] + self.pipeline.transforms[1:index]
        self.gt_seg_map_loader=Compose(self.gt_seg_map_loader)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']
    
    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']
    
    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the label map has already been applied and zero label
                    # has already been reduced by get_gt_seg_map_by_idx() i.e.
                    # LoadAnnotations.__call__(), these operations should not
                    # be duplicated. See the following issues/PRs:
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # https://github.com/open-mmlab/mmsegmentation/pull/1417
                    # https://github.com/open-mmlab/mmsegmentation/pull/2504
                    # for more details
                    label_map=dict(),
                    reduce_zero_label=False))

        return pre_eval_results
    
    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'Fscore.lane', 'Precision.lane', 'Recall.lane']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        num_classes = len(self.CLASSES)
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            # num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics_dict(results, metric, nan_to_num=0, num_classes=num_classes)
        
        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_summary={}
        ret_metrics_class={}
        for keys_ext in ret_metrics.keys():
            # ret_metrics_t=ret_metrics[keys_ext]
            ret_metrics_summary[keys_ext]={}
            ret_metrics_class[keys_ext]={}
            eval_results[keys_ext]={}
            if keys_ext!='global':   
                for keys_int in ret_metrics[keys_ext].keys():
                    # summary table
                    ret_metrics_summary[keys_ext][keys_int] = OrderedDict({
                        # ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                        ret_metric:  np.round(np.nanmean(ret_metric_value) * 100,2)
                        for ret_metric, ret_metric_value in ret_metrics[keys_ext][keys_int].items()
                    })

                    # each class table
                    ret_metrics[keys_ext][keys_int].pop('aAcc', None)
                    ret_metrics_class[keys_ext][keys_int] = OrderedDict({
                        # ret_metric: np.round(ret_metric_value * 100, 2)
                        ret_metric: np.round(ret_metric_value * 100,2)
                        for ret_metric, ret_metric_value in ret_metrics[keys_ext][keys_int].items()
                    })
                    ret_metrics_class[keys_ext][keys_int].update({'Class': class_names})
                    ret_metrics_class[keys_ext][keys_int].move_to_end('Class', last=False)

                    # for logger
                    class_table_data = PrettyTable()
                    for key, val in ret_metrics_class[keys_ext][keys_int].items():
                        class_table_data.add_column(key, val)

                    summary_table_data = PrettyTable()
                    for key, val in ret_metrics_summary[keys_ext][keys_int].items():
                        if key == 'aAcc':
                            summary_table_data.add_column(key, [val])
                        else:
                            summary_table_data.add_column('m' + key, [val])

                    print_log(f'per class {keys_ext+"_"+keys_int} results:', logger)
                    print_log('\n' + class_table_data.get_string(), logger=logger)
                    print_log(f'Summary  {keys_ext+"_"+keys_int}:', logger)
                    print_log('\n' + summary_table_data.get_string(), logger=logger)

                    # each metric dict
                    eval_results_t = {}
                    for key, value in ret_metrics_summary[keys_ext][keys_int].items():
                        if key == 'aAcc':
                            eval_results_t[key] = value / 100.0
                        else:
                            eval_results_t['m' + key] = value / 100.0

                    ret_metrics_class[keys_ext][keys_int].pop('Class', None)
                    for key, value in ret_metrics_class[keys_ext][keys_int].items():
                        eval_results_t.update({
                            key + '.' + str(name): value[idx] / 100.0
                            for idx, name in enumerate(class_names)
                        })
                    eval_results[keys_ext][keys_int]=eval_results_t
            else:
                # summary table
                ret_metrics_summary[keys_ext] = OrderedDict({
                    # ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                    ret_metric:  np.round(np.nanmean(ret_metric_value) * 100,2)
                    for ret_metric, ret_metric_value in ret_metrics[keys_ext].items()
                })

                # each class table
                ret_metrics[keys_ext].pop('aAcc', None)
                ret_metrics_class[keys_ext] = OrderedDict({
                    # ret_metric: np.round(ret_metric_value * 100, 2)
                    ret_metric: np.round(ret_metric_value * 100,2)
                    for ret_metric, ret_metric_value in ret_metrics[keys_ext].items()
                })
                ret_metrics_class[keys_ext].update({'Class': class_names})
                ret_metrics_class[keys_ext].move_to_end('Class', last=False)

                # for logger
                class_table_data = PrettyTable()
                for key, val in ret_metrics_class[keys_ext].items():
                    class_table_data.add_column(key, val)

                summary_table_data = PrettyTable()
                for key, val in ret_metrics_summary[keys_ext].items():
                    if key == 'aAcc':
                        summary_table_data.add_column(key, [val])
                    else:
                        summary_table_data.add_column('m' + key, [val])

                print_log(f'per class {keys_ext} results:', logger)
                print_log('\n' + class_table_data.get_string(), logger=logger)
                print_log(f'Summary  {keys_ext}:', logger)
                print_log('\n' + summary_table_data.get_string(), logger=logger)

                # each metric dict
                eval_results_t = {}
                for key, value in ret_metrics_summary[keys_ext].items():
                    if key == 'aAcc':
                        eval_results_t[key] = value / 100.0
                    else:
                        eval_results_t['m' + key] = value / 100.0

                ret_metrics_class[keys_ext].pop('Class', None)
                for key, value in ret_metrics_class[keys_ext].items():
                    eval_results_t.update({
                        key + '.' + str(name): value[idx] / 100.0
                        for idx, name in enumerate(class_names)
                    })
                eval_results[keys_ext]=eval_results_t

        return eval_results
    
    def evaluate_old(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'Fscore.lane', 'Precision.lane', 'Recall.lane']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric, nan_to_num=0)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results
    
# import os
# import torch 
# import numpy as np
# from torch import Tensor
# from torch.utils.data import Dataset
# from torchvision import io
# from torchvision import transforms
# from pathlib import Path
# from typing import Tuple
# import glob
# import einops
# from torch.utils.data import DataLoader
# from torch.utils.data import DistributedSampler, RandomSampler
# from semseg.augmentations_mm import get_train_augmentation
# import cv2
# import random
# from PIL import Image, ImageOps, ImageFilter

# class MCubeS(Dataset):
#     """
#     num_classes: 20
#     """
#     CLASSES = ['asphalt','concrete','metal','road_marking','fabric','glass','plaster','plastic','rubber','sand',
#     'gravel','ceramic','cobblestone','brick','grass','wood','leaf','water','human','sky',]

#     PALETTE = torch.tensor([[ 44, 160,  44],
#                 [ 31, 119, 180],
#                 [255, 127,  14],
#                 [214,  39,  40],
#                 [140,  86,  75],
#                 [127, 127, 127],
#                 [188, 189,  34],
#                 [255, 152, 150],
#                 [ 23, 190, 207],
#                 [174, 199, 232],
#                 [196, 156, 148],
#                 [197, 176, 213],
#                 [247, 182, 210],
#                 [199, 199, 199],
#                 [219, 219, 141],
#                 [158, 218, 229],
#                 [ 57,  59, 121],
#                 [107, 110, 207],
#                 [156, 158, 222],
#                 [ 99, 121,  57]])

#     def __init__(self, root: str = 'data/MCubeS/multimodal_dataset', split: str = 'train', transform = None, modals = ['image', 'aolp', 'dolp', 'nir'], case = None) -> None:
#         super().__init__()
#         assert split in ['train', 'val']
#         self.split = split
#         self.root = root
#         self.transform = transform
#         self.n_classes = len(self.CLASSES)
#         self.ignore_label = 255
#         self.modals = modals
#         self._left_offset = 192
        	
#         self.img_h = 1024
#         self.img_w = 1224
#         max_dim = max(self.img_h, self.img_w)
#         u_vec = (np.arange(self.img_w)-self.img_w/2)/max_dim*2
#         v_vec = (np.arange(self.img_h)-self.img_h/2)/max_dim*2
#         self.u_map, self.v_map = np.meshgrid(u_vec, v_vec)
#         self.u_map = self.u_map[:,:self._left_offset]

#         self.base_size = 512
#         self.crop_size = 512
#         self.files = self._get_file_names(split)
    
#         if not self.files:
#             raise Exception(f"No images found in {img_path}")
#         print(f"Found {len(self.files)} {split} images.")

#     def __len__(self) -> int:
#         return len(self.files)
    
#     def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
#         item_name = str(self.files[index])
#         rgb = os.path.join(*[self.root, 'polL_color', item_name+'.png'])
#         x1 = os.path.join(*[self.root, 'polL_aolp_sin', item_name+'.npy'])
#         x1_1 = os.path.join(*[self.root, 'polL_aolp_cos', item_name+'.npy'])
#         x2 = os.path.join(*[self.root, 'polL_dolp', item_name+'.npy'])
#         x3 = os.path.join(*[self.root, 'NIR_warped', item_name+'.png'])
#         lbl_path = os.path.join(*[self.root, 'GT', item_name+'.png'])
#         nir_mask = os.path.join(*[self.root, 'NIR_warped_mask', item_name+'.png'])
#         _mask = os.path.join(*[self.root, 'SS', item_name+'.png'])

#         _img = cv2.imread(rgb,-1)[:,:,::-1]
#         _img = _img.astype(np.float32)/65535 if _img.dtype==np.uint16 else _img.astype(np.float32)/255
#         _target = cv2.imread(lbl_path,-1)
#         _mask = cv2.imread(_mask,-1)
#         _aolp_sin = np.load(x1)
#         _aolp_cos = np.load(x1_1)
#         _aolp = np.stack([_aolp_sin, _aolp_cos, _aolp_sin], axis=2) # H x W x 3
#         dolp = np.load(x2)
#         _dolp = np.stack([dolp, dolp, dolp], axis=2) # H x W x 3
#         nir  = cv2.imread(x3,-1)
#         nir = nir.astype(np.float32)/65535 if nir.dtype==np.uint16 else nir.astype(np.float32)/255
#         _nir = np.stack([nir, nir, nir], axis=2) # H x W x 3

#         _nir_mask = cv2.imread(nir_mask,0)

#         _img, _target, _aolp, _dolp, _nir, _nir_mask, _mask = _img[:,self._left_offset:], _target[:,self._left_offset:], \
#                _aolp[:,self._left_offset:], _dolp[:,self._left_offset:], \
#                _nir[:,self._left_offset:], _nir_mask[:,self._left_offset:], _mask[:,self._left_offset:]
#         sample = {'image': _img, 'label': _target, 'aolp': _aolp, 'dolp': _dolp, 'nir': _nir, 'nir_mask': _nir_mask, 'u_map': self.u_map, 'v_map': self.v_map, 'mask':_mask}

#         if self.split == "train":
#             sample = self.transform_tr(sample)
#         elif self.split == 'val':
#             sample = self.transform_val(sample)
#         elif self.split == 'test':
#             sample = self.transform_val(sample)
#         else:
#             raise NotImplementedError()
#         label = sample['label'].long()
#         sample = [sample[k] for k in self.modals]
#         return sample, label

#     def transform_tr(self, sample):
#         composed_transforms = transforms.Compose([
#             RandomHorizontalFlip(),
#             RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
#             RandomGaussianBlur(),
#             Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensor()])

#         return composed_transforms(sample)

#     def transform_val(self, sample):
#         composed_transforms = transforms.Compose([
#             FixScaleCrop(crop_size=1024),
#             Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensor()])

#         return composed_transforms(sample)

#     def _get_file_names(self, split_name):
#         assert split_name in ['train', 'val']
#         source = os.path.join(self.root, 'list_folder/test.txt') if split_name == 'val' else os.path.join(self.root, 'list_folder/train.txt')
#         file_names = []
#         with open(source) as f:
#             files = f.readlines()
#         for item in files:
#             file_name = item.strip()
#             if ' ' in file_name:
#                 # --- KITTI-360
#                 file_name = file_name.split(' ')[0]
#             file_names.append(file_name)
#         return file_names

# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         mean (tuple): means for each channel.
#         std (tuple): standard deviations for each channel.
#     """
#     def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         img = np.array(img).astype(np.float32)
#         mask = np.array(mask).astype(np.float32)
#         img -= self.mean
#         img /= self.std

#         nir = sample['nir']
#         nir = np.array(nir).astype(np.float32)
#         # nir /= 255

#         return {'image': img,
#                 'label': mask,
#                 'aolp' : sample['aolp'], 
#                 'dolp' : sample['dolp'], 
#                 'nir'  : nir, 
#                 'nir_mask': sample['nir_mask'],
#                 'u_map': sample['u_map'],
#                 'v_map': sample['v_map'],
#                 'mask':sample['mask']}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         img = sample['image']
#         mask = sample['label']
#         aolp = sample['aolp']
#         dolp = sample['dolp']
#         nir  = sample['nir']
#         nir_mask  = sample['nir_mask']
#         SS=sample['mask']

#         img = np.array(img).astype(np.float32).transpose((2, 0, 1))
#         mask = np.array(mask).astype(np.float32)
#         aolp = np.array(aolp).astype(np.float32).transpose((2, 0, 1))
#         dolp = np.array(dolp).astype(np.float32).transpose((2, 0, 1))
#         SS = np.array(SS).astype(np.float32)
#         nir = np.array(nir).astype(np.float32).transpose((2, 0, 1))
#         nir_mask = np.array(nir_mask).astype(np.float32)
        
#         img = torch.from_numpy(img).float()
#         mask = torch.from_numpy(mask).float()
#         aolp = torch.from_numpy(aolp).float()
#         dolp = torch.from_numpy(dolp).float()
#         SS = torch.from_numpy(SS).float()
#         nir = torch.from_numpy(nir).float()
#         nir_mask = torch.from_numpy(nir_mask).float()

#         u_map = sample['u_map']
#         v_map = sample['v_map']
#         u_map = torch.from_numpy(u_map.astype(np.float32)).float()
#         v_map = torch.from_numpy(v_map.astype(np.float32)).float()

#         return {'image': img,
#                 'label': mask,
#                 'aolp' : aolp,
#                 'dolp' : dolp,
#                 'nir'  : nir,
#                 'nir_mask'  : nir_mask,
#                 'u_map': u_map,
#                 'v_map': v_map,
#                 'mask':SS}


# class RandomHorizontalFlip(object):
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         aolp = sample['aolp']
#         dolp = sample['dolp']
#         nir  = sample['nir']
#         nir_mask  = sample['nir_mask']
#         u_map = sample['u_map']
#         v_map = sample['v_map']
#         SS=sample['mask']
#         if random.random() < 0.5:
#             # img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
#             # nir = nir.transpose(Image.FLIP_LEFT_RIGHT)

#             img = img[:,::-1]
#             mask = mask[:,::-1]
#             nir = nir[:,::-1]
#             nir_mask = nir_mask[:,::-1]
#             aolp  = aolp[:,::-1]
#             dolp  = dolp[:,::-1]
#             SS  = SS[:,::-1]
#             u_map = u_map[:,::-1]

#         return {'image': img,
#                 'label': mask,
#                 'aolp' : aolp,
#                 'dolp' : dolp,
#                 'nir'  : nir,
#                 'nir_mask'  : nir_mask,
#                 'u_map': u_map,
#                 'v_map': v_map,
#                 'mask':SS}

# class RandomGaussianBlur(object):
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         nir  = sample['nir']
#         if random.random() < 0.5:
#             radius = random.random()
#             # img = img.filter(ImageFilter.GaussianBlur(radius=radius))
#             # nir = nir.filter(ImageFilter.GaussianBlur(radius=radius))
#             img = cv2.GaussianBlur(img, (0,0), radius)
#             nir = cv2.GaussianBlur(nir, (0,0), radius)

#         return {'image': img,
#                 'label': mask,
#                 'aolp' : sample['aolp'], 
#                 'dolp' : sample['dolp'], 
#                 'nir'  : nir, 
#                 'nir_mask': sample['nir_mask'],
#                 'u_map': sample['u_map'],
#                 'v_map': sample['v_map'],
#                 'mask':sample['mask']}

# class RandomScaleCrop(object):
#     def __init__(self, base_size, crop_size, fill=255):
#         self.base_size = base_size
#         self.crop_size = crop_size
#         self.fill = fill

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         aolp = sample['aolp']
#         dolp = sample['dolp']
#         nir = sample['nir']
#         nir_mask = sample['nir_mask']
#         SS=sample['mask']
#         # random scale (short edge)
#         short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
#         # w, h = img.size
#         h, w = img.shape[:2]
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

#         u_map = sample['u_map']
#         v_map = sample['v_map']
#         u_map    = cv2.resize(u_map,(ow,oh))
#         v_map    = cv2.resize(v_map,(ow,oh))
#         aolp     = cv2.resize(aolp ,(ow,oh))
#         dolp     = cv2.resize(dolp ,(ow,oh))
#         SS     = cv2.resize(SS ,(ow,oh))
#         img      = cv2.resize(img  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
#         mask     = cv2.resize(mask ,(ow,oh), interpolation=cv2.INTER_NEAREST)
#         nir      = cv2.resize(nir  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
#         nir_mask = cv2.resize(nir_mask  ,(ow,oh), interpolation=cv2.INTER_NEAREST)
#         if short_size < self.crop_size:
#             u_map_ = np.zeros((oh+padh,ow+padw))
#             u_map_[:oh,:ow] = u_map
#             u_map = u_map_
#             v_map_ = np.zeros((oh+padh,ow+padw))
#             v_map_[:oh,:ow] = v_map
#             v_map = v_map_
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
#         return {'image': img,
#                 'label': mask,
#                 'aolp' : aolp,
#                 'dolp' : dolp,
#                 'nir'  : nir,
#                 'nir_mask'  : nir_mask,
#                 'u_map': u_map,
#                 'v_map': v_map,
#                 'mask':SS}

# class FixScaleCrop(object):
#     def __init__(self, crop_size):
#         self.crop_size = crop_size

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         aolp = sample['aolp']
#         dolp = sample['dolp']
#         nir = sample['nir']
#         nir_mask = sample['nir_mask']
#         SS = sample['mask']

#         # w, h = img.size
#         h, w = img.shape[:2]

#         if w > h:
#             oh = self.crop_size
#             ow = int(1.0 * w * oh / h)
#         else:
#             ow = self.crop_size
#             oh = int(1.0 * h * ow / w)
#         # img = img.resize((ow, oh), Image.BILINEAR)
#         # mask = mask.resize((ow, oh), Image.NEAREST)
#         # nir = nir.resize((ow, oh), Image.BILINEAR)

#         # center crop
#         # w, h = img.size
#         # h, w = img.shape[:2]
#         x1 = int(round((ow - self.crop_size) / 2.))
#         y1 = int(round((oh - self.crop_size) / 2.))
#         # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#         # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#         # nir = nir.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

#         u_map = sample['u_map']
#         v_map = sample['v_map']
#         u_map = cv2.resize(u_map,(ow,oh))
#         v_map = cv2.resize(v_map,(ow,oh))
#         aolp  = cv2.resize(aolp ,(ow,oh))
#         dolp  = cv2.resize(dolp ,(ow,oh))
#         SS  = cv2.resize(SS ,(ow,oh))
#         img   = cv2.resize(img  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
#         mask  = cv2.resize(mask ,(ow,oh), interpolation=cv2.INTER_NEAREST)
#         nir   = cv2.resize(nir  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
#         nir_mask = cv2.resize(nir_mask,(ow,oh), interpolation=cv2.INTER_NEAREST)
#         u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         aolp  =  aolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         dolp  =  dolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         img   =   img[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         mask  =  mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         SS  =  SS[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         nir   =   nir[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         nir_mask = nir_mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
#         return {'image': img,
#                 'label': mask,
#                 'aolp' : aolp,
#                 'dolp' : dolp,
#                 'nir'  : nir,
#                 'nir_mask'  : nir_mask,
#                 'u_map': u_map,
#                 'v_map': v_map,
#                 'mask':SS}


# if __name__ == '__main__':
#     traintransform = get_train_augmentation((1024, 1224), seg_fill=255)

#     trainset = MCubeS(transform=traintransform, split='val')
#     trainloader = DataLoader(trainset, batch_size=1, num_workers=0, drop_last=False, pin_memory=False)

#     for i, (sample, lbl) in enumerate(trainloader):
#         print(torch.unique(lbl))