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
from mmseg.datasets.pipelines import Compose
from mmseg_custom.datasets.pipelines import LoadAnnotationsov as LoadAnnotations
import torch




import mmcv
from mmseg_custom.apis.evaluation import pre_eval_to_metrics_dict
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor



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
class NYUDv2var(CustomDataset):
    """NYUDv2 dataset.

    num_classes: 40
    
    """

    
    CLASSES= ('wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop')
    
    PALETTE = [[128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
        [128, 64, 128],
        [0, 192, 128],
        [128, 192, 128],
        [64, 64, 0],
        [192, 64, 0],
        [64, 192, 0],
        [192, 192, 0],
        [64, 64, 128],
        [192, 64, 128],
        [64, 192, 128],
        [192, 192, 128],
        [0, 0, 64],
        [128, 0, 64],
        [0, 128, 64],
        [128, 128, 64],
        [0, 0, 192],
        [128, 0, 192],
        [0, 128, 192],
        [128, 128, 192],
        [64, 0, 64]]

    def __init__(self,
                pipeline,
                img_dir,
                img_suffix='.jpg',
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
                mod_dir=['hha'],
                mod_suffix=['.png'],
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
            assert split in ['train', 'val']
            source = osp.join(self.data_root, 'test.txt') if split == 'val' else osp.join(self.data_root, 'train.txt')
            with open(source) as f:
                files=f.readlines()
                # img_names =[]
                for line in files:
                    img_name = line.strip()
                    if '\t' in img_name:
                        img_name = img_name.split('\t')[0].split('/')[-1]
                    # img_names.append(img_name)
                    img_info = dict(filename=img_name)
                    if ann_dir is not None:
                        # seg_map = img_name.replace(img_dir.split('/')[-1],ann_dir.split('/')[-1]).replace(img_suffix,seg_map_suffix)
                        seg_map = img_name.replace(img_suffix,seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    if len(modalities_name)>1:
                        for i in range(1,len(modalities_name)):
                            if mod_dir_dict[f"{modalities_name[i]}_dir"] is not None:
                                if 'hha' in mod_dir_dict[f"{modalities_name[i]}_dir"]:
                                    img_name_=img_name.split('.')[-2]
                                    tot_len=6
                                    img_len=len(img_name_)
                                    mod_name='0'*(tot_len-img_len)+img_name
                                else:
                                    mod_name=img_name
                                mod_file = mod_name.replace(img_suffix,mod_suffix_dict[f"{modalities_name[i]}_suffix"]) #replace(img_dir.split('/')[-1],mod_dir_dict[f"{modalities_name[i]}_dir"].split('/')[-1])
                                # mod_file = img_name + mod_suffix_dict[f"{modalities_name[i]}_suffix"]
                                img_info[modalities_name[i]] = dict({f"{modalities_name[i]}_file":mod_file})

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
                            if 'hha' in mod_dir_dict[f"{modalities_name[i]}_dir"]:
                                img_name=img.split('.')[-2]
                                tot_len=6
                                img_len=len(img_name)
                                mod_name='0'*(tot_len-img_len)+img
                            else:
                                mod_name=img
                            mod_file = mod_name.replace(img_suffix, mod_suffix_dict[f"{modalities_name[i]}_suffix"])
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
# import torchvision.transforms.functional as TF 
# from torchvision import io
# from pathlib import Path
# from typing import Tuple
# import glob
# import einops
# from torch.utils.data import DataLoader
# from torch.utils.data import DistributedSampler, RandomSampler
# # from semseg.augmentations_mm import get_train_augmentation

# class NYU(Dataset):
#     """
#     num_classes: 40
#     """
#     CLASSES = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
#     'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
#     'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
#     'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

#     PALETTE = None

#     def __init__(self, root: str = 'data/NYUDepthv2', split: str = 'train', transform = None, modals = ['img', 'depth'], case = None) -> None:
#         super().__init__()
#         assert split in ['train', 'val']
#         self.root = root
#         self.transform = transform
#         self.n_classes = len(self.CLASSES)
#         self.ignore_label = 255
#         self.modals = modals
#         self.files = self._get_file_names(split)
#         if not self.files:
#             raise Exception(f"No images found in {img_path}")
#         print(f"Found {len(self.files)} {split} images.")

#     def __len__(self) -> int:
#         return len(self.files)
    
    
    # @classmethod
    # def get_class_colors(*args):
    #     def uint82bin(n, count=8):
    #         """returns the binary of integer n, count refers to amount of bits"""
    #         return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    #     N = 41
    #     cmap = np.zeros((N, 3), dtype=np.uint8)
    #     for i in range(N):
    #         r, g, b = 0, 0, 0
    #         id = i
    #         for j in range(7):
    #             str_id = uint82bin(id)
    #             r = r ^ (np.uint8(str_id[-1]) << (7 - j))
    #             g = g ^ (np.uint8(str_id[-2]) << (7 - j))
    #             b = b ^ (np.uint8(str_id[-3]) << (7 - j))
    #             id = id >> 3
    #         cmap[i, 0] = r
    #         cmap[i, 1] = g
    #         cmap[i, 2] = b
    #     class_colors = cmap.tolist()
    #     return class_colors

    # def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
    #     item_name = str(self.files[index])
    #     rgb = os.path.join(*[self.root, 'RGB', item_name+'.jpg'])
    #     x1 = os.path.join(*[self.root, 'HHA', item_name+'.jpg'])
    #     lbl_path = os.path.join(*[self.root, 'Label', item_name+'.png'])

    #     sample = {}
    #     sample['img'] = io.read_image(rgb)[:3, ...]
    #     if 'depth' in self.modals:
    #         sample['depth'] = self._open_img(x1)
    #     if 'lidar' in self.modals:
    #         raise NotImplementedError()
    #     if 'event' in self.modals:
    #         raise NotImplementedError()
    #     label = io.read_image(lbl_path)[0,...].unsqueeze(0)
    #     label[label==255] = 0
    #     label -= 1
    #     sample['mask'] = label
        
    #     if self.transform:
    #         sample = self.transform(sample)
    #     label = sample['mask']
    #     del sample['mask']
    #     label = self.encode(label.squeeze().numpy()).long()
    #     sample = [sample[k] for k in self.modals]
    #     return sample, label

    # def _open_img(self, file):
    #     img = io.read_image(file)
    #     C, H, W = img.shape
    #     if C == 4:
    #         img = img[:3, ...]
    #     if C == 1:
    #         img = img.repeat(3, 1, 1)
    #     return img

    # def encode(self, label: Tensor) -> Tensor:
    #     return torch.from_numpy(label)

    # def _get_file_names(self, split_name):
    #     assert split_name in ['train', 'val']
    #     source = os.path.join(self.root, 'test.txt') if split_name == 'val' else os.path.join(self.root, 'train.txt')
    #     file_names = []
    #     with open(source) as f:
    #         files = f.readlines()
    #     for item in files:
    #         file_name = item.strip()
    #         if ' ' in file_name:
    #             file_name = file_name.split(' ')[0]
    #         file_names.append(file_name)
    #     return file_names


# if __name__ == '__main__':
#     traintransform = get_train_augmentation((480, 640), seg_fill=255)

#     trainset = NYU(transform=traintransform, split='val')
#     trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

#     for i, (sample, lbl) in enumerate(trainloader):
#         print(torch.unique(lbl))

if __name__ == '__main__':

    # Create the dataset
    dataset = build_dataset(
        dict(
            type='NYUDv2',
            split='train',
            data_root='/path/to/data',
            img_dir='image',
            ann_dir='annotation',
            pipeline=[
                dict(type='LoadImageandModalities', modalities_name=['rgb','depth'], modalities_ch=[3,1]),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(640, 480), ratio_range=(0.5, 2.0)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]
        )
    )

    # # Create the model
    # model = build_segmentor(
    #     dict(
    #         type='EncoderDecoder',
    #         backbone=dict(
    #             type='VisionTransformer',
    #             img_size=640,
    #             patch_size=16,
    #             embed_dims=768,
    #             num_layers=12,
    #             num_heads=12,
    #             mlp_ratio=4,
    #             qkv_bias=True,
    #             drop_rate=0.0,
    #             attn_drop_rate=0.0,
    #             drop_path_rate=0.1,
    #             norm_cfg=dict(type='LN', eps=1e-6),
    #             act_cfg=dict(type='GELU'),
    #             init_cfg=dict(type='TruncNormal', std=0.02)
    #         ),
    #         decode_head=dict(
    #             type='VisionTransformerHead',
    #             in_channels=768,
    #             channels=768,
    #             in_index=11,
    #             num_classes=40,
    #             dropout_ratio=0.1,
    #             norm_cfg=dict(type='LN', eps=1e-6),
    #             act_cfg=dict(type='GELU'),
    #             loss_decode=dict(
    #                 type='CrossEntropyLoss',
    #                 use_sigmoid=False,
    #                 loss_weight=1.0
    #             )
    #         ),
    #         train_cfg=dict(),
    #         test_cfg=dict(mode='whole')
    #     )
    # )

    # # Train the model
    # train_segmentor(
    #     model,
    #     dataset,
    #     cfg=dict(
    #         optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    #         lr_config=dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False),
    #         runner=dict(type='IterBasedRunner', max_iters=80000),
    #         checkpoint_config=dict(interval=5000),
    #         log_config=dict(interval=100),
    #         total_epochs=100
    #     ),
    #     distributed=False,
    #     validate=True,
    #     meta=dict()
    # )