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
from mmseg_custom.datasets.pipelines import LoadAnnotationsov as LoadAnnotations
from mmseg.datasets.pipelines import Compose
# ,LoadAnnotations
import torch




import mmcv
from mmseg_custom.apis.evaluation import pre_eval_to_metrics_dict
import random



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
class FMB_val(CustomDataset):
    # """MFNet dataset.

    # num_classes: 9
    
    # """
    # CLASSES = ('unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump')
    # PALETTE = [[0,0,0],[64,0,128],[64,64,0],[0,128,192],[0,0,192],[128,128,0],[64,64,128],[192,128,128],[192,64,0]]
    
    """
    num_classes: 14
    """
    CLASSES = ["Road", "Sidewalk", "Building", "Traffic Light", "Traffic Sign", "Vegetation", "Sky", "Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pole"]

    # PALETTE = torch.tensor([
    #         [70, 70, 70],
    #         [100, 40, 40],
    #         [55, 90, 80],
    #         [220, 20, 60],
    #         [153, 153, 153],
    #         [157, 234, 50],
    #         [128, 64, 128],
    #         [244, 35, 232],
    #         [107, 142, 35],
    #         [0, 0, 142],
    #         [102, 102, 156],
    #         [220, 220, 0],
    #         [70, 130, 180],
    #         [81, 0, 81],
    #         # [150, 100, 100],
    #         ])
    PALETTE = [
        [179, 228, 228], # road
        [181, 57, 133],  # sidewalk
        [67, 162, 177],  # building
        [200, 178, 50],  # lamp
        [132, 45, 199],  # sign
        [66, 172, 84],   # vegetation
        [179, 73, 79],   # sky
        [76, 99, 166],   # person
        [66, 121, 253],  # car
        [137, 165, 91],  # truck
        [155, 97, 152],  # bus
        [105, 153, 140], # motorcycle
        [222, 215, 158], # bicycle
        [135, 113, 90],  # pole
    ]

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
                mod_dir=['therm'],
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
            else:
                self.mod_dir_dict=None
                self.mod_suffix_dict=None

        # load annotations
        # if len(modalities_name)>1:
        self.img_infos = self.load_annotations_modalities(self.img_dir, self.img_suffix, self.mod_dir_dict, self.mod_suffix_dict, self.modalities_name,
                                                self.ann_dir,self.seg_map_suffix, self.split)
        # else:
        #     self.img_infos = self.load_annotations(self.img_dir, self.img_suffix, self.ann_dir,
        #                                         self.seg_map_suffix, self.split)

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
                source_easy = osp.join(self.data_root,split,'Visible', 'test_easy_files.txt')
                source_hard = osp.join(self.data_root,split, 'Visible','test_hard_files.txt')
                # files = [f.readlines() for f in [source_easy,source_hard]]
            elif split == 'val':
                source_easy = osp.join(self.data_root,'val','Visible', 'val_easy_files.txt')
                source_hard = osp.join(self.data_root,'val','Visible', 'val_hard_files.txt')
                # files = [f.readlines() for f in [source_easy,source_hard]]
            elif split == 'train':
                source_easy = osp.join(self.data_root,split,'Visible', 'train_easy_files.txt')
                source_hard = osp.join(self.data_root,split,'Visible', 'train_hard_files.txt')
                # files = [f.readlines() for f in [opsource_easy,source_hard]]
            with open(source_easy) as f:
                files_easy=f.readlines()
                files_easy = ['easy/' + file for file in files_easy]
            with open(source_hard) as f:
                files_hard=f.readlines()
                files_hard = ['hard/' + file for file in files_hard]
            files=files_hard+files_easy
            random.shuffle(files)
                # img_names =[]
            for line in files:
                img_name = line.strip()
                # if '\t' in img_name:
                #     img_name = img_name.split('\t')[0].split('/')[-1]
                # img_names.append(img_name)
                img_info = dict(filename=img_name)#+img_suffix
                if ann_dir is not None:
                    # seg_map = img_name.replace(img_dir.split('/')[-1],ann_dir.split('/')[-1]).replace(img_suffix,seg_map_suffix)
                    seg_map = img_name.replace(img_suffix,seg_map_suffix).replace('easy/','').replace('hard/','')
                    img_info['ann'] = dict(seg_map=seg_map)#+seg_map_suffix
                if len(modalities_name)>1:
                    for i in range(1,len(modalities_name)):
                        if mod_dir_dict[f"{modalities_name[i]}_dir"] is not None:
                            mod_file = img_name.replace(img_suffix,mod_suffix_dict[f"{modalities_name[i]}_suffix"]).replace('easy/','').replace('hard/','') #replace(img_dir.split('/')[-1],mod_dir_dict[f"{modalities_name[i]}_dir"].split('/')[-1])
                            # mod_file = img_name + mod_suffix_dict[f"{modalities_name[i]}_suffix"]
                            img_info[modalities_name[i]] = dict({f"{modalities_name[i]}_file":mod_file})#+mod_suffix_dict[f"{modalities_name[i]}_suffix"]}

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
            if ("MultiScaleFlipAug" in self.pipeline.transforms[index].__repr__()) or ("Resize_multimodal" in self.pipeline.transforms[index].__repr__()):
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
            ret_metrics = pre_eval_to_metrics(results, metric,nan_to_num=0)#nan_to_num=0

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
            ret_metrics = pre_eval_to_metrics(results, metric,nan_to_num=0)#nan_to_num=0

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
# from pathlib import Path
# from typing import Tuple
# import glob
# import einops
# from torch.utils.data import DataLoader
# from torch.utils.data import DistributedSampler, RandomSampler
# from semseg.augmentations_mm import get_train_augmentation

# class MFNet(Dataset):
#     """
#     num_classes: 9
#     """
#     CLASSES = ['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
#     PALETTE = torch.tensor([[64,0,128],[64,64,0],[0,128,192],[0,0,192],[128,128,0],[64,64,128],[192,128,128],[192,64,0]])

#     def __init__(self, root: str = 'data/MFNet', split: str = 'train', transform = None, modals = ['img', 'thermal'], case = None) -> None:
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
    
#     def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
#         item_name = str(self.files[index])
#         rgb = os.path.join(*[self.root, 'rgb', item_name+'.jpg'])
#         x1 = os.path.join(*[self.root, 'ther', item_name+'.jpg'])
#         lbl_path = os.path.join(*[self.root, 'labels', item_name+'.png'])
#         sample = {}
#         sample['img'] = io.read_image(rgb)[:3, ...]
#         if 'thermal' in self.modals:
#             sample['thermal'] = self._open_img(x1)
#         label = io.read_image(lbl_path)[0,...].unsqueeze(0)
#         sample['mask'] = label
        
#         if self.transform:
#             sample = self.transform(sample)
#         label = sample['mask']
#         del sample['mask']
#         label = self.encode(label.squeeze().numpy()).long()
#         sample = [sample[k] for k in self.modals]
#         return sample, label

#     def _open_img(self, file):
#         img = io.read_image(file)
#         C, H, W = img.shape
#         if C == 4:
#             img = img[:3, ...]
#         if C == 1:
#             img = img.repeat(3, 1, 1)
#         return img

#     def encode(self, label: Tensor) -> Tensor:
#         return torch.from_numpy(label)

#     def _get_file_names(self, split_name):
#         assert split_name in ['train', 'val']
#         source = os.path.join(self.root, 'test.txt') if split_name == 'val' else os.path.join(self.root, 'train.txt')
#         file_names = []
#         with open(source) as f:
#             files = f.readlines()
#         for item in files:
#             file_name = item.strip()
#             if ' ' in file_name:
#                 file_name = file_name.split(' ')[0]
#             file_names.append(file_name)
#         return file_names


# if __name__ == '__main__':
#     traintransform = get_train_augmentation((480, 640), seg_fill=255)

#     trainset = MFNet(transform=traintransform)
#     trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

#     for i, (sample, lbl) in enumerate(trainloader):
#         print(torch.unique(lbl))