from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from .pipelines.loading import LoadBinAnn
from mmseg.utils import get_root_logger
from mmcv.utils import print_log
from mmseg.datasets.pipelines import Compose
import numpy as np
from collections import OrderedDict
from prettytable import PrettyTable
import os.path as osp
from mmseg.datasets.pipelines import LoadAnnotations





import mmcv

def check_modality(modality_name):
    if modality_name == 'intensity':
        return 3
    elif modality_name == 'depth':
        return 2
    elif modality_name == 'x':
        return 4
    elif modality_name == 'y':
        return 5
    elif modality_name == 'z':
        return 6


@DATASETS.register_module()
class SINA_MM(CustomDataset):
    """Sina dataset.
    """
    CLASSES = ('background','lane','New_Jersey_bottom','New_Jersey_Top','Road_Curb_Inner_Edge','Road_Curb_Outer_Edge')
    #CLASSES = [('lane')]

    PALETTE = [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]]
    #PALETTE=[[255,255,255]]

    # def __init__(self, **kwargs):
    #     super(SINA_MM, self).__init__(
    #         img_suffix='_original.jpg',
    #         seg_map_suffix='_GT.png',
    #         # seg_map_suffix='_original.png',
    #         reduce_zero_label=False,
    #         ignore_index=None,
    #         **kwargs)
        # # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not osp.isabs(self.img_dir):
        #         self.img_dir = osp.join(self.data_root, self.img_dir)
        #     if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
        #         self.ann_dir = osp.join(self.data_root, self.ann_dir)
        #     if not (self.split is None or osp.isabs(self.split)):
        #         self.split = osp.join(self.data_root, self.split)

        # # load annotations
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #                                        self.seg_map_suffix, self.split)
    def __init__(self,
                pipeline,
                img_dir,
                img_suffix='original.jpg',
                ann_dir=None,
                seg_map_suffix='GT.png',
                split=None,
                data_root=None,
                test_mode=False,
                ignore_index=255,
                reduce_zero_label=False,
                classes=None,
                palette=None,
                gt_seg_map_loader_cfg=None,
                mod_dir=['HHA'],
                mod_suffix=['.jpg'],
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
        self.data_root_mod=data_root.replace('GroundTruth','MultiModal')
        self.data_root_red=data_root.replace('GroundTruth','')
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = {254:0,21:1,22:2,23:3,24:4,25:5}
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
                        self.mod_dir_dict.update({f"{modalities_name[i]}_dir":osp.join(self.data_root_mod, self.mod_dir[i-1])})
                        self.mod_suffix_dict.update({f"{modalities_name[i]}_suffix":self.mod_suffix[i-1]})


        # load annotations
        if len(modalities_name)>1:
            self.img_infos = self.load_annotations_modalities(self.img_dir, self.img_suffix, self.mod_dir_dict, self.mod_suffix_dict, self.modalities_name,
                                                self.ann_dir,self.seg_map_suffix, self.split)
        else:
            self.img_infos = self.load_annotations(self.img_dir, self.img_suffix, self.ann_dir,
                                                self.seg_map_suffix, self.split)

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
                source = osp.join(self.data_root_red, 'test.txt')
            elif split == 'val':
                source = osp.join(self.data_root_red, 'validation.txt')
            elif split == 'train':
                source = osp.join(self.data_root_red, 'train.txt')
            data_path=self.data_root_red.split('/')[0]
            with open(source) as f:
                files=f.readlines()
                # img_names =[]
                for line in files:
                    img_name = line.strip()
                    if '\t' in img_name:
                        filenames = img_name.split('\t')
                        img_name = data_path+'/'+filenames[0]
                        # img_name = img_name.split('\t')[0].split('/')[-1]
                    # img_names.append(img_name)
                    img_info = dict(filename=img_name)
                    if ann_dir is not None:
                        seg_map=data_path+'/'+filenames[1]
                        # seg_map = img_name.replace(img_dir.split('/')[-1],ann_dir.split('/')[-1]).replace(img_suffix,seg_map_suffix)
                        # seg_map = img_name.replace(img_suffix,seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    if len(modalities_name)>1:
                        for i in range(1,len(modalities_name)):
                            num=check_modality(modalities_name[i])
                            if mod_dir_dict[f"{modalities_name[i]}_dir"] is not None:
                                mod_file = data_path+'/'+filenames[num]
                                if modalities_name[i]=='intensity':
                                    mod_file = mod_file.replace('.zlib','.png')
                                # mod_file = img_name.replace(img_suffix,mod_suffix_dict[f"{modalities_name[i]}_suffix"]) #replace(img_dir.split('/')[-1],mod_dir_dict[f"{modalities_name[i]}_dir"].split('/')[-1])
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
                            mod_file = img.replace(img_suffix, mod_suffix_dict[f"{modalities_name[i]}_suffix"])
                            img_info[modalities_name[i]] = dict({f"{modalities_name[i]}_file":mod_file})
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos    
    
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory. len of training set 14383 14383 14383
        len of validation set 2056 2056 2056
        len of test set 4111 4111 4111

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
            # with open(split) as f:
            #     for line in f:
            #         img_name = line.strip()
            #         img_info = dict(filename=img_name + img_suffix)
            #         if ann_dir is not None:
            #             seg_map = img_name + seg_map_suffix
            #             img_info['ann'] = dict(seg_map=seg_map)
            #         img_infos.append(img_info)
            assert split in ['train', 'val','test']
            if split == 'test':
                source = osp.join(self.data_root_red, 'test.txt')
            elif split == 'val':
                source = osp.join(self.data_root_red, 'validation.txt')
            elif split == 'train':
                source = osp.join(self.data_root_red, 'train.txt')
            # source = osp.join(self.data_root_red, 'test.txt') if split == 'val' else osp.join(self.data_root_red, 'train.txt')
            data_path=self.data_root_red.split('/')[0]
            with open(source) as f:
                files=f.readlines()
                # img_names =[]
                for line in files:
                    img_name = line.strip()
                    if '\t' in img_name:
                        filenames = img_name.split('\t')
                        img_name = data_path+'/'+filenames[0]
                        # img_name = img_name.split('\t')[0].split('/')[-1]
                    # img_names.append(img_name)
                    img_info = dict(filename=img_name)
                    if ann_dir is not None:
                        seg_map= data_path+'/'+filenames[1]
                        # seg_map = img_name.replace(img_dir.split('/')[-1],ann_dir.split('/')[-1]).replace(img_suffix,seg_map_suffix)
                        # seg_map = img_name.replace(img_suffix,seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)

                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
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
        self.gt_seg_map_loader = [LoadAnnotations(self.reduce_zero_label)] + self.pipeline.transforms[1:index]
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
    

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None
        # if self.custom_classes:
        results['label_map'] = self.label_map
            
    
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
            ret_metrics = pre_eval_to_metrics(results, metric)

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
    
    def evaluate_temp(self,
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
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'Fscore.lane']
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
            ret_metrics = pre_eval_to_metrics(results, metric)

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


        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.nanmean(ret_metric_value)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: ret_metric_value
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value
            else:
                eval_results['m' + key] = value

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx]
                for idx, name in enumerate(class_names)
            })

        return eval_results