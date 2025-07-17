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


@DATASETS.register_module()
class DELIVER(CustomDataset):
    """DELIVER dataset.
    """

    
    CLASSES = ("Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation", 
                "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail", 
                "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck")

    PALETTE = [[70, 70, 70],
            [100, 40, 40],#Fence
            [55, 90, 80],#other
            [220, 20, 60],#pedestrian
            [153, 153, 153],
            [157, 234, 50],
            [128, 64, 128],
            [244, 35, 232],
            [107, 142, 35],
            [0, 0, 142],#Cars
            [102, 102, 156],
            [220, 220, 0],
            [70, 130, 180],
            [81, 0, 81],#Ground
            [150, 100, 100],#Bridge
            [230, 150, 140],#Railtrack
            [180, 165, 180],#Groundrail
            [250, 170, 30],#Trafficlight
            [110, 190, 160],
            [170, 120, 50],
            [45, 60, 150],
            [145, 170, 100],
            [  0,  0, 230], 
            [  0, 60, 100],
            [  0,  0, 70],
            ]
   
    def __init__(self,
                pipeline,
                img_dir,
                img_suffix='_rgb_front.png',
                ann_dir=None,
                seg_map_suffix='_semantic_front.png',
                split=None,
                data_root=None,
                test_mode=False,
                ignore_index=255,
                reduce_zero_label=False,
                classes=None,
                palette=None,
                gt_seg_map_loader_cfg=None,
                mod_dir=['samples/depth/training','samples/event/training','samples/lidar/training'],
                mod_suffix=['_depth_front.png','_event_front.png','_lidar_front.png'],
               
                modalities_name=None,
                modalities_ch=None,
                
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
        self.split = split
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

        if self.custom_classes:
            results['label_map'] = self.label_map 
    
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
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    if len(modalities_name)>1:
                        for i in range(1,len(modalities_name)):
                            if mod_dir_dict[f"{modalities_name[i]}_dir"] is not None:
                                mod_file = img_name + mod_suffix_dict[f"{modalities_name[i]}_suffix"]
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
                 metric='microIoU',
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
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'Fscore.lane', 'Precision.lane', 'Recall.lane','microIoU']
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
        mMiou_l=[]
        for keys_ext in ret_metrics.keys():
            # ret_metrics_t=ret_metrics[keys_ext]
            ret_metrics_summary[keys_ext]={}
            ret_metrics_class[keys_ext]={}
            eval_results[keys_ext]={}
            if metric[0]=='microIoU':
                if keys_ext!='global':   
                    for keys_int in ret_metrics[keys_ext].keys():
                        # summary table
                        if torch.is_tensor(ret_metrics[keys_ext][keys_int])==False:
                            ret_metrics_summary[keys_ext][keys_int] = OrderedDict({
                                # ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                                ret_metric:  np.round(np.nanmean(ret_metric_value) * 100,2)
                                for ret_metric, ret_metric_value in ret_metrics[keys_ext][keys_int].items()
                                if keys_int!='microIoU'
                            })
                            # ret_metrics_summary[keys_ext]['micro_IoU']
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

                            print_log(f'\n per class {keys_ext+"_"+keys_int} results:', logger)
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
                            ret_metrics_summary[keys_ext][keys_int] = OrderedDict({
                                # ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                                'microIoU':  np.round(np.nanmean(ret_metrics[keys_ext][keys_int].numpy()) * 100,2)
                            })
                            # ret_metrics[keys_ext][keys_int].pop('aAcc', None)
                            ret_metrics_class[keys_ext][keys_int] = OrderedDict({
                                # ret_metric: np.round(ret_metric_value * 100, 2)
                                keys_int: np.round(ret_metrics[keys_ext][keys_int].numpy() * 100,2)
                                # for ret_metric, ret_metric_value in ret_metrics[keys_ext][keys_int].items()
                            })
                            ret_metrics_class[keys_ext][keys_int].update({'Class': class_names})
                            ret_metrics_class[keys_ext][keys_int].move_to_end('Class', last=False)
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
                            mMiou_l.append(ret_metrics_summary[keys_ext][keys_int]['microIoU'])
                            
                            
                else:
                    # summary table
                    conditions=['ordinary','motionblur','overexposure','underexposure','eventlowres','lidarjitter']
                    ret_metrics_summary[keys_ext] = OrderedDict({
                        # ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                        ret_metric:  np.round(np.nanmean(np.array(ret_metric_value)) * 100,2)
                        for ret_metric, ret_metric_value in ret_metrics[keys_ext].items()
                        if ret_metric not in ['sun','rain','night','fog','cloud']
                    })
                    ret_metrics_summary_mMicro = OrderedDict({
                        'mMicroIoU': np.round(np.nanmean(mMiou_l),2)
                    })
                    

                    mMiou_per_condition_l=[]
                    for keys_int in ret_metrics[keys_ext].keys():
                        if keys_int in ['ordinary','motionblur','overexposure','underexposure','eventlowres','lidarjitter']:
                            mMiou_per_condition_l.append(np.nanmean(np.array(ret_metrics[keys_ext][keys_int])))
                    ret_metrics_summary_mMicro_per_condition = OrderedDict({
                        'mMicroIoU_per_condition': np.round(np.nanmean(mMiou_per_condition_l)*100,2)
                    })
                    
                    ret_metrics[keys_ext].pop('aAcc', None)
                    ret_metrics_class[keys_ext] = OrderedDict({
                        # ret_metric: np.round(ret_metric_value * 100, 2)
                        ret_metric: np.round(np.array(ret_metric_value) * 100,2)
                        for ret_metric, ret_metric_value in ret_metrics[keys_ext].items()
                        if ret_metric!='microIoU' and ret_metric not in ['sun','rain','night','fog','cloud']
                    })
                    
                    ret_metrics_class[keys_ext].update({'Class': class_names})
                    ret_metrics_class[keys_ext].move_to_end('Class', last=False)

                    # for logger
                    class_table_data = PrettyTable()
                    for key, val in ret_metrics_class[keys_ext].items():
                        class_table_data.add_column(key, val)

                    summary_table_data = PrettyTable()
                    for key, val in ret_metrics_summary[keys_ext].items():
                        if key == 'aAcc' or key == 'microIoU' or key in conditions:
                            summary_table_data.add_column(key, [val])
                        else:
                            summary_table_data.add_column('m' + key, [val])
                    summary_table_data_mMicro = PrettyTable()
                    for key, val in ret_metrics_summary_mMicro.items():
                        # if key == 'aAcc' or key == 'microIoU':
                        summary_table_data_mMicro.add_column("overall"+' '+key, [val])
                        # else:
                        #     summary_table_data_mMicro.add_column('m' + key, [val])
                        
                    summary_table_data_mMicro_per_condition = PrettyTable()
                    for key, val in ret_metrics_summary_mMicro_per_condition.items():
                        # if key == 'aAcc' or key == 'microIoU':
                        summary_table_data_mMicro_per_condition.add_column("overall"+' '+key, [val])
                        # else:
                        #     summary_table_data_mMicro.add_column('m' + key, [val])
                    
                    print_log(f'\n per class {keys_ext} results:', logger)
                    print_log('\n' + class_table_data.get_string(), logger=logger)
                    print_log(f'Summary  {keys_ext}:', logger)
                    print_log('\n' + summary_table_data.get_string(), logger=logger)
                    print_log(f'Summary  {keys_ext}:', logger)
                    print_log('\n' + summary_table_data_mMicro.get_string(), logger=logger)
                    print_log(f'Summary  {keys_ext}:', logger)
                    print_log('\n' + summary_table_data_mMicro_per_condition.get_string(), logger=logger)
                    
                    # each metric dict
                    eval_results_t = {}
                    for key, value in ret_metrics_summary[keys_ext].items():
                        if key == 'aAcc' or key in conditions:
                            eval_results_t[key] = value / 100.0
                        else:
                            eval_results_t['m' + key] = value / 100.0
                    for key, value in ret_metrics_summary_mMicro.items():
                        eval_results_t[key] = value / 100.0
                    for key, value in ret_metrics_summary_mMicro_per_condition.items():
                        eval_results_t[key] = value / 100.0
                    ret_metrics_class[keys_ext].pop('Class', None)
                    for key, value in ret_metrics_class[keys_ext].items():
                        if key in conditions:
                            eval_results_t.update({
                                key+'_microIoU' + '.' + str(name): value[idx] / 100.0
                                for idx, name in enumerate(class_names)
                            })
                        else:
                            eval_results_t.update({
                                key + '.' + str(name): value[idx] / 100.0
                                for idx, name in enumerate(class_names)
                            })
                    
                    eval_results[keys_ext]=eval_results_t
            else:
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

                        print_log(f'\n per class {keys_ext+"_"+keys_int} results:', logger)
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
    