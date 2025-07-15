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
# ,LoadAnnotations
from mmseg_custom.datasets.pipelines import LoadAnnotations_Muses as LoadAnnotations
import torch
import re



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
class MUSES(CustomDataset):
    """MUSES dataset.
    """
    
    # CLASSES = ("Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation", 
    #             "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail", 
    #             "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck")

    # PALETTE = [[70, 70, 70],
    #         [100, 40, 40],#Fence
    #         [55, 90, 80],#other
    #         [220, 20, 60],#pedestrian
    #         [153, 153, 153],
    #         [157, 234, 50],
    #         [128, 64, 128],
    #         [244, 35, 232],
    #         [107, 142, 35],
    #         [0, 0, 142],#Cars
    #         [102, 102, 156],
    #         [220, 220, 0],
    #         [70, 130, 180],
    #         [81, 0, 81],#Ground
    #         [150, 100, 100],#Bridge
    #         [230, 150, 140],#Railtrack
    #         [180, 165, 180],#Groundrail
    #         [250, 170, 30],#Trafficlight
    #         [110, 190, 160],
    #         [170, 120, 50],
    #         [45, 60, 150],
    #         [145, 170, 100],
    #         [  0,  0, 230], 
    #         [  0, 60, 100],
    #         [  0,  0, 70],
    #         ]
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    
    
    def __init__(self,
                pipeline,
                img_dir,
                img_suffix='_frame_camera.png',
                ann_dir=None,
                seg_map_suffix='_gt_labelTrainIds.png',
                split=None,
                data_root=None,
                test_mode=False,
                ignore_index=255,
                reduce_zero_label=False,
                classes=None,
                palette=None,
                gt_seg_map_loader_cfg=None,
                mod_dir=['projected_to_rgb/event_camera/train','projected_to_rgb/lidar/train'],
                mod_suffix=['_event_camera.npz','_lidar.npz'],
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
                cases=['clear','rain','fog','snow'],
                conditions=['day','night'],
                **kwargs):
        # super(SYNTHIA_multimodal, self).__init__(
        #     img_suffix='.png',
        #     seg_map_suffix='.png',
        #     reduce_zero_label=False,
        #     ignore_index=None,
        #     **kwargs)
        # join paths if data_root is specified
        # self.ev_dir = ev_dir
        # self.ev_suffix = ev_suffix
        # self.lid_dir = lid_dir
        # self.lid_suffix = lid_suffix
        # self.depth_dir = depth_dir
        # self.depth_suffix = depth_suffix
        self.cases=cases
        self.conditions=conditions 
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
            # results['event_prefix'] = self.ev_dir
            # results['lidar_prefix'] = self.lid_dir
            # results['depth_prefix'] = self.depth_dir
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
            for case in self.cases:
                for condition in self.conditions:
                    for img in mmcv.scandir(img_dir+'/'+case+'/'+condition, img_suffix, recursive=True):
                        img=case+'_'+condition+'_'+img
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
        # summary table
        # ret_metrics_summary_global = OrderedDict()
        # for keys in ret_metrics.keys():
            # ret_metrics_summary_global = OrderedDict({
            #     ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            #     for ret_metric, ret_metric_value in ret_metrics[keys].items()
            # })
        # ret_metrics_summary_global=OrderedDict({
        #     key: np.round(np.nanmean([ret_metrics[keys][key] for keys in ret_metrics.keys()]) * 100,2)
        #     for key in ret_metrics[keys].keys()
        # })
        # # each class table
        # ret_metrics.pop('aAcc', None)
        # # ret_metr_global=np.empty(0)
        # ret_metrics_class_global = OrderedDict({
        # #     ret_metric: np.round(ret_metric_value * 100, 2)
        # #     for ret_metric, ret_metric_value in ret_metrics.items()
        # # })
        #     key: [(ret_metrics[keys][key]* 100) for keys in ret_metrics.keys()] 
        #     for key in ret_metrics[keys].keys()
        # })
        # ret_metrics_class_global.update({'Class': class_names})
        # ret_metrics_class_global.move_to_end('Class', last=False)

        # # for logger
        # class_table_data_global = PrettyTable()
        # for key, val in ret_metrics_class_global.items():
        #     class_table_data_global.add_column(key, val)

        # summary_table_data = PrettyTable()
        # for key, val in ret_metrics_summary.items():
        #     if key == 'aAcc':
        #         summary_table_data.add_column(key, [val])
        #     else:
        #         summary_table_data.add_column('m' + key, [val])

        # print_log(f'per class {keys} results:', logger)
        # print_log('\n' + class_table_data.get_string(), logger=logger)
        # print_log(f'Summary  {keys}:', logger)
        # print_log('\n' + summary_table_data.get_string(), logger=logger)

        # # each metric dict
        # for key, value in ret_metrics_summary.items():
        #     if key == 'aAcc':
        #         eval_results['global'] = value / 100.0
        #     else:
        #         eval_results['global']['m' + key] = value / 100.0

        # ret_metrics_class.pop('Class', None)
        # for key, value in ret_metrics_class.items():
        #     eval_results.update({
        #         key + '.' + str(name): value[idx] / 100.0
        #         for idx, name in enumerate(class_names)
        #     })
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
    
    def format_results(self, results, out_file, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        out_dir = out_file.split('/')[0]+'/'+'labelTrainIds/'
        mmcv.mkdir_or_exist(out_dir)
        filename = out_file.split('/')[-1].replace('_frame_camera','')
        filename = re.sub(r'.*_R', 'R', filename)
        # basename = osp.splitext(osp.basename(filename))[0]
        png_filename = osp.join(out_dir, filename)
        output = results[0].astype(np.uint8)
        mmcv.imwrite(output, png_filename)
        return results
    # def evaluate_temp(self,
    #              results,
    #              metric='mIoU',
    #              logger=None,
    #              gt_seg_maps=None,
    #              **kwargs):
    #     """Evaluate the dataset.

    #     Args:
    #         results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
    #              results or predict segmentation map for computing evaluation
    #              metric.
    #         metric (str | list[str]): Metrics to be evaluated. 'mIoU',
    #             'mDice' and 'mFscore' are supported.
    #         logger (logging.Logger | None | str): Logger used for printing
    #             related information during evaluation. Default: None.
    #         gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
    #             used in ConcatDataset

    #     Returns:
    #         dict[str, float]: Default metrics.
    #     """
    #     if isinstance(metric, str):
    #         metric = [metric]
    #     allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'Fscore.lane', 'Precision.lane', 'Recall.lane']
    #     if not set(metric).issubset(set(allowed_metrics)):
    #         raise KeyError('metric {} is not supported'.format(metric))

    #     eval_results = {}
    #     # test a list of files
    #     if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
    #             results, str):
    #         if gt_seg_maps is None:
    #             gt_seg_maps = self.get_gt_seg_maps()
    #         num_classes = len(self.CLASSES)
    #         ret_metrics = eval_metrics(
    #             results,
    #             gt_seg_maps,
    #             num_classes,
    #             self.ignore_index,
    #             metric,
    #             label_map=self.label_map,
    #             reduce_zero_label=self.reduce_zero_label)
    #     # test a list of pre_eval_results
    #     else:
    #         ret_metrics = pre_eval_to_metrics(results, metric)

    #     # Because dataset.CLASSES is required for per-eval.
    #     if self.CLASSES is None:
    #         class_names = tuple(range(num_classes))
    #     else:
    #         class_names = self.CLASSES

    #     # summary table
    #     ret_metrics_summary = OrderedDict({
    #         ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
    #         for ret_metric, ret_metric_value in ret_metrics.items()
    #     })

    #     # each class table
    #     ret_metrics.pop('aAcc', None)
    #     ret_metrics_class = OrderedDict({
    #         ret_metric: np.round(ret_metric_value * 100, 2)
    #         for ret_metric, ret_metric_value in ret_metrics.items()
    #     })
    #     ret_metrics_class.update({'Class': class_names})
    #     ret_metrics_class.move_to_end('Class', last=False)

    #     # for logger
    #     class_table_data = PrettyTable()
    #     for key, val in ret_metrics_class.items():
    #         class_table_data.add_column(key, val)

    #     summary_table_data = PrettyTable()
    #     for key, val in ret_metrics_summary.items():
    #         if key == 'aAcc':
    #             summary_table_data.add_column(key, [val])
    #         else:
    #             summary_table_data.add_column('m' + key, [val])

    #     print_log('per class results:', logger)
    #     print_log('\n' + class_table_data.get_string(), logger=logger)
    #     print_log('Summary:', logger)
    #     print_log('\n' + summary_table_data.get_string(), logger=logger)


    #     # summary table
    #     ret_metrics_summary = OrderedDict({
    #         ret_metric: np.nanmean(ret_metric_value)
    #         for ret_metric, ret_metric_value in ret_metrics.items()
    #     })

    #     # each class table
    #     ret_metrics.pop('aAcc', None)
    #     ret_metrics_class = OrderedDict({
    #         ret_metric: ret_metric_value
    #         for ret_metric, ret_metric_value in ret_metrics.items()
    #     })
    #     ret_metrics_class.update({'Class': class_names})
    #     ret_metrics_class.move_to_end('Class', last=False)
    #     # each metric dict
    #     for key, value in ret_metrics_summary.items():
    #         if key == 'aAcc':
    #             eval_results[key] = value
    #         else:
    #             eval_results['m' + key] = value

    #     ret_metrics_class.pop('Class', None)
    #     for key, value in ret_metrics_class.items():
    #         eval_results.update({
    #             key + '.' + str(name): value[idx]
    #             for idx, name in enumerate(class_names)
    #         })

    #     return eval_results
    

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
# # import einops
# from torch.utils.data import DataLoader
# from torch.utils.data import DistributedSampler, RandomSampler
# # from semseg.augmentations_mm import get_train_augmentation

# class DELIVER(Dataset):
#     """
#     num_classes: 25
#     """
#     CLASSES = ["Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation", 
#                 "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail", 
#                 "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck"]

#     PALETTE = torch.tensor([[70, 70, 70],
#             [100, 40, 40],
#             [55, 90, 80],
#             [220, 20, 60],
#             [153, 153, 153],
#             [157, 234, 50],
#             [128, 64, 128],
#             [244, 35, 232],
#             [107, 142, 35],
#             [0, 0, 142],
#             [102, 102, 156],
#             [220, 220, 0],
#             [70, 130, 180],
#             [81, 0, 81],
#             [150, 100, 100],
#             [230, 150, 140],
#             [180, 165, 180],
#             [250, 170, 30],
#             [110, 190, 160],
#             [170, 120, 50],
#             [45, 60, 150],
#             [145, 170, 100],
#             [  0,  0, 230], 
#             [  0, 60, 100],
#             [  0,  0, 70],
#             ])
    
#     def __init__(self, root: str = 'data/DELIVER', split: str = 'train', transform = None, modals = ['img'], case = None) -> None:
#         super().__init__()
#         assert split in ['train', 'val', 'test']
#         self.transform = transform
#         self.n_classes = len(self.CLASSES)
#         self.ignore_label = 255
#         self.modals = modals
#         self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))
#         # --- debug
#         # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
#         # --- split as case
#         if case is not None:
#             assert case in ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'], "Case name not available."
#             _temp_files = [f for f in self.files if case in f]
#             self.files = _temp_files
#         if not self.files:
#             raise Exception(f"No images found in {img_path}")
#         print(f"Found {len(self.files)} {split} {case} images.")

#     def __len__(self) -> int:
#         return len(self.files)
    
#     def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
#         rgb = str(self.files[index])
#         x1 = rgb.replace('/img', '/hha').replace('_rgb', '_depth')
#         x2 = rgb.replace('/img', '/lidar').replace('_rgb', '_lidar')
#         x3 = rgb.replace('/img', '/event').replace('_rgb', '_event')
#         lbl_path = rgb.replace('/img', '/semantic').replace('_rgb', '_semantic')

#         sample = {}
#         sample['img'] = io.read_image(rgb)[:3, ...]
#         H, W = sample['img'].shape[1:]
#         if 'depth' in self.modals:
#             sample['depth'] = self._open_img(x1)
#         if 'lidar' in self.modals:
#             sample['lidar'] = self._open_img(x2)
#         if 'event' in self.modals:
#             eimg = self._open_img(x3)
#             sample['event'] = TF.resize(eimg, (H, W), TF.InterpolationMode.NEAREST)
#         label = io.read_image(lbl_path)[0,...].unsqueeze(0)
#         label[label==255] = 0
#         label -= 1
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