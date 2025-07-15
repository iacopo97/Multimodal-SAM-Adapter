dataset_type = 'DELIVER'
data_root = 'data/DELIVER/'
modalities_name = ['rgb','depth', 'event', 'lidar']
modalities_ch= [3,3,3,1]
img_scale = (1024,1024)
train_pipeline = [
    dict(type='LoadImageandModalities'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(1024,1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=(1024,1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageandModalities'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='samples/images/training', 
        mod_dir=['samples/depth/training','samples/event/training','samples/lidar/training'],
        mod_suffix=['_depth_front.png','_event_front.png','_lidar_front.png'],
        ann_dir='samples/annotations/training',
        pipeline=train_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='samples/images/validation',
        mod_dir=['samples/depth/validation','samples/event/validation','samples/lidar/validation'],
        mod_suffix=['_depth_front.png','_event_front.png','_lidar_front.png'],
        ann_dir='samples/annotations/validation', 
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='samples/images/test', 
        mod_dir=['samples/depth/test','samples/event/test','samples/lidar/test'],
        mod_suffix=['_depth_front.png','_event_front.png','_lidar_front.png'],
        ann_dir='samples/annotations/test',
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch))


