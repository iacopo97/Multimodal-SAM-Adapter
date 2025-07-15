dataset_type = 'DELIVER'
data_root = 'data/DELIVER_2/'
modalities_name = ['rgb','depth', 'event', 'lidar']
modalities_ch= [3,3,3,1]

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#crop_size = (896, 896)
crop_size = (640, 640)
img_scale = (1024,1024)
#448,#896
train_pipeline = [
    dict(type='LoadImageandModalities'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageandModalities'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=img_scale,
    #     # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='RandomFlip'),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img']),
    #     ])
]
data = dict(
    samples_per_gpu=2,#2
    workers_per_gpu=2,#2
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='samples/images/training2', #training training2 validation
        # ev_dir='samples/event/training',
        # lid_dir='samples/lidar/training',
        # depth_dir='samples/depth/training',
        mod_dir=['samples/depth/training2','samples/event/training2','samples/lidar/training2'],
        mod_suffix=['_depth_front.png','_event_front.png','_lidar_front.png'],
        # in_dir='samples/events/training', #training training2 validation
        # x_dir='samples/x/training', #training training2 validation
        ann_dir='samples/annotations/training2', #training training2 validation
        pipeline=train_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='samples/images/validation', #validation test2
        img_dir='samples/images/training2', #validation test2
        # ev_dir='samples/event/validation',
        # lid_dir='samples/lidar/validation',
        # depth_dir='samples/depth/validation',
        # mod_dir=['samples/depth/validation','samples/event/validation','samples/lidar/validation'],
        mod_dir=['samples/depth/training2','samples/event/training2','samples/lidar/training2'],
        mod_suffix=['_depth_front.png','_event_front.png','_lidar_front.png'],
        # in_dir='samples/events/validation', #validation test2
        # x_dir='samples/x/validation', #validation test2
        # ann_dir='samples/annotations/validation', #validation test2
        ann_dir='samples/annotations/training2', #validation test2
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='samples/images/training2', #test validation
        # ev_dir='samples/event/test',
        # lid_dir='samples/lidar/test',
        # depth_dir='samples/depth/test',
        mod_dir=['samples/depth/training2','samples/event/training2','samples/lidar/training2'],
        mod_suffix=['_depth_front.png','_event_front.png','_lidar_front.png'],
        # in_dir='samples/events/test', #test validation
        # x_dir='samples/x/test', #test validation
        ann_dir='samples/annotations/training2', #test validation
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch))


