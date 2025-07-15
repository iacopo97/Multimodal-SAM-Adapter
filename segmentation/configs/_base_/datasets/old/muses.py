dataset_type = 'MUSES'
data_root = 'data/muses/'
modalities_name = ['rgb', 'event', 'lidar']
modalities_ch= [3,3,3]

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#crop_size = (896, 896)
crop_size = (1024, 1024)
img_scale = (1080,1920)
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
    samples_per_gpu=1,#2
    workers_per_gpu=2,#2
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='frame_camera/train', #training training2 validation
        # ev_dir='samples/event/training',
        # lid_dir='samples/lidar/training',
        # depth_dir='samples/depth/training',_png
        mod_dir=['projected_to_rgb/event_camera/train','projected_to_rgb/lidar/train'],
        mod_suffix=['_event_camera.npz','_lidar.npz'],
        # in_dir='samples/events/training', #training training2 validation
        # x_dir='samples/x/training', #training training2 validation
        ann_dir='gt_semantic/train', #training training2 validation
        pipeline=train_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='samples/images/validation', #validation test2
        img_dir='frame_camera/val', #training training2 validation
        # ev_dir='samples/event/training',
        # lid_dir='samples/lidar/training',
        # depth_dir='samples/depth/training',_png
        mod_dir=['projected_to_rgb/event_camera/val','projected_to_rgb/lidar/val'],
        mod_suffix=['_event_camera.npz','_lidar.npz'],
        # in_dir='samples/events/training', #training training2 validation
        # x_dir='samples/x/training', #training training2 validation
        ann_dir='gt_semantic/val', #training training2 validation
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    test=dict(
        type=dataset_type,
        data_root=data_root,
         img_dir='frame_camera/test', #training training2 validation
        # ev_dir='samples/event/training',
        # lid_dir='samples/lidar/training',
        # depth_dir='samples/depth/training',_png
        mod_dir=['projected_to_rgb/event_camera/test','projected_to_rgb/lidar/test'],
        mod_suffix=['_event_camera.npz','_lidar.npz'],
        # in_dir='samples/events/training', #training training2 validation
        # x_dir='samples/x/training', #training training2 validation
        ann_dir='gt_semantic/test', #training training2 validation
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch))


