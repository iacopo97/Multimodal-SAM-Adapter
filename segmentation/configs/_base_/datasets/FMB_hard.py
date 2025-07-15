dataset_type = 'FMB_hard'
data_root = 'data/FMB/'
modalities_name = ['rgb','ther']
modalities_ch= [3,1]
crop_size = (800, 800)
img_scale = (800,600)
train_pipeline = [
    dict(type='LoadImageandModalities'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
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
        img_dir='train/Visible', 
        mod_dir=['train/Infrared'],
        mod_suffix=['.png'],
        ann_dir='train/Label', 
        split='train',
        pipeline=train_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/Visible',
        mod_dir=['test/Infrared'],
        mod_suffix=['.png'],
        ann_dir='test/Label', 
        split='val',
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/Visible', 
        mod_dir=['test/Infrared'],
        mod_suffix=['.png'],
        ann_dir='test/Label', 
        split='test',
        pipeline=test_pipeline,
        modalities_name=modalities_name,
        modalities_ch=modalities_ch))



