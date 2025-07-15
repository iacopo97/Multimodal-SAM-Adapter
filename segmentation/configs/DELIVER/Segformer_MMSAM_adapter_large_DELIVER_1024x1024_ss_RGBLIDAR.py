# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/DELIVER_MM.py', #DELIVER_MM.py
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40ep.py'
]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
crop_size =(1024,1024)
checkpoint_conv='https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_in21k-pre_3rdparty_in1k-384px_20221219-96f0bb87.pth'
mod_dir_tr='samples/lidar/training',
mod_dir_val='samples/lidar/validation',
mod_dir_test='samples/lidar/test',
mod_suffix='_lidar_front.png',
stride=(640,640)
img_scale = (1042,1042)
modalities_name=['rgb','lidar']
modalities_ch=[3,3]

pretrained = 'pretrained/sam_vit_l_image_encoder_no_neck.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained=pretrained,
    backbone=dict( 
        _delete_=True,
        type='SAMAdapterbimodalMixModNewInTwinConvNEW',
        img_size=crop_size[0],
        modalities_name=modalities_name,
        modalities_ch=modalities_ch,
        init_values=1e-6,
        gamma_init_values=1e-6,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.3,
        drop_multimodal_path=0,
        conv_inplane=48,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True, 
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        global_attn_indexes= [5, 11, 17, 23],
        window_size=14,
        arch='small',
        checkpoint=checkpoint_conv,
        ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=25,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='OhemCrossEntropy')),
    test_cfg=dict(mode='whole_dim',rescale=True, dim=(1024,1024))
)
# dataset settings
mod_norm_cfg =dict(
mean=[0.485, 0.456, 0.406,0,0,0], std=[0.229, 0.224, 0.225,1,1,1], to_rgb=[True,True]
)
# dataset settings
train_pipeline = [
    dict(type='LoadImageandModalities3ch', modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='LoadAnnotations'),    
    dict(type='RandomGaussianBlur', kernel_size=(3,3), p=0.2,modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='Resize_multimodal', img_scale=img_scale, ratio_range=(0.5, 2.0),modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion_multimodal',modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='Normalize_multimodal', **mod_norm_cfg,modalities_name=modalities_name, modalities_ch=modalities_ch, norm_by_max=True),
    dict(type='Pad_multimodal', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageandModalities3ch', modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='Resize_multimodal', img_scale=crop_size, seg_scale=(1024,1024),keep_ratio=True,modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Normalize_multimodal', **mod_norm_cfg,modalities_name=modalities_name, modalities_ch=modalities_ch, norm_by_max=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collectmod', keys=['img'],modalities_name=modalities_name,modalities_ch=modalities_ch),
        ])
]
optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01,#lr=2e-5,weight_decay=0.05 lr=2e-4 geminifusion lr=6e-5 CMNEXT
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))
lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='exp',
                 warmup_iters=10, 
                 warmup_ratio=0.1,
                 power=0.9,
                 min_lr=0.0, by_epoch=True, warmup_by_epoch=True)

data = dict(samples_per_gpu=1,
            train=dict(pipeline=train_pipeline, mod_dir=mod_dir_tr, mod_suffix=mod_suffix, modalities_name=modalities_name, modalities_ch=modalities_ch),
            val=dict(pipeline=test_pipeline,  mod_dir=mod_dir_val, mod_suffix=mod_suffix, modalities_name=modalities_name, modalities_ch=modalities_ch),
            test=dict(pipeline=test_pipeline,  mod_dir=mod_dir_test, mod_suffix=mod_suffix, modalities_name=modalities_name, modalities_ch=modalities_ch))



optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)

runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)
custom_hooks_config=[dict(type='DistSamplerSeedHook'),dict(
        type="EarlyStoppingHook",
        monitor="mIoU",
        patience=30,
        min_delta=0.005)]
evaluation = dict(start=1, interval=1, by_epoch=True, metric='mIoU', save_best='mIoU', resize_dim=(1024,1024), case=['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'])
freeze_backbone = False