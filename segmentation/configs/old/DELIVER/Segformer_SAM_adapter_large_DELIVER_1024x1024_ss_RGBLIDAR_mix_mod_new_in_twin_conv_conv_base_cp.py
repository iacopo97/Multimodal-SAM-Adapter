# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    # /media/data4/sora/projects/ViT-Adapter/segmentation/configs/_base_/models/
    # '../_base_/models/mask2former_beit_DELIVER.py',
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/DELIVER_MM.py', #DELIVER_MM.py
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40ep.py'
]
log_config = dict(
    interval=50,
    hooks=[
        # dict(type='WandbLoggerHook',
        # init_kwargs={
        # 'entity': "iacopo-curti",
        # 'project': "Sina_lane_detection",
        # 'name': "mask2former_beit_adapter_large_640_40k_pipeline2"},
        # by_epoch=False),
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='MMSegWandbHook',
        # init_kwargs={
        # 'entity': "iacopo-curti",
        # 'project': "Synthia_lane_detection",
        # 'name': "mask2former_SAM-ADAPTER_large_640_80k_pipeline2_stoch_depth_at_end_and_inj_ext_v2"},
        # log_checkpoint=True,
        # log_checkpoint_metadata=False,
        # by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# crop_size = (448, 448)#480,480 
# stride = (320,320)
# crop_size =(960,960)
# crop_size =(960,960)
crop_size =(1024,1024)
# crop_size =(896,896)
# stride=(368,200)
# checkpoint_conv='https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext/convnext-small_32xb128_in1k-384px.py
#https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt
# checkpoint_conv='https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt'
# checkpoint_conv='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'
checkpoint_conv='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'
mod_dir_tr='samples/lidar/training',
mod_dir_val='samples/lidar/validation',
mod_dir_test='samples/lidar/test',
mod_suffix='_lidar_front.png',
stride=(640,640)
img_scale = (1042,1042)
modalities_name=['rgb','lidar']
# modalities_ch=[3,1]
modalities_ch=[3,3]
# modalities_name=['rgb']
# modalities_ch=[3]

# box_crop=(500/2464, 1965/2464, 510/2048, 1540/2048)
# pretrained = 'https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth'
# pretrained = 'pretrained/beitt_b_large_patch16_224_pt22k_ft22k.pth'
pretrained = 'pretrained/sam_vit_l_image_encoder_no_neck.pth'#/media/data2/icurti/projects/sina/AI_server_client/ViTAdapter/segmentation/pretrained/sam_vit_b_01ec64.pth
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    # type='EncoderDecoder',
    pretrained=pretrained,
    # backbone=dict(
    #     type='BEiTAdapter',
    #     img_size=crop_size[0],
    #     patch_size=16,
    #     embed_dim=1024,
    #     depth=24,
    #     num_heads=16,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     use_abs_pos_emb=False,
    #     use_rel_pos_bias=True,
    #     init_values=1e-6,
    #     drop_path_rate=0.3,
    #     conv_inplane=64,
    #     n_points=4,
    #     deform_num_heads=16,
    #     cffn_ratio=0.25,
    #     deform_ratio=0.5,
    #     with_cp=True, # set with_cp=True to save memory
    #     interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
    # ),
    backbone=dict( 
        _delete_=True,
        type='SAMAdapterbimodalMixModNewInTwinConvNEWwithcp',
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
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        # window_attn=[False] * 24,
        # window_size=[None] * 24,
        global_attn_indexes= [5, 11, 17, 23],
        window_size=14,
        arch='base',
        checkpoint=checkpoint_conv,
        ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        # feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        num_classes=25,
        norm_cfg=norm_cfg,
        align_corners=False,
        # decoder_params=dict(embed_dim=768),
        # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        loss_decode=dict(type='OhemCrossEntropy')),
    test_cfg=dict(mode='whole_dim',rescale=True, dim=(1024,1024))
)
# dataset settings
mod_norm_cfg =dict(
    # mean=[0.485, 0.456, 0.406,0], std=[0.229, 0.224, 0.225,1], to_rgb=[True,False]
    # mean=[0.485*255, 0.456*255, 0.406*255, 0.485*255, 0.456*255, 0.406*255,0.485*255, 0.456*255, 0.406*255,0.406*255], std=[0.229*255, 0.224*255, 0.225*255,  0.229*255, 0.224*255, 0.225*255,0.229*255, 0.224*255, 0.225*255,0.225*255], to_rgb=[True,True,True,False]
mean=[0.485, 0.456, 0.406,0,0,0], std=[0.229, 0.224, 0.225,1,1,1], to_rgb=[True,True]
)
# dataset settings
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadImageandModalities3ch', modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='LoadAnnotations'),    
    dict(type='RandomGaussianBlur', kernel_size=(3,3), p=0.2,modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='Resize_multimodal', img_scale=img_scale, ratio_range=(0.5, 2.0),modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion_multimodal',modalities_name=modalities_name, modalities_ch=modalities_ch),
    # dict(type='PhotoMetricDistortion_multimodal', modalities_name=modalities_name, modalities_ch=modalities_ch),
    # dict(type='Normalize_multimodal_deliver', **mod_norm_cfg, modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='Normalize_multimodal', **mod_norm_cfg,modalities_name=modalities_name, modalities_ch=modalities_ch, norm_by_max=True),
    # dict(type='Pad_multimodal', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='Pad_multimodal', size=crop_size, pad_val=0, seg_pad_val=255),
    # dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_semantic_seg'])# 'gt_masks', 'gt_labels'
]

test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadImageandModalities3ch', modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(type='Resize_multimodal', img_scale=crop_size, seg_scale=(1024,1024),keep_ratio=True,modalities_name=modalities_name, modalities_ch=modalities_ch),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=img_scale,
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], single scale test strategy used in the paper cmnext
        flip=False,
        transforms=[
            # dict(type='Normalize_multimodal_deliver',  modalities_name=modalities_name, modalities_ch=modalities_ch,  **norm_cfg),
            dict(type='Normalize_multimodal', **mod_norm_cfg,modalities_name=modalities_name, modalities_ch=modalities_ch, norm_by_max=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collectmod', keys=['img'],modalities_name=modalities_name,modalities_ch=modalities_ch),
            # dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01,#lr=2e-5,weight_decay=0.05 lr=2e-4 geminifusion lr=6e-5 CMNEXT
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))
lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='exp',#'linear'
                 warmup_iters=10,#1500,
                 warmup_ratio=0.1,#1e-6
                 power=0.9,#1.0,
                 min_lr=0.0, by_epoch=True, warmup_by_epoch=True)
# lr_config = dict(_delete_=True,
#                  policy='poly',
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)
# optimizer = dict(_delete_=True, type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05,
#                  constructor='LayerDecayOptimizerConstructor',
#                  paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))
# lr_config = dict(_delete_=True,
#                  policy='poly',
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)
data = dict(samples_per_gpu=1,
            # workers_per_gpu=8,
            train=dict(pipeline=train_pipeline, mod_dir=mod_dir_tr, mod_suffix=mod_suffix, modalities_name=modalities_name, modalities_ch=modalities_ch),
            val=dict(pipeline=test_pipeline,  mod_dir=mod_dir_val, mod_suffix=mod_suffix, modalities_name=modalities_name, modalities_ch=modalities_ch),
            test=dict(pipeline=test_pipeline,  mod_dir=mod_dir_test, mod_suffix=mod_suffix, modalities_name=modalities_name, modalities_ch=modalities_ch))


# OPTIMIZER:
#   NAME          : adamw           # optimizer name
#   LR            : 0.00006         # initial learning rate used in optimizer
#   WEIGHT_DECAY  : 0.01            # decay rate used in optimizer

# SCHEDULER:
#   NAME          : warmuppolylr    # scheduler name
#   POWER         : 0.9             # scheduler power
#   WARMUP        : 10              # warmup epochs used in scheduler
#   WARMUP_RATIO  : 0.1             # warmup ratio
  
# optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=8) #maybe 8,16
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4) #maybe 8,16
# optimizer_config = dict(type="GradientCumulativeFp16OptimizerHook", cumulative_iters=4) #maybe 8,16
# runner = dict(type='IterBasedRunner')

runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)
# #evaluation = dict(interval=40000, metric='mIoU', save_best='mIoU')
# # evaluation = dict(interval=2000, metric='mFscore', save_best='auto')
evaluation = dict(start=1, interval=1, by_epoch=True, metric='mIoU', save_best='mIoU', resize_dim=(1024,1024), case=['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'])#['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'])#, show=True), out_dir="/media/data4/sora/projects/ViT-Adapter/segmentation/training_DELIVER_dataset_RGB_SAMADAPTER_1024x1024/test")#by_epoch=True, _real_train _deliver_pipeline
# find_unused_parameters = Falsecurlallv
freeze_backbone = False