default_scope = 'magicmaster'
import os

ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.0002,
    update_buffers=True,
    start_iter=20000)

model = dict(
    type='VQGAN',
    data_preprocessor=dict(type='DataPreprocessor',output_channel_order='RGB',),
    ema_config=ema_config,
    generator=dict(
        type='VQGenerator',
        encoder=dict(type='VQGANEncoder',),
        post_encoder=dict(type='ConvConnector',in_channels=256, out_channels=256),
        quantizer=dict(type='FSQ',
                       levels=[8,8,8,5,5,5],
                       dim=256,
                       ),
        pre_decoder=dict(type='ConvConnector',in_channels=256, out_channels=256),
        decoder=dict(type='VQGANDecoder')),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=256
    ),
    loss_config=dict(
    perceptual_weight=1.0,
    disc_weight=0.75,
    perceptual_loss=dict(type='LPIPS')
    )
)

# dataset settings
dataset_type = 'ImageNetCls'

# different from mmcls, we adopt the setting used in BigGAN.
# Remove `RandomFlip` augmentation and change `RandomCropLongEdge` to
# `CenterCropLongEdge` to eliminate randomness.
# dataset settings
train_pipeline = [
    dict(type='LoadImageNetFromFile'),
    dict(type='CenterCropLongEdge'),
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageNetFromFile'),
    dict(type='CenterCropLongEdge'),
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(type='PackInputs')
]
data_root = os.path.join(os.getenv("INPUT_PATH", './'), 'data/ImageNet/')
train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        data_root = data_root,
        ann_file="ImageNet_train.json",
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="ImageNet_val.json",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = val_dataloader

# config for model wrapper
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)

# config for optim_wrapper_constructor lr=5.4e-5,
#             betas=(0.5, 0.9),
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=5.4e-5, betas=(0.5, 0.9))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=4.32e-4, betas=(0.5, 0.9))))


# config for training
# train_cfg = dict(by_epoch=False, max_iters=500000,val_begin=1, val_interval=10000)
train_cfg = dict(by_epoch=True, max_epochs=40,val_begin=1, val_interval=1)



metrics = [
    dict(
        type='TransFID',
        prefix='FID-50k',
        fake_nums=50000,
        # inception_path='./work_dirs/init/FID/inception-2015-12-05.pt',
        inception_pkl='./work_dirs/init/FID/inception_state-capture_mean_cov-full-torch.pkl',
        inception_style='Pytorch',
        sample_model='ema'),
    # dict(
    #     type='IS',
    #     prefix='IS-50k',
    #     fake_nums=50000,
    #     inception_style='StyleGAN',
    #     sample_model='ema')
]

# config for val
val_cfg = dict(type='MultiValLoop')
val_evaluator = dict(type='Evaluator',metrics=metrics)

# config for test
test_cfg = dict(type='MultiTestLoop')
test_evaluator = dict(type='Evaluator',metrics=metrics)


# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = True


# configure for default hooks
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=20,
        less_keys=['FID-Full-50k/fid', 'FID-50k/fid', 'swd/avg'],
        greater_keys=['IS-50k/is', 'ms-ssim/avg'],
        save_optimizer=True))

custom_hooks = [
    # dict(
    #     type='VisualizationHook',
    #     interval=5000,
    #     fixed_input=True,
    #     vis_kwargs_list=[
    #         dict(
    #             type='Translation',  # 在训练集可视化结果
    #             name='trans'),  #  保存`trans`字段的图像
    #         dict(
    #             type='Translationval',  # 在验证集可视化结果
    #             name='trans_val'),  #  保存`trans_val`字段的图像
    #     ])
    # dict(
    #     type='VisualizationHook',
    #     interval=5000,
    #     fixed_input=True,
    #     vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

# config for environment
env_cfg = dict(
    # whether to enable cudnn benchmark.
    cudnn_benchmark=True,
    # set multi process parameters.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters.
    dist_cfg=dict(backend='nccl'))

# set log level
log_level = 'INFO'
log_processor = dict(type='LogProcessor', by_epoch=False)

# env settings
dist_params = dict(backend='nccl')
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# set visualizer
vis_backends = [dict(type='VisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)






