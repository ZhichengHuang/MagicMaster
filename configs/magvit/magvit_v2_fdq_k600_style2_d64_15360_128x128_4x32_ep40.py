default_scope = 'magicmaster'
import os



ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.0002,
    update_buffers=True,
    start_iter=2000)

model = dict(
    type='MagVitVQGAN',
    data_preprocessor=dict(type='DataPreprocessor',output_channel_order='RGB',),
    ema_config=ema_config,
    generator=dict(
        type='MagVitGenerator',
        encoder=dict(type='MagvitV2encoder',
                image_size=128,
                channels=3,
                init_dim=64,
                layers=(
                 ('consecutive_residual', 4),
                 ('spatial_down', 1),
                 ('channel_residual', 1),
                 ('consecutive_residual', 3),
                 ('time_spatial_down', 1),
                 ('consecutive_residual', 4),
                 ('time_spatial_down', 1),
                 ('channel_residual', 1),
                 ('consecutive_residual', 3),
                 ('consecutive_residual', 4),)
                ),
        quantizer=dict(type='FDQ_15360',
                       levels=[8, 8, 5, 4, 4, 3],
                       use_tanh=False,
                       up=1.0,
                       dim=256),
        decoder=dict(type='MagvitV2Adadecoder',
                     image_size=128,
                     channels=3,
                     init_dim=64,
                     layers=(
                         ('consecutive_residual', 3),
                         ('channel_residual', 1),
                         ('condation',1),
                         ('spatial_up', 1),
                         ('consecutive_residual', 4),
                         ('condation',1),
                         ('time_spatial_up', 1),
                         ('consecutive_residual', 3),
                         ('channel_residual', 1),
                         ('condation',1),
                         ('time_spatial_up', 1),
                         ('consecutive_residual', 4),
                         ('condation',1),
                         ('consecutive_residual', 4)
                     ),
    ),),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=128
    ),
    loss_config=dict(
    perceptual_weight=1.0,
    disc_weight=0.75,
    grad_penalty_loss_weight=1.,
    perceptual_loss=dict(type='LPIPS')
    )
)

# dataset settings
dataset_type = 'kinetics600'

# different from mmcls, we adopt the setting used in BigGAN.
# Remove `RandomFlip` augmentation and change `RandomCropLongEdge` to
# `CenterCropLongEdge` to eliminate randomness.
# dataset settings
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=17, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='CenterCropLongEdgeVideo'),
    dict(type='ResizeVideo', scale=(128, 128),keep_ratio=False),
    dict(type='PackVideoInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=17,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='CenterCropLongEdgeVideo'),
    dict(type='ResizeVideo', scale=(128, 128), keep_ratio=False),
    dict(type='PackVideoInputs')
]
data_root = os.path.join(os.getenv("INPUT_PATH", './'), 'data/k600/')
train_dataloader = dict(
    batch_size=4,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_root = data_root,
        prefix='train',
        ann_file="train_new.txt",
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        prefix='val',
        ann_file="val_new.txt",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = val_dataloader

# config for model wrapper
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True)

# config for optim_wrapper_constructor lr=5.4e-5,
#             betas=(0.5, 0.9),
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=5.4e-5, betas=(0.5, 0.9))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=4.32e-4, betas=(0.5, 0.9))))


# config for training
# train_cfg = dict(by_epoch=False, max_iters=500000,val_begin=1, val_interval=10000)
train_cfg = dict(by_epoch=True, max_epochs=40,val_begin=10, val_interval=1)



metrics = [
    dict(
        type='FVD',
        prefix='FVD',
        fake_nums=29739,
        inception_path='./work_dirs/init/FVD/i3d_torchscript.pt',
        # inception_pkl='./work_dirs/init/FVD/i3d_state-capture_mean_cov-full.pkl',
        inception_style='StyleGAN',
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






