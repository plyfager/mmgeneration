_base_ = [
    '../_base_/models/agile_encoder.py', '../_base_/datasets/unconditional_imgs_flip_256x256.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(
    encoder=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999)))

# define dataset
# you must set `samples_per_gpu`
# `samples_per_gpu` and `imgs_root` need to be set.
imgs_root = 'data/imgs_256'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(imgs_root=imgs_root))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['real_imgs','downsample_imgs'],
        interval=100)
]
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# 30000 images in celeba-hq
total_iters = 19000

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)
