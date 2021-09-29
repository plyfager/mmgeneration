_base_ = [
    '../_base_/models/agile_encoder.py',
    '../_base_/datasets/unconditional_imgs_flip_256x256.py',
    '../_base_/default_runtime.py'
]

use_ranger = False
model = dict(use_ranger=use_ranger, start_from_mean_latent=False)
optimizer = None if use_ranger else dict(encoder=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999)))
# define dataset
# you must set `samples_per_gpu`
# `samples_per_gpu` and `imgs_root` need to be set.
# train_imgs_root = '/mnt/lustre/share_data/xurui/ffhq/ffhq_imgs/ffhq_256'
train_imgs_root = 'data/imgs_256'
eval_imgs_root = '/mnt/lustre/yangyifei1/dataset/encoder1k'
test_pipeline = [
    dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
    dict(type='Resize', keys=['real_img'], scale=(256, 256)),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(dataset=dict(imgs_root=train_imgs_root)),
    test=dict(dataset=dict(type='UnconditionalImageDataset', imgs_root=eval_imgs_root, pipeline=test_pipeline)))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=10)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['real_imgs', 'downsample_imgs'],
        interval=200)
]
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
# 30000 images in celeba-hq
total_iters = 125000

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)
