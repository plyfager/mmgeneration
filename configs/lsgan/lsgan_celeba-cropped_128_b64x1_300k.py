_base_ = [
    '../_base_/models/dcgan_128x128.py',
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/default_runtime.py'
]
model = dict(
    discriminator=dict(output_scale=1),
    gan_loss=dict(type='GANLoss', gan_type='lsgan'))
# define dataset
# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=64,
    train=dict(imgs_root='data/celeba-cropped/cropped_images_aligned_png'))

optimizer = dict(
    generator=dict(type='Adam', lr=0.001, betas=(0.5, 0.99)),
    discriminator=dict(
        type='Adam', lr=0.0018823529411764706, betas=(0.5, 0.99)))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000)
]

total_iters = 1000000
# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 128, 128)))
