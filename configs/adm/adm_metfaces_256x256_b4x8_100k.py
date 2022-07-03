_base_ = [
    './adm_base.py', '../_base_/datasets/adm_256x256.py',
    '../_base_/default_runtime.py'
]

# define ADM model
attention_resolutions = "16"
class_cond = False
NUM_CLASSES = 1000

diffusion_steps = 1000
dropout = 0.0
image_size = 256
learn_sigma = True
noise_schedule = 'linear'
num_channels = 128
num_head_channels = 64
num_res_blocks = 1
resblock_updown = True
use_fp16 = False
use_scale_shift_norm = True
timestep_respacing = "100"
model_path = 'models/ffhq_10m.pt'
base_samples = 'ref_imgs/face'
down_N = 32
range_t = 20
save_dir = 'output'
channel_mult = (1, 1, 2, 2, 4, 4)

attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(image_size // int(res))

diffusion_cfg = dict(
    steps=diffusion_steps,
    learn_sigma=learn_sigma,
    noise_schedule=noise_schedule,
    # use_kl=use_kl,
    # predict_xstart=predict_xstart,
    # rescale_timesteps=rescale_timesteps,
    # rescale_learned_sigmas=rescale_learned_sigmas,
    timestep_respacing=timestep_respacing,
)
image_size = 64
model = dict(
    type='ADM',
    image_size=image_size,
    in_channels=3,
    model_channels=num_channels,
    out_channels=(3 if not learn_sigma else 6),
    num_res_blocks=num_res_blocks,
    attention_resolutions=tuple(attention_ds),
    dropout=dropout,
    channel_mult=channel_mult,
    num_classes=(NUM_CLASSES if class_cond else None),
    use_checkpoint=False,
    use_fp16=use_fp16,
    num_heads=4,
    num_head_channels=num_head_channels,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=resblock_updown,
    use_new_attention_order=False,
    diffusion_cfg=diffusion_cfg)

train_cfg = dict(use_ema=True)
test_cfg = dict()

lr = 1e-4,

# # optimizer = dict(type='AdamW', lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
# optimizer = dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))
imgs_root = './data/diffusion_datasets/metface'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        resolution=image_size,
        data_dir=imgs_root,  # set by user
    ),
    val=dict(data_dir=imgs_root))

# define optimizer
optimizer = dict(model=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))

# learning policy
total_iters = 800002
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=0, interval=1)

custom_hooks = [
    dict(
        type='VisDMSamples',
        image_size=64,
        batch_size=1,
        output_dir='training_samples',
        interval=100),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('model_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.99),
        priority='VERY_HIGH')
]

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
