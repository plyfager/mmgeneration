__base__ = ['./adm_base.py']

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

model = dict(
    type='UNetModel',
    image_size=256,
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
train_cfg = dict()
test_cfg = dict()

data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root='./data/ffhq/ffhq_imgs/ffhq_256')),
    val=dict(imgs_root='./data/ffhq/ffhq_imgs/ffhq_256'))

# define optimizer
# optimizer = dict(
#     generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
#     discriminator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
