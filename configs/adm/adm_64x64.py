# dict(
#     image_size=64,
#     num_channels=192,
#     num_res_blocks=3,
#     channel_mult='',
#     learn_sigma=True,
#     class_cond=True,
#     use_checkpoint=False,
#     attention_resolutions='32,16,8',
#     num_heads=4,
#     num_head_channels=64,
#     num_heads_upsample=-1,
#     use_scale_shift_norm=True,
#     dropout=0.1,
#     resblock_updown=True,
#     use_fp16=True,
#     use_new_attention_order=True)

# dict(
#     steps=1000,
#     learn_sigma=True,
#     noise_schedule='cosine',
#     use_kl=False,
#     predict_xstart=False,
#     rescale_timesteps=False,
#     rescale_learned_sigmas=False,
#     timestep_respacing='250')

model = dict(
    type='AdmUNetModel',
    image_size=64,
    in_channels=3,
    model_channels=192,
    out_channels=6,
    num_res_blocks=3,
    attention_resolutions=(2, 4, 8),
    dropout=0.1,
    channel_mult=(1, 2, 3, 4),
    conv_resample=True,
    dims=2,
    num_classes=1000,
    use_checkpoint=False,
    use_fp16=True,
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_new_attention_order=True)

# image_size,
# in_channels,
# model_channels,
# out_channels,
# num_res_blocks,
# attention_resolutions,
# dropout = 0,
# channel_mult = (1, 2, 4, 8),
# conv_resample = True,
# dims = 2,
# num_classes = None,
# use_checkpoint = False,
# use_fp16 = False,
# num_heads = 1,
# num_head_channels = -1,
# num_heads_upsample = -1,
# use_scale_shift_norm = False,
# resblock_updown = False,
# use_new_attention_order = False
