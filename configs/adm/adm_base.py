diffusion_cfg = dict(
    learn_sigma=False,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
)

model = dict(
    type='UNetModel',
    image_size=64,
    num_channels=128,
    num_res_blocks=2,
    num_heads=4,
    num_heads_upsample=-1,
    num_head_channels=-1,
    attention_resolutions="16,8",
    channel_mult="",
    dropout=0.0,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
)
