# {'image_size': 512, 'num_channels': 256, 'num_res_blocks': 2,
#  'channel_mult': '',
#  'learn_sigma': True, 'class_cond': True, 'use_checkpoint': False, 
#  'attention_resolutions': '32,16,8', 'num_heads': 4, 'num_head_channels': 64,
#   'num_heads_upsample': -1, 'use_scale_shift_norm': True, 'dropout': 0.0,
#    'resblock_updown': True, 'use_fp16': False, 'use_new_attention_order': False}

# {'steps': 1000, 'learn_sigma': True, 'noise_schedule': 'linear',
#  'use_kl': False,
#   'predict_xstart': False, 'rescale_timesteps': False, 
#   'rescale_learned_sigmas': False, 'timestep_respacing': '250'}

model = dict(
    type='AdmUNetModel',
    image_size=512,
    in_channels=3,
    model_channels=256,
    out_channels=6,
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    dropout=0.0,
    channel_mult=(1, 2, 3, 4),
    conv_resample=True,
    dims=2,
    num_classes=1000,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_new_attention_order=False)

