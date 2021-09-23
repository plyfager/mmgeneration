encoder_ckpt_path = 'work_dirs/pre-trained/agilegan_encoder_celeba-hq_bs8x2_lr_1e-4_20_cvt.pkl'
encoder_ckpt_prefix = 'encoder'
decoder_ckpt_path = 'work_dirs/pre-trained/agilegan_cartoon_cvt.pkl'
decoder_ckpt_prefix = 'generator_ema'

model = dict(
    type='AgileEncoder',
    encoder=dict(
        type='VAEStyleEncoder',
        num_layers=50,
        pretrained=dict(
            ckpt_path=encoder_ckpt_path, prefix=encoder_ckpt_prefix)),
    decoder=dict(
        type='DualGenerator',
        out_size=1024,
        style_channels=512,
        pretrained=dict(
            ckpt_path=decoder_ckpt_path, prefix=decoder_ckpt_prefix)),
    id_loss=None,
    perceptual_loss=None,
    kl_loss=None)
train_cfg = None
test_cfg = None
