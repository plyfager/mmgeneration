encoder_ckpt_path = 'work_dirs/ckpt/agile-encoder/iter_37000.pth'
encoder_ckpt_prefix = 'encoder'
decoder_ckpt_path = 'work_dirs/ckpt/agile-transfer/iter_1600.pth'
decoder_ckpt_prefix = 'generator'

model = dict(
    type='AgileEncoder',
    encoder=dict(
        type='VAEStyleEncoder',
        num_layers=50,
        pretrained=dict(
            ckpt_path=encoder_ckpt_path, prefix=encoder_ckpt_prefix)),
    decoder=dict(
        type='StyleGANv2Generator',
        out_size=1024,
        style_channels=512,
        pretrained=dict(
            ckpt_path=decoder_ckpt_path, prefix=decoder_ckpt_prefix)),
    id_loss=None,
    perceptual_loss=None,
    kl_loss=None)
train_cfg = None
test_cfg = None
