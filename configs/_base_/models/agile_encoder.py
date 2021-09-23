encoder_pretrain_path = 'work_dirs/pre-trained/model_ir_se50.pth'
ckpt_path = 'work_dirs/pre-trained/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth'
facenet_path = 'work_dirs/pre-trained/model_ir_se50.pth'
# ckpt_path = "work_dirs/pre-trained/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth"
prefix = 'generator_ema'
model = dict(
    type='AgileEncoder',
    encoder=dict(type='VAEStyleEncoder', num_layers=50, 
                 pretrained=dict(ckpt_path=encoder_pretrain_path, prefix='')),
    decoder=dict(
        type='StyleGANv2Generator',
        out_size=1024,
        style_channels=512,
        pretrained=dict(ckpt_path=ckpt_path, prefix=prefix)),
    id_loss=dict(type='IDLoss', model_path=facenet_path, loss_weight=0.1),
    perceptual_loss=dict(
        type='PerceptualLoss',
        vgg_type='vgg16',
        layer_weights={
            '4': 1.,
            '9': 1.,
            '16': 1.,
            '23': 1.,
            '30': 1.,
        },
        perceptual_weight=0.8,
        pretrained=('torchvision://vgg16')),
    kl_loss=dict(type='KLloss', loss_weight=5e-4))

train_cfg = None
test_cfg = None
optimizer = dict(encoder=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999)))
# optimizer = dict(
# encoder=dict(type='RAdam', lr=0.001, betas=(0.0, 0.999)))
