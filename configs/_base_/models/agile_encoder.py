# ckpt_path = "work_dirs/pre-trained/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth"
ckpt_path = "work_dirs/pre-trained/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth"
prefix='generator'
model = dict(
    type='AgileEncoder',
    encoder=dict(type="VAEStyleEncoder", num_layers=50),
    decoder=dict(type="StyleGANv2Generator",out_size=256,
                 style_channels=512,
                 pretrained=dict(
                     ckpt_path = ckpt_path, 
                     prefix=prefix)),
    loss=None)

train_cfg = None
test_cfg = None
optimizer = dict(
    encoder=dict(type='Adam', lr=0.001, betas=(0.0, 0.999)))
# optimizer = dict(
    # encoder=dict(type='RAdam', lr=0.001, betas=(0.0, 0.999)))
