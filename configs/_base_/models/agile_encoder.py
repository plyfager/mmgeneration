ckpt_path = "work_dirs/pre-trained/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth"
facenet_path="work_dirs/pre-trained/model_ir_se50.pth"
# ckpt_path = "work_dirs/pre-trained/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth"
prefix='generator'
model = dict(
    type='AgileEncoder',
    encoder=dict(type="VAEStyleEncoder", num_layers=50),
    decoder=dict(type="StyleGANv2Generator",out_size=1024,
                 style_channels=512,
                 pretrained=dict(
                     ckpt_path = ckpt_path, 
                     prefix=prefix)),
    # id_loss=dict(type="IDLoss",model_path=facenet_path,loss_weight=0.), 
    id_loss=None, 
    perceptual_loss=None,
    kl_loss=None)

train_cfg = None
test_cfg = None
optimizer = dict(
    encoder=dict(type='Adam', lr=0.001, betas=(0.0, 0.999)))
# optimizer = dict(
    # encoder=dict(type='RAdam', lr=0.001, betas=(0.0, 0.999)))
