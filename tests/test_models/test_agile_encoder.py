from mmgen.apis import init_model
import torch
cfg_path = 'configs/_base_/models/agile_encoder.py'
model = init_model(cfg_path, device='cpu')
src_x = torch.randn(2,3,256,256)
rec_x, logvar , mu = model(src_x)
print(rec_x.shape)
img_gen = rec_x
batch, channel, height, width = img_gen.shape
if height > 256:
    factor = height // 256

    img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                width // factor, factor)
    img_gen = img_gen.mean([3, 5])
id_loss = model.id_loss(src_x, img_gen)
print(id_loss)

### test perceptual loss
per_loss = model.perceptual_loss(img_gen, src_x)
print(per_loss)
### test kl loss
print(logvar.shape, mu.shape)
kl_loss = model.kl_loss(logvar, mu)
print(kl_loss)