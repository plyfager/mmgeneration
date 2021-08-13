from mmgen.apis import init_model
import torch
cfg_path = 'configs/_base_/models/agile_encoder.py'
model = init_model(cfg_path, device='cpu').cuda()
y = model(torch.randn(2,3,256,256).cuda())
print(y.shape)