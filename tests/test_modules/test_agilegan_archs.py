import torch

from mmgen.models import StyleGANv2Generator
from mmgen.models.architectures.agilegan import VAEStyleEncoder

if __name__ == '__main__':
    model = VAEStyleEncoder(50)
    image = torch.randn((2, 3, 256, 256))
    code, _, _ = model(image)
    code = code.permute(1, 0, 2)
    code = code.unbind(dim=0)
    g = StyleGANv2Generator(
        1024,
        512,
        pretrained=dict(
            ckpt_path=
            './work_dirs/pre-trained/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth',
            prefix='generator'))
    image = g(code, input_is_latent=True, truncation=0.5)
    image = image[:, [2, 1, 0], ...]
    from torchvision.utils import save_image
    save_image(image, 'work_dirs/samples/rec.png', normalize=True)
