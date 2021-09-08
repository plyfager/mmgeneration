import argparse
import os
import sys

import mmcv
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_uncoditional_model  # isort:skip  # noqa
# yapf: enable

import normal_image 
import torchvision.transforms as transforms
import cv2
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Generation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('source_path', help='source image path')
    parser.add_argument(
        '--show-input',
        type=bool,
        default=True,
        help='Whether show input')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CUDA device id')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/agile_result.png',
        help='path to save image transfer result')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=None, device=args.device)
    
    # Image Normalized
    # TODO: Put this part into test_pipeline in the future 
    img = cv2.imread(args.source_path)
    assert img is not None
    normal = normal_image.Normal_Image()
    img = normal.run(img)
    
    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = img.convert("RGB")
    transformed_image = T(img)
    
    transformed_image = transformed_image.unsqueeze(0).to(args.device).float()
    # rgb 012
    input_image = transformed_image[:, [2, 1, 0]]
    
    results,_,_ = model(input_image, test_mode=True)
    
    results = (results[:, [2, 1, 0]] + 1.) / 2.
    # show input
    if args.show_input:
        import torch.nn.functional as F
        down_results = F.interpolate(results, (256, 256))
        transformed_image = (transformed_image + 1.0) / 2
        results = torch.cat([transformed_image, down_results], dim=0)
    # save images
    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(results, args.save_path)


if __name__ == '__main__':
    main()
