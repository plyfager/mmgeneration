import argparse
import os
import sys

import cv2
import mmcv
import normal_image
import torch
import torchvision.transforms as transforms
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_uncoditional_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Generation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('source_path', help='source image path')
    parser.add_argument(
        '--show-input', type=bool, default=False, help='Whether show input')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CUDA device id')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/agile_result.png',
        help='path to save image transfer result')
    parser.add_argument(
        '--out-channel-order',
        type=str,
        default='bgr',
        choices=['bgr', 'rgb'],
        help='channel order of output')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(args.config, checkpoint=None, device=args.device)
    model.eval()
    # Image Normalized
    # TODO: Put this part into test_pipeline in the future
    img = cv2.imread(args.source_path)
    assert img is not None
    normal = normal_image.Normal_Image()
    img = normal.run(img)

    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = img.convert('RGB')
    transformed_image = T(img)

    transformed_image = transformed_image.unsqueeze(0).to(args.device).float()
    # rgb 012
    input_image = transformed_image
    # input_image = transformed_image[:, [2, 1, 0]]

    results, _, _ = model(input_image, test_mode=True)

    imageA = results[0]
    imageB = results[1]

    if args.out_channel_order == 'bgr':
        imageB = (imageB[:, [2, 1, 0]] + 1.) / 2.
    else:
        imageB = (imageB + 1.) / 2.

    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(imageB, args.save_path, normalize=True)


if __name__ == '__main__':
    main()
