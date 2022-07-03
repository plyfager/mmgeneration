import argparse
import math
import os

import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from resizer import Resizer
from torchvision import utils

from mmgen.apis import init_model
from mmgen.datasets.adm_image_datasets import load_data
from mmgen.models.architectures.adm.script_util import (
    add_dict_to_argparser, args_to_dict, create_model_and_diffusion,
    model_and_diffusion_defaults)
from mmgen.utils import get_root_logger


# added
def load_reference(data_dir,
                   batch_size,
                   image_size,
                   class_cond=False,
                   dist=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
        dist=dist)
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs


def main():
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist('pytorch', backend='nccl')
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        # cfg.gpu_ids = range(world_size)

    ckpt = args.checkpoint
    log_name = ckpt.split('.')[0] + '_ilvr_log' + '.txt'
    log_path = os.path.join(args.dirname, log_name)
    os.makedirs(os.path.join(args.dirname, "images"), exist_ok=True)

    logger = get_root_logger(log_file=log_path, file_mode='a')

    logger.info('ILVR Sampling')
    logger.info("creating model...")

    # build models

    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    diffusion = model.diffusion

    if torch.cuda.is_available():
        model.cuda()

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.info("creating resizers...")
    assert math.log(args.down_N, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N),
               int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.info("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        dist=distributed)

    logger.info("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.cuda() for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model, (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=args.range_t)

        for i in range(args.batch_size):
            out_path = os.path.join(
                args.dirname, 'images',
                f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        count += 1
        logger.info(f"created {count * args.batch_size} samples")

    if distributed:
        dist.barrier()
    logger.info("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=8,
        batch_size=4,
        down_N=4,
        range_t=20,
        use_ddim=False,
        base_samples="",
        save_latents=False,
        launcher='none',
        device='cuda',
        dirname='work_dirs/samples')
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='model checkpoint file')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
