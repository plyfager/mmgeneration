from mmgen.apis import init_model
import torch
from mmcv.runner import obj_from_dict

def test_agile_transfer():
    cfg_path = 'configs/agilegan/agilegan_transfer_metfaces_bs8x2_lr_1e-4_20e.py'
    model = init_model(cfg_path, device='cpu')
    # prepare data_batch
    data_batch = dict(real_img=torch.randn((2,3,1024,1024)))
    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))
    optimizer = {
            'generator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(model, 'generator').parameters())),
            'discriminator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(model, 'discriminator').parameters()))
        }
    model.train_step(data_batch, optimizer)

def test_agile_transfer_128x128():
    cfg_path = 'configs/agilegan/agilegan_transfer_128x128_metfaces_bs8x2_lr_1e-4_20e.py'
    model = init_model(cfg_path, device='cpu')
    # prepare data_batch
    data_batch = dict(real_img=torch.randn((2,3,128,128)))
    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))
    optimizer = {
            'generator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(model, 'generator').parameters())),
            'discriminator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(model, 'discriminator').parameters()))
        }
    model.train_step(data_batch, optimizer)

    ### training loss ###
    ## TODO: hinge loss
    ## TODO: Modified LPIPS
    ## R1 GP
    ## PPL