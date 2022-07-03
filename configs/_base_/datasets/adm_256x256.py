dataset_type = 'ADMImageDataset'

# `samples_per_gpu` and `imgs_root` need to be set.
data = dict(
    samples_per_gpu=None,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        resolution=256,
        data_dir=None,  # set by user
    ),
    val=dict(type=dataset_type, resolution=256, data_dir=None))
