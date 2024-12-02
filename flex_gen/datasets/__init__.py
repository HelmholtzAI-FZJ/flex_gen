from datasets import load_dataset
import torch
import numpy as np
import torch
from .transforms import build_transforms
import datasets
import sys
from functools import partial


def build_dataset(dataset_config, transforms=None):
    # _old_fun = datasets.DatasetBuilder.download_and_prepare
    # # max_size = sys.maxsize
    # max_size = 10 * (1024**3)  # 5GB

    # def _new_fun(*args, max_shard_size=max_size, **kwargs):
    #     return _old_fun(*args, max_shard_size=max_shard_size, **kwargs)

    # datasets.DatasetBuilder.download_and_prepare = _new_fun

    dataset = load_dataset(**dataset_config)
    if transforms is not None:
        dataset.set_transform(transforms)
    return dataset


def build_distributed_dataloader(dataset, collate_fn=None, dataloader_config=None):
    dataloader_config.pop("shuffle", None)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset),
        collate_fn=collate_fn,
        pin_memory=True,
        **dataloader_config,
    )


def build_dataloader(
    dataset, collate_fn=None, dataloader_config=None, use_distributed_sampler=False
):
    if dataloader_config is None:
        dataloader_config = {}

    if use_distributed_sampler:
        return partial(
            build_distributed_dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            dataloader_config=dataloader_config,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        pin_memory=True,
        **dataloader_config,
    )
    return dataloader


def build_collate_fn(collate_fn_config=None):
    if collate_fn_config is None:
        return None
    from .collate_func import video_data_collate_fn

    return partial(video_data_collate_fn, **collate_fn_config)
