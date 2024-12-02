import pyarrow as pa
import os
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from PIL import Image
import io
from datasets import Dataset
import torch
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from flex_gen.datasets.transforms import build_transforms
from flex_gen.datasets import build_dataset
from flex_gen.utils.config import name_output_dir
from pathlib import Path
from timm import create_model
from flex_gen import autoencoders
import argparse
import flex_gen.utils.distributed as distributed
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
import tqdm
import numpy as np
import torch.distributed as dist
import random
import torch.nn.utils.rnn as rnn_utils
from data_process.statisitc import OnlineStats, compute_global_stats
from einops import rearrange


def get_image_schema(use_quantized=False):
    if use_quantized:
        image_schema = pa.schema(
            [
                pa.field("label", pa.int32()),
                pa.field("image_codes", pa.list_(pa.list_(pa.int32()))),
            ]
        )
    else:
        image_schema = pa.schema(
            [
                # pa.field("image", pa.list_(pa.list_(pa.list_(pa.float32())))),
                # pa.field("label", pa.float32()),
                pa.field("label", pa.int32()),
                pa.field("image_latent", pa.list_(pa.list_(pa.list_(pa.float32())))),
            ]
        )
    return image_schema


def get_video_schema(use_quantized=False):
    if use_quantized:
        video_schema = pa.schema(
            [
                pa.field("label", pa.int32()),
                pa.field("video_codes", pa.list_(pa.list_(pa.list_(pa.int32())))),
            ]
        )
    else:
        video_schema = pa.schema(
            [
                # pa.field("label", pa.float32()),
                pa.field("label", pa.int32()),
                pa.field(
                    "video_latent", pa.list_(pa.list_(pa.list_(pa.list_(pa.float32()))))
                ),
            ]
        )
    return video_schema


def create_parquet_file(filename, example_table):
    return pq.ParquetWriter(filename, example_table.schema)


def convert_batch_data_to_arrow_table(batch, schema):
    if "label" not in batch:
        batch["label"] = torch.zeros(1)

    df = pd.DataFrame({k: v.numpy().tolist() for k, v in batch.items()})
    table = pa.Table.from_pandas(df, schema=schema)
    return table


def main(configs):
    original_yaml = OmegaConf.load(configs.config_file)
    dataset_config = original_yaml.train_data.dataset
    transform_config = original_yaml.train_data.transforms
    dataset_name = original_yaml.train_data.dataset.get("path")

    data_transform = build_transforms(transform_config, dataset_name=dataset_name)
    dataset = build_dataset(dataset_config, transforms=data_transform)

    tokenzier = create_model(
        model_name=configs.model_name,
        path=configs.model_path,
        pretrained=True,
        map_location=torch.device("cpu"),
    ).cuda()
    tokenzier.eval()
    print(f"Model loaded, model name {configs.model_name}, path {configs.model_path}")

    target_folder = Path(configs.output_dir)
    os.makedirs(target_folder, exist_ok=True)
    print(f"Output folder {target_folder}")

    global_rank = distributed.get_global_rank()
    batch_size = configs.batch_size
    worker = configs.worker

    try:
        try_gvq_group = tokenzier.quantizers.__dict__.get("num_group", 1)
    except:
        try_gvq_group = 1

    stats = [OnlineStats() for i in range(try_gvq_group)]

    if configs.is_debug != 0:
        configs.is_debug = max(configs.is_debug, 5)

    use_quantized = configs.quantized
    image_schema = get_image_schema(use_quantized)
    video_schema = get_video_schema(use_quantized)

    print(f"Start processing data,   BS {batch_size}, Worker {worker}")
    for split in dataset.keys():
        sub_set = dataset[split]
        sampler = (
            DistributedSampler(sub_set) if distributed.get_global_size() > 1 else None
        )
        if sampler is not None:
            sampler.set_epoch(0)
        data_loader = DataLoader(
            sub_set,
            batch_size=batch_size,
            num_workers=worker,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
            # collate_fn=collate_fn,
        )
        filename = target_folder / f"{split}_shard_{global_rank}.parquet"
        parquet_writer = None
        step = 0

        for example in tqdm.tqdm(
            data_loader, disable=not distributed.is_main_process()
        ):
            with torch.no_grad():
                if "image" in example:
                    if use_quantized:
                        codes = tokenzier.tokenize(example["image"].cuda())
                        example["image_codes"] = codes.cpu().int()
                        # stats.update(codes.float())
                        stats[0].update(codes.float())

                    else:
                        latent = tokenzier.encode(example["image"].cuda())["quantized"]
                        example["image_latent"] = latent.cpu()
                        # print(f"========= {example['image_latent'].shape}")
                        # latent is (B, C, H, W)

                        group_latent = rearrange(
                            latent, "b (g d) h w -> b g d h w", g=try_gvq_group
                        )
                        for i in range(try_gvq_group):
                            stats[i].update(group_latent[:, i, :, :, :].contiguous())
                    example.pop("image")
                    arrow_table = convert_batch_data_to_arrow_table(
                        example, image_schema
                    )
                elif "video" in example:
                    assert batch_size == 1
                    # the input should be batch_size = 1
                    # and list of (C, H, W), length of the list is the number of frames
                    concact_video = torch.cat(example["video"], dim=0)

                    if configs.max_video_length != -1:
                        # its optioinal to extract only the first N frames
                        concact_video = concact_video[: configs.max_video_length]

                    res = []

                    if use_quantized:
                        for i in range(0, concact_video.shape[0], 100):
                            res.append(
                                tokenzier.tokenize(
                                    concact_video[i : i + 100].cuda()
                                ).cpu()
                            )
                        example["video_codes"] = (
                            torch.cat(res, dim=0).unsqueeze(0).int()
                        )
                    else:
                        # incase the video is too long, we use mini_batch=100 to avoid OOM
                        for i in range(0, concact_video.shape[0], 100):
                            res.append(
                                tokenzier.encode(concact_video[i : i + 100].cuda())[
                                    "quantized"
                                ].cpu()
                            )
                        example["video_latent"] = torch.cat(res, dim=0).unsqueeze(0)
                    example.pop("video")
                    stats[0].update(example["video_latent"])
                    arrow_table = convert_batch_data_to_arrow_table(
                        example, video_schema
                    )

            if parquet_writer is None:
                parquet_writer = create_parquet_file(filename, arrow_table)
            parquet_writer.write_table(arrow_table)
            step += 1
            if configs.is_debug != 0 and step >= configs.is_debug:
                break

        for group_id in range(try_gvq_group):
            local_stats = stats[group_id].get_stats()
            global_mean, global_var = compute_global_stats(local_stats)
            std = torch.sqrt(global_var)
            if dist.get_rank() == 0:
                print(
                    f"SPLIT: {split}  | group {group_id} |   MEAN {global_mean} | STD {std}"
                )

        dist.barrier()
        parquet_writer.close()


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Pre-Token", add_help=add_help)
    parser.add_argument(
        "--config-file",
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        default=96,
        type=int,
    )
    parser.add_argument(
        "--worker",
        default=20,
        help="Jureca for 64 and booster for 20",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="logs_TEST",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--quantized",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--is_debug",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--max_video_length",
        default=-1,
        type=int,
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    args.output_dir = name_output_dir(args.output_dir)
    distributed.enable(
        overwrite=True, dist_init=True, restrict_print_to_main_process=True
    )
    main(args)
