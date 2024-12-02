from timm.data import create_transform
import torch
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from flex_gen.datasets.DATASET_CONFIG import DATASET_CONFIGS


def fix_transforms_config(transforms_config, dataset_name=None):
    if dataset_name is not None:
        dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        if "ucf101" in dataset_name or "faceforensics" in dataset_name:
            image_key = "video"
            label_key = "label"
        else:
            print(
                f" Cannot find dataset {dataset_name},  USING DEFAULT DATASET KEYS - 'image' and 'label'."
            )
            image_key = "image"
            label_key = "label"
    else:
        image_key, label_key = DATASET_CONFIGS[dataset_name]["data_keys"]

    if transforms_config is None:
        transforms_config = {}

    if len(transforms_config) == 0:
        transforms_config = None
    else:
        if "input_size" not in transforms_config:
            if dataset_name is None or dataset_name not in DATASET_CONFIGS:
                transforms_config["input_size"] = 32
            else:
                transforms_config["input_size"] = DATASET_CONFIGS[dataset_name][
                    "image_size"
                ]

        if "mean" not in transforms_config:
            if dataset_name is None or dataset_name not in DATASET_CONFIGS:
                transforms_config["mean"] = 0.5
            else:
                transforms_config["mean"] = DATASET_CONFIGS[dataset_name]["mean"]

        if isinstance(transforms_config["mean"], int) or isinstance(
            transforms_config["mean"], float
        ):
            transforms_config["mean"] = [transforms_config["mean"]] * 3

        if "std" not in transforms_config:
            if dataset_name is None or dataset_name not in DATASET_CONFIGS:
                transforms_config["std"] = 1.0
            else:
                transforms_config["std"] = DATASET_CONFIGS[dataset_name]["std"]

        if isinstance(transforms_config["std"], int) or isinstance(
            transforms_config["std"], float
        ):
            transforms_config["std"] = [transforms_config["std"]] * 3

    return image_key, label_key, transforms_config


def build_transforms(transforms_config, dataset_name=None):
    """Builds the transforms for a dataset."""
    dataset_name = dataset_name.split("/")[-1].lower()

    image_key, label_key, transforms_config = fix_transforms_config(
        transforms_config, dataset_name
    )

    # we should not pop the mean and std
    # transforms_config.pop("std", None)
    # transforms_config.pop("mean", None)
    def convert_maybe_pillow_to_tensor(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return torch.Tensor(image)

    if transforms_config is None:
        inet_transforms = lambda x: x
    else:
        inet_transforms = create_transform(**transforms_config)

    def transforms_func(examples):
        # ret["image"] = [
        #     inet_transforms(image.convert("RGB")) for image in examples[image_key]
        # ]
        # if label_key in examples:
        #     ret["label"] = [
        #         torch.tensor(label, dtype=torch.long)
        #         for label in examples[label_key]
        #     ]
        for k, v in examples.items():
            if k == image_key:
                if image_key == "video":
                    examples[k] = [
                        [
                            inet_transforms(convert_maybe_pillow_to_tensor(image))
                            for image in video
                        ]
                        for video in v
                    ]
                else:
                    examples[k] = [
                        inet_transforms(convert_maybe_pillow_to_tensor(image))
                        for image in v
                    ]
            elif "video_latent" in k:
                examples[k] = [
                    torch.tensor(v_latent, dtype=torch.float32) for v_latent in v
                ]
            elif k == label_key and k in examples:
                examples[k] = torch.tensor(v, dtype=torch.long)
            else:
                examples[k] = convert_maybe_pillow_to_tensor(v)
        return examples

    return transforms_func
