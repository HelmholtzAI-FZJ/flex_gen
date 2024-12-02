"""
For AutoEncoder model, we use the following metrics:
- Mean Squared Error (MSE)
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- FDD (Fréchet Inception Distance with DINO)
- sFID (sFréchet Inception Distance)
- IS (Inception Score)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Peak Structural Similarity Index)
- SSIM (Structural Similarity Index)



For image generation model, we use the following metrics:
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- FDD (Fréchet Inception Distance with DINO)

For video generation model, we use the following metrics:
- Frame-FID (Fréchet Inception Distance for Video)
- FVD (Fréchet Video Distance)
"""

import torch.distributed as dist
from einops import rearrange
import torch
from torchmetrics import MeanSquaredError

# from torchmetrics.image.kid import KernelInceptionDistance as KID
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM

from .torchmetric_fid import FrechetInceptionDistance as FID
from .torchmetric_fdd import FrechetDinovDistance as FDD
from .torchmetric_fvd import FrechetVideoDistance as FVD
from .torchmetric_sfid import sFrechetInceptionDistance as SFID

from functools import partial

# from .torchmetric_prdc import PRDC

ALL_METRICS_DICT = {
    "mse": MeanSquaredError,
    "fid": FID,
    # "kid": partial(KID, subset_size=2), # we dont have DDP version of KID yet
    "is": InceptionScore,
    "sfid": SFID,
    "fdd": FDD,
    "lpip": LPIPS,
    "lpips": LPIPS,
    "psnr": partial(PSNR, data_range=1.0),
    "ssim": partial(SSIM, data_range=1.0),
    # "frame_fid": FrechetInceptionDistance,
    "fvd": FVD,
}

_image_based_metrics = [
    "mse",
    "ssim",
    "psnr",
    "lpips",
    "is",
]

_features_based_metrics = ["fid", "kid", "sfid", "fdd", "frame_fid", "fvd"]

from flex_gen.datasets.DATASET_CONFIG import query_mean_std_from_dataset_name
from flex_gen.utils.utils import rescale_image_tensor


def monkey_patch_compute(compute_foo, metric_name):
    def wrapper(*args, **kwargs):
        try_get_results = compute_foo(*args, **kwargs)
        if isinstance(try_get_results, tuple) or isinstance(try_get_results, list):
            if len(try_get_results) == 1:
                return {"metric_name": try_get_results}
            elif len(try_get_results) == 2:
                return {
                    f"{metric_name}_mean": try_get_results[0],
                    f"{metric_name}_std": try_get_results[1],
                }
            else:
                raise ValueError(
                    f"Unexpected results from {metric_name}, {try_get_results}"
                )
        else:
            return {metric_name: try_get_results}

    return wrapper


def monkey_patch_update(update_foo, metric_name):
    # ! TODO
    # check if update works well for all metrics in bf/fp16
    def wrapper(real, fake, *args, **kwargs):
        # the orignal image should be in the range of [0, 1]
        device = real.device if real is not None else fake.device
        autocast_dtype = torch.bfloat16 if metric_name != "ssim" else torch.float32
        real = torch.clamp(real, 0, 1) if real is not None else None
        fake = torch.clamp(fake, 0, 1) if fake is not None else None
        with torch.autocast(device.type, dtype=autocast_dtype):
            if "lpip" in metric_name:
                update_foo(
                    torch.clamp(2 * real - 1, -1, 1), torch.clamp(2 * fake - 1, -1, 1)
                )
            elif "is" in metric_name:
                update_foo((fake * 255).type(torch.uint8))
            elif metric_name in _image_based_metrics:
                update_foo(fake, real)
            elif metric_name in _features_based_metrics:
                if real is not None:
                    real = (real * 255).type(torch.uint8)
                    update_foo(real, real=True)
                if fake is not None:
                    fake = (fake * 255).type(torch.uint8)
                    update_foo(fake, real=False)
            else:
                raise ValueError(f"Unknown metric {metric_name}")

    return wrapper


def get_metrics_cls_by_name(metric_name):
    metric_obj = ALL_METRICS_DICT[metric_name]()
    metric_obj.compute = monkey_patch_compute(metric_obj.compute, metric_name)
    metric_obj.update = monkey_patch_update(metric_obj.update, metric_name)
    return metric_obj


class Metrics:
    need_prepare = True

    def __init__(
        self,
        metrics_list=[],
        dataset_name: str = None,
        device="cuda:0",
    ):
        self.device = device
        self.metrics = {
            metric_name: get_metrics_cls_by_name(metric_name)
            for metric_name in metrics_list
        }

        self.norm_mean, self.norm_std = query_mean_std_from_dataset_name(dataset_name)

    def prepare_model(self):
        if self.device == "cpu" or self.need_prepare is False:
            self.need_prepare = False
            return
        for model_key in self.metrics:
            self.metrics[model_key].to(self.device)
        self.need_prepare = False

    def off_load(self):
        for model_key in self.features_based_metrics:
            self.features_based_metrics[model_key].to("cpu")
        for model_key in self.image_based_metrics:
            self.image_based_metrics[model_key].to("cpu")
        self.need_prepare = True

    def reset(self):
        for model_key in self.metrics:
            self.metrics[model_key].reset()

    def update(self, real=None, fake=None):
        self.prepare_model()
        if real is not None:
            real = rescale_image_tensor(real, self.norm_mean, self.norm_std)
        if fake is not None:
            fake = rescale_image_tensor(fake, self.norm_mean, self.norm_std)
        # after scaling, the image should be in the range of [0, 1]

        for metric_name in self.metrics:
            self.metrics[metric_name].update(real=real, fake=fake)

    def compute(self, off_load=False):
        self.prepare_model()
        results = {}
        for model_key in self.metrics:
            results.update(self.metrics[model_key].compute())

        if off_load:
            self.off_load()
        return results

    def compute_and_reduce(self, off_load=False):
        results = self.compute(off_load)

        # Prepare dictionaries to store the sums and sum of squares
        mean_values = {}
        sum_of_squares = {}
        keys_need_to_take_care_of_std = []

        for key, value in results.items():
            if "_std" in key:
                continue  # Skip std entries for now
            if "_mean" in key:
                base_key = key.replace("_mean", "")
                mean_values[base_key] = value
                std_key = base_key + "_std"
                if std_key in results:
                    std_value = results[std_key]
                    sum_of_squares[base_key] = value**2 + std_value**2
                    keys_need_to_take_care_of_std.append(base_key)
                else:
                    sum_of_squares[base_key] = value**2
            else:
                mean_values[key] = value
                sum_of_squares[key] = value**2

        # Convert dictionaries to tensors
        keys = list(mean_values.keys())
        means_tensor = torch.tensor(
            [mean_values[key] for key in keys], dtype=torch.float32, device="cuda"
        )
        sum_of_squares_tensor = torch.tensor(
            [sum_of_squares[key] for key in keys], dtype=torch.float32, device="cuda"
        )

        # Reduce the tensors across all devices
        dist.reduce(means_tensor, dst=0, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        dist.reduce(
            sum_of_squares_tensor, dst=0, op=dist.ReduceOp.SUM, group=dist.group.WORLD
        )

        if dist.get_rank() == 0:
            # Compute the global mean
            world_size = dist.get_world_size()
            global_means = means_tensor / world_size

            # Compute the global variance and std
            global_variances = (sum_of_squares_tensor / world_size) - (global_means**2)
            global_stds = torch.sqrt(global_variances)

            # Convert tensors back to dictionary format
            average_results = {}
            for i, key in enumerate(keys):
                if key in keys_need_to_take_care_of_std:
                    average_results[key + "_mean"] = global_means[i].item()
                    average_results[key + "_std"] = global_stds[i].item()
                else:
                    average_results[key] = global_means[i].item()

            return average_results
        else:
            return {}
