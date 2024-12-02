# https://github.com/CompVis/zigma/blob/main/utils/torchmetric_fvd.py


from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import adaptive_avg_pool2d

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _MATPLOTLIB_AVAILABLE,
    _TORCH_FIDELITY_AVAILABLE,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from einops import rearrange
import torch.distributed as dist

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["FrechetInceptionDistance.plot"]

if _TORCH_FIDELITY_AVAILABLE:
    from torch_fidelity.feature_extractor_inceptionv3 import (
        FeatureExtractorInceptionV3 as _FeatureExtractorInceptionV3,
    )
    from torch_fidelity.helpers import vassert
    from torch_fidelity.interpolate_compat_tensorflow import (
        interpolate_bilinear_2d_like_tensorflow1x,
    )
else:

    class _FeatureExtractorInceptionV3(Module):  # type: ignore[no-redef]
        pass

    vassert = None
    interpolate_bilinear_2d_like_tensorflow1x = None

    __doctest_skip__ = ["FrechetInceptionDistance", "FrechetInceptionDistance.plot"]
"""
Copy-pasted from Copy-pasted from https://github.com/NVlabs/stylegan2-ada-pytorch
"""

import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid

from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Dict


def open_url(
    url: str,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content_str.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError(
                                "Google Drive download quota exceeded -- please try again later"
                            )

                    match = re.search(
                        r'filename="([^"]*)"',
                        res.headers.get("Content-Disposition", ""),
                    )
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


import torch.nn as nn


class VideoDetector(nn.Module):
    def __init__(
        self,
        detector_url="https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
        detector_kwargs=dict(rescale=False, resize=False, return_features=True),
        device="cuda",
    ):
        nn.Module.__init__(self)

        # Return raw features before the softmax layer.

        with open_url(detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval().to(device)
        self.detector_kwargs = detector_kwargs

    def forward(self, *args):
        return self.detector(*args, **self.detector_kwargs)


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class FrechetVideoDistance(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    inception: Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[int, Module] = 400,
        reset_real_features: bool = True,
        normalize: bool = False,
        device="cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(feature, int):
            num_features = feature
            self.inception = VideoDetector()

        elif isinstance(feature, Module):
            self.inception = feature
            if hasattr(self.inception, "num_features"):
                num_features = self.inception.num_features
            else:
                dummy_image = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8)
                num_features = self.inception(dummy_image).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        mx_num_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def update(self, videos: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        # the input should be B, T, C, H, W
        # the input range should be [0, 255]
        # the input should be 3x224x224
        b, t, c, h, w = videos.shape
        assert b != 1, "Batch should be greater than 1"
        # assert t > 8, "Video length should be greater than 8"
        videos = videos.float()

        if h != 224 or w != 224:
            videos = rearrange(videos, "b t c h w -> (b t) c h w")
            videos = torch.nn.functional.interpolate(
                videos, size=(224, 224), mode="bilinear", align_corners=False
            )
            videos = rearrange(videos, "(b t) c h w -> b t c h w", b=b)

        videos = rearrange(videos, "b t c h w -> b c t h w")

        if t == 8:
            # in prinpcle, we should have t >= 9
            # this is to test some special cases that t = 8
            videos = torch.concat([videos, videos[:, :, -2:-1:, :, :]], dim=2).to(
                videos.device
            )
        features = self.inception(videos)  # .detach().cpu()
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += videos.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += videos.shape[0]

    def _compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(
            0
        )
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(
            0
        )

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions using DDP."""
        # if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
        #     raise RuntimeError(
        #         "More than one sample is required for both the real and fake distributions to compute FID"
        #     )

        # Initialize tensors to hold the reduced sums and counts
        real_features_sum = self.real_features_sum.clone()
        fake_features_sum = self.fake_features_sum.clone()
        real_features_cov_sum = self.real_features_cov_sum.clone()
        fake_features_cov_sum = self.fake_features_cov_sum.clone()
        # real_features_num_samples = torch.tensor(
        #     self.real_features_num_samples, dtype=torch.float32
        # ).to(self.real_features_sum.device)
        # fake_features_num_samples = torch.tensor(
        #     self.fake_features_num_samples, dtype=torch.float32
        # ).to(self.fake_features_sum.device)

        real_features_num_samples = (
            self.real_features_num_samples.clone()
            .detach()
            .to(self.real_features_sum.device)
        )

        fake_features_num_samples = (
            self.fake_features_num_samples.clone()
            .detach()
            .to(self.real_features_sum.device)
        )

        # Reduce sums and covariances across all processes
        # dist.reduce(real_features_sum, dst=0, op=dist.ReduceOp.SUM)
        # dist.reduce(fake_features_sum, dst=0, op=dist.ReduceOp.SUM)
        # dist.reduce(real_features_cov_sum, dst=0, op=dist.ReduceOp.SUM)
        # dist.reduce(fake_features_cov_sum, dst=0, op=dist.ReduceOp.SUM)
        # dist.reduce(real_features_num_samples, dst=0, op=dist.ReduceOp.SUM)
        # dist.reduce(fake_features_num_samples, dst=0, op=dist.ReduceOp.SUM)

        dist.all_reduce(real_features_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(fake_features_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(real_features_cov_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(fake_features_cov_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(real_features_num_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(fake_features_num_samples, op=dist.ReduceOp.SUM)

        # Calculate global means
        mean_real = real_features_sum / real_features_num_samples
        mean_fake = fake_features_sum / fake_features_num_samples

        # Calculate global covariances
        cov_real_num = (
            real_features_cov_sum
            - real_features_num_samples
            * mean_real.unsqueeze(1).mm(mean_real.unsqueeze(0))
        )
        cov_fake_num = (
            fake_features_cov_sum
            - fake_features_num_samples
            * mean_fake.unsqueeze(1).mm(mean_fake.unsqueeze(0))
        )

        cov_real = cov_real_num / (real_features_num_samples - 1)
        cov_fake = cov_fake_num / (fake_features_num_samples - 1)

        # Compute FID score
        fid_score = _compute_fid(mean_real, cov_real, mean_fake, cov_fake).to(
            self.orig_dtype
        )
        return fid_score

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as ``torch.dtype`` or string

        """
        out = super().set_dtype(dst_type)
        if isinstance(out.inception, NoTrainInceptionV3):
            out.inception._dtype = dst_type
        return out

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> metric.update(imgs_dist1, real=True)
            >>> metric.update(imgs_dist2, real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = lambda: torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = lambda: torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(imgs_dist1(), real=True)
            ...     metric.update(imgs_dist2(), real=False)
            ...     values.append(metric.compute())
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


if __name__ == "__main__":
    import numpy as np
    import torch

    if False:
        seed_fake = 1
        seed_real = 2
        num_videos = 128
        video_len = 16
        _fvd = FrechetVideoDistance()
        videos_fake = (
            np.random.RandomState(seed_fake)
            .rand(num_videos, video_len, 224, 224, 3)
            .astype(np.float32)
        )
        videos_real = (
            np.random.RandomState(seed_real)
            .rand(num_videos, video_len, 224, 224, 3)
            .astype(np.float32)
        )
        _fvd.update(torch.tensor(videos_real).to("cuda"), True)
        _fvd.update(torch.tensor(videos_fake).to("cuda"), False)
        print(_fvd.compute())
    elif False:
        seed_fake = 1
        seed_real = 2
        num_videos = 128
        video_len = 16
        _fvd = FrechetVideoDistance()
        videos_fake = torch.rand(
            (num_videos, video_len, 224, 224, 3), dtype=torch.float
        ).to("cuda")
        videos_real = torch.rand(
            (num_videos, video_len, 224, 224, 3), dtype=torch.float
        ).to("cuda")
        _fvd.update(videos_real, True)
        _fvd.update(videos_fake, False)
        print(_fvd.compute())
    elif True:
        seed_fake = 1
        seed_real = 2
        num_videos = 8
        video_len = 50
        frame_size = 224
        _fvd = FrechetVideoDistance()
        print("fvd test")
        videos_fake = torch.zeros(
            (num_videos, video_len, 3, frame_size, frame_size), dtype=torch.float
        ).to("cuda")
        videos_real = torch.ones(
            (num_videos, video_len, 3, frame_size, frame_size), dtype=torch.float
        ).to("cuda")
        _fvd.update(videos_real, True)
        _fvd.update(videos_fake, False)
        print(_fvd.compute())
