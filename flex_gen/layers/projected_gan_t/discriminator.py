# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Projected discriminator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# from torch.nn.utils.spectral_norm import SpectralNorm
from .spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize
from .diffaug import DiffAugment


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if we use GP loss in discriminator, we need to remove spectral norm
        # SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class BatchNormLocal(nn.Module):
    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        virtual_bs: int = 8,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        # BatchNormLocal(channels),
        nn.BatchNorm1d(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int = 0, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9)),
        )

        # if self.c_dim > 0:
        #     # raise NotImplementedError
        #     # self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
        #     self.cls = SpectralConv1d(
        #         channels, cmap_dim, kernel_size=1, padding=0)
        # else:
        #     self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)
        self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:

        # =====
        h = self.main(x)
        out = self.cls(h)

        # if self.c_dim > 0:
        #     cmap = self.cmapper(c).unsqueeze(-1)
        #     out = (out * cmap).sum(1, keepdim=True) * \
        #         (1 / np.sqrt(self.cmap_dim))

        return out


class DINO(torch.nn.Module):
    def __init__(
        self,
        hooks: list[int] = [2, 5, 8, 11],
        hook_patch: bool = True,
        base_model: str = "dinov2_vits14_reg",
    ):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)
        self.hooks = hooks
        self.hook_patch = hook_patch

        # self.dino = torch.hub.load("facebookresearch/dinov2", base_model)

        try:
            self.dino = torch.hub.load(
                os.path.join(
                    os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache")),
                    "hub/facebookresearch_dinov2_main",
                ),
                base_model,
                source="local",
            )
        except Exception as e:
            self.dino = torch.hub.load("facebookresearch/dinov2", base_model)

        self.dino = self.dino.eval().requires_grad_(False)

        self.img_resolution = self.dino.patch_embed.img_size
        self.embed_dim = self.dino.embed_dim
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward_dino(self, x, masks=None):
        ret = []

        x = self.dino.prepare_tokens_with_masks(x, masks)

        if self.hook_patch:
            ret += [x.transpose(1, 2).contiguous()]

        for block_count, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if block_count in self.hooks:
                ret += [x.transpose(1, 2).contiguous()]
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # """input: x in [0, 1]; output: dict of activations"""
        # print("--", self.img_resolution, self.embed_dim)
        x = F.interpolate(x, self.img_resolution, mode="area")
        x = self.norm(x)
        features = self.forward_dino(x)

        return features


class ProjectedDiscriminator(nn.Module):
    # def __init__(self, c_dim: int = 0, diffaug: bool = False, p_crop: float = 0.5):
    def __init__(
        self,
        c_dim: int = 0,
        diffaug: bool = False,
        aug_policy: str = "color,translation,cutout",
        p_crop: float = 0.0,
        base_model: str = "dinov2_vits14_reg",
    ):

        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.aug_policy = aug_policy
        self.p_crop = p_crop

        self.dino = DINO(base_model=base_model).eval()

        heads = []
        for i in range(self.dino.n_hooks):
            heads += ([str(i), DiscHead(self.dino.embed_dim, c_dim)],)
        self.heads = nn.ModuleDict(heads)

        # dino is frozen by default.
        for param in self.dino.parameters():
            param.requires_grad = False
        for head in self.heads.values():
            for param in head.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy=self.aug_policy)

        # TO CHECK, waht if we normalize the image to [0, 1] before passing to DINO
        # -1 ,1 -> 0, 1
        # 0  ,1 -> 0.5, 1?
        # Transform to [0, 1].
        # x = x.add(1).div(2)

        """
        Ablation:   without normalization here, can improve <=0.01 - make sense    
        """

        # Take crops with probablity p_crop if the image is larger.
        if (
            x.size(-1) > self.dino.img_resolution[-1]
            and np.random.random() < self.p_crop
        ):
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward pass through DINO ViT.
        # with torch.no_grad():
        features = self.dino(x)

        # Apply discriminator heads.

        # with torch.autocast(device_type=x.device.type):
        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[int(k)], c).view(x.size(0), -1))
        logits = torch.cat(logits, dim=1)

        return logits


# fake = torch.randn(
#     2,
#     3,
#     128,
#     128,
# ).cuda()
# d = ProjectedDiscriminator().cuda()
# logits = d(fake)
# real_loss = F.relu(torch.ones_like(logits) + logits)
# print(real_loss.shape, logits.mean().shape)
