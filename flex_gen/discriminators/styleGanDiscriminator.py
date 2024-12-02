from .base_discriminator import BaseDiscriminator
from torch import nn, einsum
import torch
from kornia.filters import filter3d

from einops import rearrange


# discriminator with anti-aliased downsampling (blurpool Zhang et al.)
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x, space_only=False, time_only=False):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum("i, j -> i j", f, f)
            f = rearrange(f, "... -> 1 1 ...")
        elif time_only:
            f = rearrange(f, "f -> 1 f 1 1")
        else:
            f = einsum("i, j, k -> i j k", f, f, f)
            f = rearrange(f, "... -> 1 ...")

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, "b c h w -> b c 1 h w")

        out = filter3d(x, f, normalized=True)

        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")

        return out


class ResBlockDown(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            Blur(),
            nn.AvgPool2d(2),
            nn.Conv2d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
        )
        self.skip = nn.Sequential(
            Blur(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_dim, out_dim, 1),
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class StyleGanDiscriminator(BaseDiscriminator):
    def __init__(
        self, in_channels=3, dim=128, channels_multiplier=(2, 4, 4, 4, 4), **kwargs
    ):
        # for magvit
        #  self, in_channels=3, dim=128, channels_multiplier=(2, 4, 4, 4, 4), **kwargs

        super().__init__(**kwargs)

        blocks = [
            nn.Conv2d(in_channels, dim, 3, padding=1),
            nn.LeakyReLU(),
        ]

        dim_in = dim
        for mult in channels_multiplier:
            dim_out = dim * mult
            blocks.append(ResBlockDown(dim_in, dim_out))
            dim_in = dim_out

        dim = dim_in
        blocks += [
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
        ]
        self.blocks = nn.Sequential(*blocks)

    def disc_forward(self, x):
        return self.blocks(x)
