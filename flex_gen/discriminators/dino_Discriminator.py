# https://github.com/mosaicml/diffusion/blob/main/diffusion/models/autoencoder.py#L403
from .base_discriminator import BaseDiscriminator
from torch import nn, einsum
import torch
from torch.nn.functional import normalize
from flex_gen.layers.projected_gan_t.discriminator import ProjectedDiscriminator


class DinoDiscriminator(BaseDiscriminator):
    """Defines a PatchGAN discriminator.

    Based on code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        num_filters (int): Number of filters in the first layer. Default: `64`.
        num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(
        self,
        c_dim: int = 0,
        diffaug: bool = True,
        p_crop: float = 1.0,
        base_model: str = "dinov2_vits14_reg",
        aug_policy: str = "color,translation,cutout",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ProjectedDiscriminator = ProjectedDiscriminator(
            c_dim=c_dim,
            diffaug=diffaug,
            p_crop=p_crop,
            base_model=base_model,
            aug_policy=aug_policy,
        )
        """
        This dino-discriminator CAN-NOT work with Gradient Penalty.  
        - Not fixed yet.
        - Although it is not recommended to use this discriminator with GP    
        """

    def disc_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the discriminator."""
        return self.ProjectedDiscriminator(x)
