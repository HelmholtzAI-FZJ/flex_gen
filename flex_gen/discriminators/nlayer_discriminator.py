# https://github.com/mosaicml/diffusion/blob/main/diffusion/models/autoencoder.py#L403
from .base_discriminator import BaseDiscriminator
from torch import nn, einsum
import torch


class NLayerDiscriminator(BaseDiscriminator):
    """Defines a PatchGAN discriminator.

    Based on code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        num_filters (int): Number of filters in the first layer. Default: `64`.
        num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_filters: int = 64,
        num_layers: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_layers = num_layers

        self.blocks = nn.Sequential()
        input_conv = nn.Conv2d(
            self.input_channels, self.num_filters, kernel_size=4, stride=2, padding=1
        )
        nn.init.kaiming_normal_(input_conv.weight, nonlinearity="linear")
        nonlinearity = nn.LeakyReLU(0.2, True)
        self.blocks.extend([input_conv, nonlinearity])

        num_filters = self.num_filters
        out_filters = self.num_filters
        for n in range(1, self.num_layers):
            out_filters = self.num_filters * min(2**n, 8)
            conv = nn.Conv2d(
                num_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False
            )
            num_filters = out_filters
            # Init these as if a linear layer follows them because batchnorm happens before leaky relu.
            nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
            norm = nn.BatchNorm2d(out_filters)
            nonlinearity = nn.LeakyReLU(0.2, True)
            self.blocks.extend([conv, norm, nonlinearity])
        # Make the output layers
        final_out_filters = self.num_filters * min(2**self.num_layers, 8)
        conv = nn.Conv2d(
            out_filters,
            final_out_filters,
            kernel_size=4,
            stride=1,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
        norm = nn.BatchNorm2d(final_out_filters)
        nonlinearity = nn.LeakyReLU(0.2, True)
        self.blocks.extend([conv, norm, nonlinearity])
        # Output layer
        output_conv = nn.Conv2d(
            final_out_filters, 1, kernel_size=4, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(output_conv.weight, nonlinearity="linear")
        # Should init output layer such that outputs are generally within the linear region of a sigmoid.
        # This likely makes little difference in practice.
        output_conv.weight.data *= 0.1
        self.blocks.append(output_conv)

    def disc_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the discriminator."""
        return self.blocks(x)
