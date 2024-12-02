from typing import Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    ResNetBlock,
    Downsample,
    AttentionBlock,
)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_channels=128,
        embedding_dim=4,
        channel_multipliers=(1, 2, 2, 4),
        num_residual_blocks=(4, 3, 4, 3, 4),
        attention_resolutions=(),
        dropout_probability=0.0,
        use_attention=False,
        residual_conv_kernel_size=3,
        input_conv_kernel_size=3,
    ):
        super().__init__()

        assert len(num_residual_blocks) == len(channel_multipliers) + 1, (
            "The number of num_residual_blocks should be one more than the number of "
            "channel_multipliers."
        )

        # Inital conv layer to get to the hidden dimensionality
        self.conv_in = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=input_conv_kernel_size,
            padding=input_conv_kernel_size // 2,
        )
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity="linear")

        # construct the residual blocks
        self.blocks = nn.ModuleList()
        block_input_channels = hidden_channels
        block_output_channels = hidden_channels
        current_downsample_factor = 1
        for i, cm in enumerate(channel_multipliers):
            block_output_channels = cm * hidden_channels
            # Create the residual blocks

            if block_input_channels != block_output_channels:
                # this is introduced by MagViT
                self.blocks.append(
                    ResNetBlock(
                        input_channels=block_input_channels,
                        output_channels=block_output_channels,
                        kernel_size=residual_conv_kernel_size,
                        dropout=dropout_probability,
                    )
                )

            for _ in range(num_residual_blocks[i]):
                block = ResNetBlock(
                    input_channels=block_output_channels,
                    output_channels=block_output_channels,
                    kernel_size=residual_conv_kernel_size,
                    dropout=dropout_probability,
                )
                self.blocks.append(block)

                if current_downsample_factor in attention_resolutions:
                    attention = AttentionBlock(input_channels=block_output_channels)
                    self.blocks.append(attention)

            # Add the downsampling block at the end, but not the very end.
            if i < len(channel_multipliers) - 1:
                down_block = Downsample(
                    input_channels=block_output_channels,
                    resample_with_conv=True,
                )
                self.blocks.append(down_block)
                current_downsample_factor *= 2

            block_input_channels = block_output_channels

        # Make the middle blocks
        number_of_middle_blocks = num_residual_blocks[-1]
        for i in range(number_of_middle_blocks):
            # insert attention block if needed
            # attention only at the intermediate blocks
            self.blocks.append(
                ResNetBlock(
                    input_channels=block_output_channels,
                    output_channels=block_output_channels,
                    kernel_size=residual_conv_kernel_size,
                    dropout=dropout_probability,
                )
            )
            if use_attention and i < number_of_middle_blocks - 1:
                attention = AttentionBlock(input_channels=block_output_channels)
                self.blocks.append(attention)

        self.fianl_layers = nn.Sequential(
            nn.GroupNorm(32, block_output_channels),
            nn.SiLU(),
            nn.Conv2d(block_output_channels, embedding_dim, 1),
        )

        nn.init.kaiming_normal_(self.fianl_layers[-1].weight, nonlinearity="linear")
        # Output layer is immediately after a silu. Need to account for that in init.
        self.fianl_layers[-1].weight.data *= 1.6761

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the encoder."""
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.fianl_layers(h)
        return h
