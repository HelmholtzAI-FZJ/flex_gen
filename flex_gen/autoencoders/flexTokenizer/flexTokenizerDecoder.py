from typing import Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    ResNetBlock,
    Upsample,
    AttentionBlock,
    DepthToSpace2DWithConv,
    AdaptiveGroupNorm2D,
)


class Decoder(nn.Module):
    def __init__(
        self,
        output_channels=3,
        hidden_channels=128,
        embedding_dim=4,
        channel_multipliers=(1, 2, 2, 4),
        num_residual_blocks=(3, 4, 3, 4, 4),
        attention_resolutions=(),
        dropout_probability=0.0,
        use_depth_to_space=False,
        use_attention=False,
        use_adaptive_norm=False,
        residual_conv_kernel_size=3,
        output_conv_kernel_size=3,
    ):
        super().__init__()

        assert len(num_residual_blocks) == len(channel_multipliers) + 1, (
            "The number of num_residual_blocks should be one more than the number of "
            "channel_multipliers."
        )

        channels = hidden_channels * channel_multipliers[-1]
        # Inital conv layer to get to the hidden dimensionality
        self.conv_in = nn.Conv2d(
            embedding_dim,
            channels,
            kernel_size=residual_conv_kernel_size,
            padding=residual_conv_kernel_size // 2,
        )
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity="linear")

        # construct the residual blocks
        self.blocks = nn.ModuleList()

        number_of_middle_blocks = num_residual_blocks[-1]
        num_residual_blocks = num_residual_blocks[:-1]
        for i in range(number_of_middle_blocks):
            # insert attention block if needed
            # attention only at the intermediate blocks
            self.blocks.append(
                ResNetBlock(
                    input_channels=channels,
                    output_channels=channels,
                    kernel_size=residual_conv_kernel_size,
                    dropout=dropout_probability,
                )
            )
            if use_attention and i < number_of_middle_blocks - 1:
                attention = AttentionBlock(input_channels=channels)
                self.blocks.append(attention)

        if use_adaptive_norm:
            self.blocks.append(
                AdaptiveGroupNorm2D(
                    num_groups=32,
                    num_channels=channels,
                    cond_channels=embedding_dim,
                )
            )

        # construct the residual blocks
        block_input_channels = channels
        current_downsample_factor = 2 ** (len(channel_multipliers) - 1)

        for i, cm in enumerate(channel_multipliers[::-1]):
            block_out_channels = hidden_channels * cm
            if block_input_channels != block_out_channels:
                # this is introduced by MagViT
                self.blocks.append(
                    ResNetBlock(
                        input_channels=block_input_channels,
                        output_channels=block_out_channels,
                        kernel_size=residual_conv_kernel_size,
                        dropout=dropout_probability,
                    )
                )

            for _ in range(num_residual_blocks[::-1][i]):
                block = ResNetBlock(
                    input_channels=block_out_channels,
                    output_channels=block_out_channels,
                    kernel_size=residual_conv_kernel_size,
                    dropout=dropout_probability,
                )
                self.blocks.append(block)

                if current_downsample_factor in attention_resolutions:
                    attention = AttentionBlock(input_channels=block_out_channels)
                    self.blocks.append(attention)
            # Add the upsampling block at the end, but not the very end.
            if i < len(channel_multipliers) - 1:
                if use_depth_to_space:
                    upsample = DepthToSpace2DWithConv(block_out_channels, 2, 2)
                else:
                    upsample = Upsample(
                        input_channels=block_out_channels,
                        resample_with_conv=True,
                    )
                self.blocks.append(upsample)

                if use_adaptive_norm:
                    self.blocks.append(
                        AdaptiveGroupNorm2D(
                            num_groups=32,
                            num_channels=block_out_channels,
                            cond_channels=embedding_dim,
                        )
                    )
                current_downsample_factor /= 2

            block_input_channels = block_out_channels

        self.final_layers = nn.Sequential(
            nn.GroupNorm(32, block_out_channels),
            nn.SiLU(),
            nn.Conv2d(
                block_out_channels,
                output_channels,
                output_conv_kernel_size,
                padding=output_conv_kernel_size // 2,
            ),
        )

        nn.init.kaiming_normal_(self.final_layers[-1].weight, nonlinearity="linear")
        self.final_layers[-1].weight.data *= 1.6761

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        copy_x = x.clone()
        x = self.conv_in(x)
        for block in self.blocks:
            if isinstance(block, AdaptiveGroupNorm2D):
                x = block(x, copy_x)
            else:
                x = block(x)
        x = self.final_layers(x)
        return x
