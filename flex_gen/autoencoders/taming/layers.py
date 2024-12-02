# https://github.com/mosaicml/diffusion/blob/main/diffusion/models/autoencoder.py#L134
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.checkpoint import checkpoint
from typing import Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers  # type: ignore
except:
    pass

_T = TypeVar("_T", bound=nn.Module)


def zero_module(module: _T) -> _T:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ClippedAttnProcessor2_0:
    """Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).

    Modified from https://github.com/huggingface/diffusers/blob/v0.21.0-release/src/diffusers/models/attention_processor.py#L977 to
    allow clipping QKV values.

    Args:
        clip_val (float, defaults to 6.0): Amount to clip query, key, and value by.
    """

    def __init__(self, clip_val=6.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.clip_val = clip_val

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        else:
            channel, height, width = None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = query.clamp(min=-self.clip_val, max=self.clip_val)
        key = key.clamp(min=-self.clip_val, max=self.clip_val)
        value = value.clamp(min=-self.clip_val, max=self.clip_val)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ClippedXFormersAttnProcessor:
    """Processor for implementing memory efficient attention using xFormers.

    Modified from https://github.com/huggingface/diffusers/blob/v0.21.0-release/src/diffusers/models/attention_processor.py#L888 to
    allow clipping QKV values.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
        clip_val (float, defaults to 6.0): Amount to clip query, key, and value by.
    """

    def __init__(self, clip_val=6.0, attention_op=None):
        self.attention_op = attention_op
        self.clip_val = clip_val

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        else:
            channel, height, width = None, None, None

        batch_size, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = query.clamp(min=-self.clip_val, max=self.clip_val)
        key = key.clamp(min=-self.clip_val, max=self.clip_val)
        value = value.clamp(min=-self.clip_val, max=self.clip_val)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            assert channel
            assert height
            assert width
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ResNetBlock(nn.Module):
    """Basic ResNet block.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        use_conv_shortcut (bool): Whether to use a conv on the shortcut. Default: `False`.
        dropout (float): Dropout probability. Defaults to 0.0.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: Optional[int] = None,
        use_conv_shortcut: bool = False,
        dropout_probability: float = 0.0,
        zero_init_last: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = (
            output_channels if output_channels is not None else input_channels
        )
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.zero_init_last = zero_init_last

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=self.input_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
        # Output layer is immediately after a silu. Need to account for that in init.
        self.conv1.weight.data *= 1.6761
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=self.output_channels, eps=1e-6, affine=True
        )
        self.dropout = nn.Dropout2d(p=self.dropout_probability)
        self.conv2 = nn.Conv2d(
            self.output_channels,
            self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Optionally use a conv on the shortcut, but only if the input and output channels are different
        if self.input_channels != self.output_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    self.input_channels,
                    self.output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.conv_shortcut = nn.Conv2d(
                    self.input_channels,
                    self.output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            nn.init.kaiming_normal_(self.conv_shortcut.weight, nonlinearity="linear")
        else:
            self.conv_shortcut = nn.Identity()

        # Init the final conv layer parameters to zero if desired. Otherwise, kaiming uniform
        if self.zero_init_last:
            self.residual_scale = 1.0
            self.conv2 = zero_module(self.conv2)
        else:
            self.residual_scale = 0.70711
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")
            # Output layer is immediately after a silu. Need to account for that in init.
            self.conv2.weight.data *= 1.6761 * self.residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the residual block."""
        shortcut = self.residual_scale * self.conv_shortcut(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + shortcut


class AttentionLayer(nn.Module):
    """Basic single headed attention layer for use on tensors with HW dimensions.

    Args:
        input_channels (int): Number of input channels.
        dropout (float): Dropout probability. Defaults to 0.0.
    """

    def __init__(self, input_channels: int, dropout_probability: float = 0.0):
        super().__init__()
        self.input_channels = input_channels
        self.dropout_probability = dropout_probability
        # Normalization layer. Here we're using groupnorm to be consistent with the original implementation.
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=self.input_channels, eps=1e-6, affine=True
        )
        # Conv layer to transform the input into q, k, and v
        self.qkv_conv = nn.Conv2d(
            self.input_channels,
            3 * self.input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Init the qkv conv weights
        nn.init.kaiming_normal_(self.qkv_conv.weight, nonlinearity="linear")
        # Conv layer to project to the output.
        self.proj_conv = nn.Conv2d(
            self.input_channels, self.input_channels, kernel_size=1, stride=1, padding=0
        )
        nn.init.kaiming_normal_(self.proj_conv.weight, nonlinearity="linear")

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor for attention."""
        # x is (B, C, H, W), need it to be (B, H*W, C) for attention
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]).contiguous()
        return x

    def _reshape_from_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reshape the input tensor from attention."""
        # x is (B, H*W, C), need it to be (B, C, H, W) for conv
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the attention layer."""
        # Need to remember H, W to get back to it
        H, W = x.shape[2:]
        h = self.norm(x)
        # Get q, k, and v
        qkv = self.qkv_conv(h)
        qkv = self._reshape_for_attention(qkv)
        q, k, v = torch.split(qkv, self.input_channels, dim=2)
        # Use torch's built in attention function
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_probability)
        # Reshape back into an image style tensor
        h = self._reshape_from_attention(h, H, W)
        # Project to the output
        h = self.proj_conv(h)
        return x + h


class Downsample(nn.Module):
    """Downsampling layer that downsamples by a factor of 2.

    Args:
        input_channels (int): Number of input channels.
        resample_with_conv (bool): Whether to use a conv for downsampling.
    """

    def __init__(self, input_channels: int, resample_with_conv: bool):
        super().__init__()
        self.input_channels = input_channels
        self.resample_with_conv = resample_with_conv
        if self.resample_with_conv:
            self.conv = nn.Conv2d(
                self.input_channels,
                self.input_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample_with_conv:
            # Need to do asymmetric padding to ensure the correct pixels are used in the downsampling conv
            # and ensure the correct output size is generated for even sizes.
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """Upsampling layer that upsamples by a factor of 2.

    Args:
        input_channels (int): Number of input channels.
        resample_with_conv (bool): Whether to use a conv for upsampling.
    """

    def __init__(self, input_channels: int, resample_with_conv: bool):
        super().__init__()
        self.input_channels = input_channels
        self.resample_with_conv = resample_with_conv
        if self.resample_with_conv:
            self.conv = nn.Conv2d(
                self.input_channels,
                self.input_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest", antialias=False)
        if self.resample_with_conv:
            x = self.conv(x)
        return x


class GradientScalingLayer(nn.Module):
    """Layer that scales the gradient by a multiplicative constant.

    By default, this constant is 1.0, so this layer acts as an identity function.

    To use, one must also register the backward hook:
    scaling_layer = GradientScalingLayer()
    scaling_layer.register_full_backward_hook(scaling_layer.backward_hook)

    And then set the scale via
    scaling_layer.set_scale(scale)
    """

    def __init__(self):
        super().__init__()
        self.scale: float = 1.0

    def set_scale(self, scale: float):
        self.scale = scale

    def forward(self, x):
        return x

    def backward_hook(self, module, grad_input, grad_output):
        return (self.scale * grad_input[0],)


class Encoder(nn.Module):
    """Encoder module for an autoencoder.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the latent channels. Default: `True`.
        channel_multipliers (Tuple[int, ...]): Multipliers for the number of channels in each block. Default: `(1, 2, 4, 8)`.
        num_residual_blocks (int): Number of residual blocks in each block. Default: `4`.
        use_conv_shortcut (bool): Whether to use a conv for the shortcut. Default: `False`.
        dropout (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a conv for downsampling. Default: `True`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
        use_attention (bool): Whether to use attention layers. Default: `True`.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 4,
        double_latent_channels: bool = True,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_residual_blocks: int = 4,
        use_conv_shortcut: bool = False,
        dropout_probability: float = 0.0,
        resample_with_conv: bool = True,
        zero_init_last: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.double_latent_channels = double_latent_channels

        self.hidden_channels = hidden_channels
        self.channel_multipliers = channel_multipliers
        self.num_residual_blocks = num_residual_blocks
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last
        self.use_attention = use_attention

        # Inital conv layer to get to the hidden dimensionality
        self.conv_in = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        )
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity="linear")

        # construct the residual blocks
        self.blocks = nn.ModuleList()
        block_input_channels = self.hidden_channels
        block_output_channels = self.hidden_channels
        for i, cm in enumerate(self.channel_multipliers):
            block_output_channels = cm * self.hidden_channels
            # Create the residual blocks
            for _ in range(self.num_residual_blocks):
                block = ResNetBlock(
                    input_channels=block_input_channels,
                    output_channels=block_output_channels,
                    use_conv_shortcut=use_conv_shortcut,
                    dropout_probability=dropout_probability,
                    zero_init_last=zero_init_last,
                )
                self.blocks.append(block)
                block_input_channels = block_output_channels
            # Add the downsampling block at the end, but not the very end.
            if i < len(self.channel_multipliers) - 1:
                down_block = Downsample(
                    input_channels=block_output_channels,
                    resample_with_conv=self.resample_with_conv,
                )
                self.blocks.append(down_block)
        # Make the middle blocks
        middle_block_1 = ResNetBlock(
            input_channels=block_output_channels,
            output_channels=block_output_channels,
            use_conv_shortcut=use_conv_shortcut,
            dropout_probability=dropout_probability,
            zero_init_last=zero_init_last,
        )
        self.blocks.append(middle_block_1)

        if self.use_attention:
            attention = AttentionLayer(input_channels=block_output_channels)
            self.blocks.append(attention)

        middle_block_2 = ResNetBlock(
            input_channels=block_output_channels,
            output_channels=block_output_channels,
            use_conv_shortcut=use_conv_shortcut,
            dropout_probability=dropout_probability,
            zero_init_last=zero_init_last,
        )
        self.blocks.append(middle_block_2)

        # Make the final layers for the output
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_output_channels, eps=1e-6, affine=True
        )
        output_channels = (
            2 * self.latent_channels
            if self.double_latent_channels
            else self.latent_channels
        )
        self.conv_out = nn.Conv2d(
            block_output_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity="linear")
        # Output layer is immediately after a silu. Need to account for that in init.
        self.conv_out.weight.data *= 1.6761

        if (
            "USE_GRADIENT_CHECKPOINTING" in os.environ
            and int(os.environ["USE_GRADIENT_CHECKPOINTING"]) == 1
        ):
            self.use_gradient_checkpointing = True
        else:
            self.use_gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the encoder."""
        h = self.conv_in(x)
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(block, h)
            else:
                h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """Decoder module for an autoencoder.

    Args:
        latent_channels (int): Number of latent channels. Default: `4`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        channel_multipliers (Tuple[int, ...]): Multipliers for the number of channels in each block. Default: `(1, 2, 4, 8)`.
        num_residual_blocks (int): Number of residual blocks in each block. Default: `4`.
        use_conv_shortcut (bool): Whether to use a conv for the shortcut. Default: `False`.
        dropout (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a conv for upsampling. Default: `True`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
        use_attention (bool): Whether to use attention layers. Default: `True`.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        output_channels: int = 3,
        hidden_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_residual_blocks: int = 4,
        use_conv_shortcut=False,
        dropout_probability: float = 0.0,
        resample_with_conv: bool = True,
        zero_init_last: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.channel_multipliers = channel_multipliers
        self.num_residual_blocks = num_residual_blocks
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last
        self.use_attention = use_attention

        # Input conv layer to get to the hidden dimensionality
        channels = self.hidden_channels * self.channel_multipliers[-1]
        self.conv_in = nn.Conv2d(
            self.latent_channels, channels, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity="linear")

        # Make the middle blocks
        self.blocks = nn.ModuleList()
        middle_block_1 = ResNetBlock(
            input_channels=channels,
            output_channels=channels,
            use_conv_shortcut=use_conv_shortcut,
            dropout_probability=dropout_probability,
            zero_init_last=zero_init_last,
        )
        self.blocks.append(middle_block_1)

        if self.use_attention:
            attention = AttentionLayer(input_channels=channels)
            self.blocks.append(attention)

        middle_block_2 = ResNetBlock(
            input_channels=channels,
            output_channels=channels,
            use_conv_shortcut=use_conv_shortcut,
            dropout_probability=dropout_probability,
            zero_init_last=zero_init_last,
        )
        self.blocks.append(middle_block_2)

        # construct the residual blocks
        block_channels = channels
        for i, cm in enumerate(self.channel_multipliers[::-1]):
            block_channels = self.hidden_channels * cm
            for _ in range(self.num_residual_blocks + 1):  # Why the +1?
                block = ResNetBlock(
                    input_channels=channels,
                    output_channels=block_channels,
                    use_conv_shortcut=use_conv_shortcut,
                    dropout_probability=dropout_probability,
                    zero_init_last=zero_init_last,
                )
                self.blocks.append(block)
                channels = block_channels
            # Add the upsampling block at the end, but not the very end.
            if i < len(self.channel_multipliers) - 1:
                upsample = Upsample(
                    input_channels=block_channels,
                    resample_with_conv=self.resample_with_conv,
                )
                self.blocks.append(upsample)
        # Make the final layers for the output
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_channels, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_channels, self.output_channels, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity="linear")
        # Output layer is immediately after a silu. Need to account for that in init.
        # Also want the output variance to mimic images with pixel values uniformly distributed in [-1, 1].
        # These two effects essentially cancel out.

        if (
            "USE_GRADIENT_CHECKPOINTING" in os.environ
            and int(os.environ["USE_GRADIENT_CHECKPOINTING"]) == 1
        ):
            self.use_gradient_checkpointing = True
        else:
            self.use_gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the decoder."""
        h = self.conv_in(x)
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(block, h)
            else:
                h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class GaussianDistribution:
    """Gaussian distribution parameterized with mean and log variance."""

    def __init__(self, mean: torch.Tensor, log_var: torch.Tensor):
        self.mean = mean
        self.log_var = log_var
        self.var = torch.exp(log_var)
        self.std = torch.exp(0.5 * log_var)

    def __getitem__(self, key):
        if key == "latent_dist":
            return self
        elif key == "mean":
            return self.mean
        elif key == "log_var":
            return self.log_var
        else:
            raise KeyError(key)

    @property
    def latent_dist(self):
        return self

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        return self.mean + self.std * torch.randn_like(self.mean)
