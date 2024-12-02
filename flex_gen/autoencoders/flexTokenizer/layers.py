import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from einops import rearrange


class ResNetBlock(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        activation=nn.SiLU(),
        dropout=0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, input_channels),
            activation,
            nn.Conv2d(
                input_channels, output_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(32, output_channels),
            activation,
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                output_channels, output_channels, kernel_size, padding=kernel_size // 2
            ),
        )
        if input_channels != output_channels:
            self.skip = nn.Conv2d(input_channels, output_channels, 1)
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")
        else:
            self.skip = nn.Identity()

        nn.init.kaiming_normal_(self.net[2].weight, nonlinearity="linear")
        self.net[2].weight.data *= 1.6761

        nn.init.kaiming_normal_(self.net[6].weight, nonlinearity="linear")
        self.net[6].weight.data *= 1.6761 * 0.70711

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Downsample(nn.Module):
    """Downsampling layer that downsamples by a factor of 2.

    Args:
        input_channels (int): Number of input channels.
        resample_with_conv (bool): Whether to use a conv for downsampling.
    """

    def __init__(self, input_channels: int, resample_with_conv: bool, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.resample_with_conv = resample_with_conv
        if self.resample_with_conv:
            self.conv = nn.Conv2d(
                self.input_channels,
                self.input_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=(kernel_size - 1) // 2,
            )
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample_with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class AttentionBlock(nn.Module):
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


class AdaptiveGroupNorm2D(nn.Module):
    def __init__(self, cond_channels, num_channels, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=num_channels, eps=eps, affine=False
        )
        self.gamma = nn.Linear(cond_channels, num_channels)
        self.beta = nn.Linear(cond_channels, num_channels)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps  # not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

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


# my oriignal Impl.
# def depth_to_space_2d(x, block_size_h, block_size_w):
#     batch_size, C, H, W = x.size()
#     block_size_h = block_size_h
#     block_size_w = block_size_w

#     # Ensure the channel dimension is divisible by the product of the block sizes
#     assert (
#         C % (block_size_h * block_size_w) == 0
#     ), "Channels must be divisible by block_size_h * block_size_w"

#     new_C = C // (block_size_h * block_size_w)
#     x = x.view(batch_size, block_size_h, block_size_w, new_C, H, W)
#     x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
#     x = x.view(batch_size, new_C, H * block_size_h, W * block_size_w)
#     return x


# from open-magvit
def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row) core implementation.

    Args:
        x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
        block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)

    return x


class DepthToSpace2DWithConv(nn.Module):
    def __init__(
        self,
        channels_in,
        block_size_h,
        block_size_w,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels_in,
            block_size_h * block_size_w * channels_in,
            kernel_size=3,
            padding=1,
        )
        # nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w

    def forward(self, x):
        x = self.conv(x)
        # x = depth_to_space_2d(x, self.block_size_h, self.block_size_w)
        x = depth_to_space(x, self.block_size_h)
        return x
