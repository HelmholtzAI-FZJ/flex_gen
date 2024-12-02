from flex_gen.autoencoders.taming.layers import Encoder, Decoder
import torch
import torch.nn as nn
from typing import Dict, Tuple
from flex_gen.autoencoders.basic_tokenizer import Basictokenizer

# from torch.nn import GroupNorm
from flex_gen.quantizers import AnyQuantizer
from torch.nn import functional as F
from einops import rearrange

NUM_GROUPS = 1
# NUM_GROUPS = 32


class BaselineResConvVAE(Basictokenizer):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 256,
        num_res_blocks=2,
        channel_multipliers=[1, 1, 1],
        out_channels=None,
        quantizer_aux_loss_weight=1.0,
        quantizer_config={
            "quantize_type": "vq",
            "embed_dim": 4,
            "num_embed": 16384,
        },
        bias=False,
    ):
        super().__init__(**self.capture_init_args(locals()))
        self.config = {}
        self.config["in_channels"] = in_channels
        self.config["hidden_channels"] = hidden_channels
        self.config["embedding_dim"] = embedding_dim
        self.config["out_channels"] = out_channels
        self.config["num_res_blocks"] = num_res_blocks
        self.config["channel_multipliers"] = channel_multipliers
        self.config["quantizer_aux_loss_weight"] = quantizer_aux_loss_weight
        self.config["quantizer_config"] = quantizer_config
        if out_channels is None:
            out_channels = in_channels
        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

        self.encoder = ResConvEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            embed_dim=embedding_dim,
            bias=bias,
        )
        self.quantizers = AnyQuantizer.build_quantizer(quantizer_config)
        self.decoder = ResConvDecoder(
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            embed_dim=embedding_dim,
            bias=bias,
        )

        # self.conv_out = nn.Conv2d(hidden_channels, out_channels, 1)

        self.conv_out = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.quantizers.parameters(),
            *self.conv_out.parameters(),
        ]

    def get_extra_state(self):
        return {"config": self.config}

    def set_extra_state(self, state):
        pass

    def get_last_dec_layer(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.conv_out.weight

    def encode(self, x: torch.Tensor):
        """Encode an input tensor into a latent tensor."""
        latent = self.encoder(x)
        # latent = self.quant_conv(latent)
        # Split the moments into mean and log variance
        return self.quantizers(latent)

    def decode(self, z: torch.Tensor):
        """Decode a latent tensor into an output tensor."""
        x_recon = self.decoder(z)
        x_recon = self.conv_out(x_recon)
        return x_recon

    def forward(
        self,
        x: torch.Tensor,
        return_recon_loss_only=False,
    ) -> Dict[str, torch.Tensor]:
        """Forward through the autoencoder."""

        quantize_ret = self.encode(x)
        quantized = quantize_ret.pop("quantized")
        codes = quantize_ret.pop("codes")
        x_recon = self.decode(quantized)

        recon_loss = torch.nn.functional.mse_loss(x, x_recon)
        if return_recon_loss_only:
            return {
                "codes": codes,
                "recon": x_recon,
                "quantized": quantized,
                "recon_loss": recon_loss,
            }

        aux_losses = quantize_ret.pop("aux_loss", self.zero)
        quantizer_loss_breakdown = {"QUANT_" + k: v for k, v in quantize_ret.items()}

        loss_sum = recon_loss + aux_losses * self.quantizer_aux_loss_weight
        loss_breakdown = {
            "codes": codes,
            "recon": x_recon,
            "recon_loss": recon_loss,
            "aux_loss": aux_losses,
            "quantized": quantized,
        }
        loss_breakdown.update(quantizer_loss_breakdown)

        return loss_sum, loss_breakdown


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, bias: bool = False):
        """
        :param in_channels: input channels of the residual block
        :param out_channels: if None, use in_channels. Else, adds a 1x1 conv layer.
        """
        super().__init__()

        if out_channels is None or out_channels == in_channels:
            out_channels = in_channels
            self.conv_shortcut = None
        else:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding="same", bias=bias
            )

        self.norm1 = GroupNorm(num_groups=NUM_GROUPS, num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", bias=bias
        )

        self.norm2 = GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same", bias=bias
        )

    def forward(self, x):

        residual = F.silu(self.norm1(x))
        residual = self.conv1(residual)

        residual = F.silu(self.norm2(residual))
        residual = self.conv2(residual)

        if self.conv_shortcut is not None:
            # contiguous prevents warning:
            # https://github.com/pytorch/pytorch/issues/47163
            # https://discuss.pytorch.org/t/why-does-pytorch-prompt-w-accumulate-grad-h-170-warning-grad-and-param-do-not-obey-the-gradient-layout-contract-this-is-not-an-error-but-may-impair-performance/107760
            x = self.conv_shortcut(x.contiguous())

        return x + residual


class Downsample(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        res = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return res


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        scale_factor: float = 2.0,
        mode: str = "nearest-exact",
        bias: bool = False,
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding="same", bias=bias
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class ResConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_res_blocks,
        channel_multipliers,
        embed_dim,
        bias,
    ):
        super().__init__()
        # get params start
        channels = hidden_channels
        num_res_blocks = num_res_blocks
        channel_multipliers = channel_multipliers
        embed_dim = embed_dim
        bias = bias
        # get params end

        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        blocks = []
        ch_in = channels

        for i in range(len(channel_multipliers)):

            ch_out = channels * channel_multipliers[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out, bias))
                ch_in = ch_out

            blocks.append(Downsample())

        self.blocks = nn.Sequential(*blocks)

        self.final_residual = nn.Sequential(
            *[ResBlock(ch_in, ch_in, bias) for _ in range(num_res_blocks)]
        )

        self.norm = GroupNorm(num_groups=NUM_GROUPS, num_channels=ch_in)
        self.conv_out = nn.Conv2d(
            ch_in, embed_dim, kernel_size=1, padding="same", bias=bias
        )

    def forward(self, x):

        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.final_residual(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class ResConvDecoder(nn.Module):
    def __init__(
        self, hidden_channels, num_res_blocks, channel_multipliers, embed_dim, bias
    ):

        super().__init__()

        # get params start
        channels = hidden_channels
        num_res_blocks = num_res_blocks
        channel_multipliers = channel_multipliers
        embed_dim = embed_dim
        bias = bias
        # get params end

        ch_in = channels * channel_multipliers[-1]

        self.conv_in = nn.Conv2d(
            embed_dim, ch_in, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.initial_residual = nn.Sequential(
            *[ResBlock(ch_in, ch_in, bias) for _ in range(num_res_blocks)]
        )

        blocks = []
        for i in reversed(range(len(channel_multipliers))):
            blocks.append(Upsample(ch_in))
            ch_out = channels * channel_multipliers[i - 1] if i > 0 else channels

            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out, bias))
                ch_in = ch_out

        self.blocks = nn.Sequential(*blocks)

        self.norm = GroupNorm(num_groups=NUM_GROUPS, num_channels=channels)
        # self.conv_out = nn.Conv2d(
        #     channels, 3, kernel_size=3, stride=1, padding=1, bias=bias
        # )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.initial_residual(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = F.silu(x)
        # x = self.conv_out(x)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        """
        We use a custom implementation for GroupNorm, since h=w=1 may raise some problem,
        see https://github.com/pytorch/pytorch/issues/115940
        """
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, h, w = x.shape

        x = rearrange(x, "b (g n) h w -> b g (n h w)", g=self.num_groups)
        mean = torch.mean(x, dim=2, keepdim=True)
        variance = torch.var(x, dim=2, keepdim=True)

        x = (x - mean) / (variance + self.eps).sqrt()

        x = rearrange(x, "b g (n h w) -> b (g n) h w", h=h, w=w)

        x = x * self.weight + self.bias

        return x

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"
