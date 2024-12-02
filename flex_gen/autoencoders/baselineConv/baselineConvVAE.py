from flex_gen.autoencoders.taming.layers import Encoder, Decoder
import torch
import torch.nn as nn
from typing import Dict, Tuple
from flex_gen.autoencoders.basic_tokenizer import Basictokenizer

from flex_gen.quantizers import AnyQuantizer


class BaselineConvVAE(Basictokenizer):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 256,
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
        self.config["quantizer_aux_loss_weight"] = quantizer_aux_loss_weight
        self.config["quantizer_config"] = quantizer_config
        if out_channels is None:
            out_channels = in_channels
        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=bias
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=bias
            ),
            nn.ReLU(inplace=True),
        )
        self.quant_conv = nn.Conv2d(
            hidden_channels, embedding_dim, kernel_size=1, stride=1, padding=0
        )
        nn.init.kaiming_normal_(self.quant_conv.weight, nonlinearity="linear")

        self.quantizers = AnyQuantizer.build_quantizer(quantizer_config)

        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=bias
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=bias
            ),
            nn.ReLU(inplace=True),
        )

        # self.conv_out = nn.Conv2d(hidden_channels, out_channels, 1)

        self.conv_out = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.post_quant_conv = nn.ConvTranspose2d(
            embedding_dim,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        nn.init.kaiming_normal_(self.post_quant_conv.weight, nonlinearity="linear")

    def parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.quant_conv.parameters(),
            *self.post_quant_conv.parameters(),
            *self.quantizers.parameters(),
            *self.conv_out.parameters(),
        ]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_extra_state(self):
        return {"config": self.config}

    def set_extra_state(self, state):
        pass

    def get_last_dec_layer(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.conv_out.weight

    def encode(self, x: torch.Tensor):
        """Encode an input tensor into a latent tensor."""
        h = self.encoder(x)
        latent = self.quant_conv(h)
        # Split the moments into mean and log variance
        return self.quantizers(latent)

    def decode(self, z: torch.Tensor):
        """Decode a latent tensor into an output tensor."""
        z = self.post_quant_conv(z)
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
