from flex_gen.autoencoders.taming.layers import (
    Encoder,
    Decoder,
    GaussianDistribution,
)
import torch
import torch.nn as nn
from typing import Dict, Tuple
from flex_gen.autoencoders.basic_tokenizer import Basictokenizer


class KLAutoEncoder(Basictokenizer):
    """Autoencoder module for training a latent diffusion model.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the latent channels. Default: `True`.
        channel_multipliers (Tuple[int, ...]): Multipliers for the number of channels in each block. Default: `(1, 2, 4, 8)`.
        num_residual_blocks (int): Number of residual blocks in each block. Default: `4`.
        use_conv_shortcut (bool): Whether to use a conv for the shortcut. Default: `False`.
        dropout (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a conv for down/up sampling. Default: `True`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
        use_attention (bool): Whether to use attention layers. Default: `True`.
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 4,
        double_latent_channels: bool = True,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_residual_blocks: int = 2,
        use_conv_shortcut=False,
        dropout_probability: float = 0.0,
        resample_with_conv: bool = True,
        zero_init_last: bool = False,
        use_attention: bool = True,
        quantizer_aux_loss_weight=1.0,
        use_auto_kl_loss_lambda=False,
    ):
        super().__init__(**self.capture_init_args(locals()))
        self.config = {}
        self.config["input_channels"] = input_channels
        self.config["output_channels"] = output_channels
        self.config["hidden_channels"] = hidden_channels
        self.config["latent_channels"] = latent_channels
        self.config["double_latent_channels"] = double_latent_channels
        self.config["channel_multipliers"] = channel_multipliers
        self.config["num_residual_blocks"] = num_residual_blocks
        self.config["use_conv_shortcut"] = use_conv_shortcut
        self.config["dropout_probability"] = dropout_probability
        self.config["resample_with_conv"] = resample_with_conv
        self.config["use_attention"] = use_attention
        self.config["zero_init_last"] = zero_init_last
        self.config["quantizer_aux_loss_weight"] = quantizer_aux_loss_weight
        self.set_extra_state(None)
        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight
        self.encoder = Encoder(
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["hidden_channels"],
            latent_channels=self.config["latent_channels"],
            double_latent_channels=self.config["double_latent_channels"],
            channel_multipliers=self.config["channel_multipliers"],
            num_residual_blocks=self.config["num_residual_blocks"],
            use_conv_shortcut=self.config["use_conv_shortcut"],
            dropout_probability=self.config["dropout_probability"],
            resample_with_conv=self.config["resample_with_conv"],
            zero_init_last=self.config["zero_init_last"],
            use_attention=self.config["use_attention"],
        )

        channels = (
            2 * self.config["latent_channels"]
            if self.config["double_latent_channels"]
            else self.config["latent_channels"]
        )
        self.quant_conv = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )
        nn.init.kaiming_normal_(self.quant_conv.weight, nonlinearity="linear")
        # KL divergence is minimized when mean is 0.0 and log variance is 0.0
        # However, this corresponds to no information in the latent space.
        # So, init these such that latents are mean 0 and variance 1, with a rough snr of 1
        self.quant_conv.weight.data[: channels // 2] *= 0.707
        self.quant_conv.weight.data[channels // 2 :] *= 0.707
        if self.quant_conv.bias is not None:
            self.quant_conv.bias.data[channels // 2 :].fill_(-0.9431)

        self.decoder = Decoder(
            latent_channels=self.config["latent_channels"],
            output_channels=self.config["output_channels"],
            hidden_channels=self.config["hidden_channels"],
            channel_multipliers=self.config["channel_multipliers"],
            num_residual_blocks=self.config["num_residual_blocks"],
            use_conv_shortcut=self.config["use_conv_shortcut"],
            dropout_probability=self.config["dropout_probability"],
            resample_with_conv=self.config["resample_with_conv"],
            zero_init_last=self.config["zero_init_last"],
            use_attention=self.config["use_attention"],
        )

        self.post_quant_conv = nn.Conv2d(
            self.config["latent_channels"],
            self.config["latent_channels"],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.kaiming_normal_(self.post_quant_conv.weight, nonlinearity="linear")
        self.use_auto_kl_loss_lambda = use_auto_kl_loss_lambda

    def parameters(self):
        return [
            *self.encoder.parameters(),
            *self.quant_conv.parameters(),
            *self.decoder.parameters(),
            *self.post_quant_conv.parameters(),
        ]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_extra_state(self):
        return {"config": self.config}

    def set_extra_state(self, state):
        pass

    def get_last_layer(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.decoder.conv_out.weight

    def get_last_dec_layer(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> GaussianDistribution:
        """Encode an input tensor into a latent tensor."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # Split the moments into mean and log variance
        mean, log_var = (
            moments[:, : self.config["latent_channels"]],
            moments[:, self.config["latent_channels"] :],
        )
        return {"quantized": GaussianDistribution(mean, log_var)}

    @torch.no_grad()
    def tokenize(self, x):
        self.eval()
        encoded_dist = self.encode(x)
        z = encoded_dist.sample()
        return {
            "codes": z,
        }

    def decode(self, z: torch.Tensor):
        """Decode a latent tensor into an output tensor."""
        z = self.post_quant_conv(z)
        x_recon = self.decoder(z)
        return x_recon

    def forward(
        self,
        x: torch.Tensor,
        return_recon_loss_only=False,
    ) -> Dict[str, torch.Tensor]:
        """Forward through the autoencoder."""
        encoded_dist = self.encode(x)["quantized"]
        # z = encoded_dist.mean

        z = encoded_dist.sample()
        # z = encoded_dist.mode()
        x_recon = self.decode(z)

        recon_loss = torch.nn.functional.mse_loss(x, x_recon)
        if return_recon_loss_only:
            return {
                "quantized": z,
                "recon": x_recon,
                "recon_loss": recon_loss,
            }
        var = encoded_dist.var
        log_var = encoded_dist.log_var
        mean = encoded_dist.mean
        if self.use_auto_kl_loss_lambda:
            # This is a better imrpoved version of kl-loss with dimensionality weight
            num_latent_elements = mean[0].numel()
            num_output_elements = x[0].numel()
            dimensionality_weight = num_latent_elements / num_output_elements
            aux_losses = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        else:
            # This is LDM/SD's implementation of kl-loss
            aux_losses = 0.5 * torch.sum(
                torch.pow(mean, 2) + var - 1.0 - log_var, dim=[1, 2, 3]
            )
            dimensionality_weight = 1

        loss_sum = (
            recon_loss
            + aux_losses.mean() * self.quantizer_aux_loss_weight * dimensionality_weight
        )
        loss_breakdown = {
            "recon": x_recon,
            "recon_loss": recon_loss,
            "aux_loss": aux_losses,
            "quantized": z,
            "random_latent": encoded_dist.sample(),
        }

        return loss_sum, loss_breakdown
