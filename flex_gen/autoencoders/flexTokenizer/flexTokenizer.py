import torch
import torch.nn.functional as F
from torch import nn, Tensor
from flex_gen.quantizers import AnyQuantizer
from flex_gen.autoencoders.basic_tokenizer import Basictokenizer

from .flexTokenizerEncoder import Encoder
from .flexTokenizerDecoder import Decoder


class FlexTokenizer(Basictokenizer):
    def __init__(
        self,
        *,
        input_channels=3,
        hidden_channels=128,
        encoder_layer_configs=(4, 3, 4, 3, 4),
        decoder_layer_configs=(3, 4, 3, 4, 4),
        # from w.r.to the channel multiplier, the first one is the input channel
        channel_multipliers=(1, 2, 2, 4),
        mid_block_attn=False,
        attention_resolutions=(),
        embedding_dim=4,
        use_adaptive_norm=False,
        use_learnable_up_down_sample=False,
        residual_conv_kernel_size=3,
        input_conv_kernel_size=3,
        output_conv_kernel_size=3,
        dropout_probability=0.0,
        quantizer_aux_loss_weight=1.0,
        quantizer_config={
            "quantize_type": "vq",
            "embed_dim": 4,
            "num_embed": 8192,
        },
    ):
        super().__init__(**self.capture_init_args(locals()))

        if attention_resolutions is None:
            attention_resolutions = []
        elif isinstance(attention_resolutions, int):
            attention_resolutions = [attention_resolutions]

        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            channel_multipliers=channel_multipliers,
            num_residual_blocks=encoder_layer_configs,
            attention_resolutions=attention_resolutions,
            dropout_probability=dropout_probability,
            use_attention=mid_block_attn,
            residual_conv_kernel_size=residual_conv_kernel_size,
            input_conv_kernel_size=input_conv_kernel_size,
        )

        self.decoder = Decoder(
            output_channels=input_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            channel_multipliers=channel_multipliers,
            num_residual_blocks=decoder_layer_configs,
            attention_resolutions=attention_resolutions,
            dropout_probability=dropout_probability,
            use_depth_to_space=use_learnable_up_down_sample,
            use_attention=mid_block_attn,
            use_adaptive_norm=use_adaptive_norm,
            residual_conv_kernel_size=residual_conv_kernel_size,
            output_conv_kernel_size=output_conv_kernel_size,
        )
        self.quantizers = AnyQuantizer.build_quantizer(quantizer_config)

    @property
    def device(self):
        return self.zero.device

    def parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.quantizers.parameters(),
        ]

    def encode(
        self,
        x: Tensor,
        quantize=False,
    ):
        x = self.encoder(x)
        return self.quantizers(x)

    def decode(self, quantized: Tensor):
        return self.decoder(quantized)

    def get_last_dec_layer(self):
        return self.decoder.final_layers[-1].weight

    def forward(
        self,
        video: Tensor,
        return_recon_loss_only=False,
    ):

        quantize_ret = self.encode(video)
        quantized = quantize_ret.pop("quantized")
        codes = quantize_ret.pop("codes")

        recon_video = self.decode(quantized)

        recon_loss = F.mse_loss(video, recon_video)
        if return_recon_loss_only:
            return {
                "codes": codes,
                "recon": recon_video,
                "recon_loss": recon_loss,
                "quantized": quantized,
            }

        aux_losses = quantize_ret.pop("aux_loss", self.zero)
        quantizer_loss_breakdown = {"QUANT_" + k: v for k, v in quantize_ret.items()}

        loss_sum = recon_loss + aux_losses * self.quantizer_aux_loss_weight
        loss_breakdown = {
            "codes": codes,
            "recon": recon_video,
            "recon_loss": recon_loss,
            "aux_loss": aux_losses,
            "quantized": quantized,
        }
        loss_breakdown.update(quantizer_loss_breakdown)

        return loss_sum, loss_breakdown
