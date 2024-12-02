import torch
from flex_gen.autoencoders.basic_tokenizer import Basictokenizer


class HFAutoEncoder(Basictokenizer):
    def __init__(self, *, model, quantizer_aux_loss_weight=1.0):
        super().__init__(**self.capture_init_args(locals()))
        self.model = model
        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        # x = self.encode(video)
        # x = self.quantizers(x, return_loss_breakdown=True)
        # return x["codes"]

        if not hasattr(self.model, "quantize"):
            raise ValueError("No encode function found")
        latents = self.encode(video)

        if "codes" in latents:
            return latents
        else:
            raise ValueError("No codes found")

    @torch.no_grad()
    def decode_from_code_indices(
        self,
        codes,
    ):
        # quantized = self.quantizers.indice_to_code(codes)
        # return self.decode(quantized)
        raise NotImplementedError

    def encode(self, x, **kwargs):
        """
        return is the latent itself
        """
        encode_output = self.model.encode(x)

        if isinstance(encode_output, torch.Tensor):
            return {"quantized": encode_output}
        ret = {}

        if "latent_dist" in encode_output:
            ret["quantized"] = encode_output["latent_dist"].sample()
        if "latent" in encode_output:
            ret["quantized"] = encode_output["latent"]
        if "codes" in encode_output:
            ret["codes"] = encode_output["codes"]
        if len(ret) == 0:
            raise ValueError("Unknow encode output")
        return ret

    def decode(self, x):
        """
        return is the reconstracted image
        """
        x_recon = self.model.decode(x)
        if "sample" in x_recon:
            return x_recon["sample"]
        elif isinstance(x_recon, torch.Tensor):
            return x_recon
        else:
            raise ValueError("Unknow decode output")

    def forward(self, x, return_codes=False, return_recon_loss_only=False):
        """
        return is a 2-tuple of (loss_sum, loss_breakdown)
        loss_breakdown = {
            "recon": x_recon,
            "recon_loss": recon_loss,
            "aux_loss": aux_losses,
            "quantized": z,
            "codes": codes,
            "random_latent": encoded_dist.sample(),
        }
        """
        encode_output = self.model.encode(x)
        is_kl_autoencoder = False
        if "latent_dist" in encode_output:
            # latent = encode_output["latent_dist"].mode()
            latent = encode_output["latent_dist"].sample()
            random_latent = encode_output["latent_dist"].sample()
            is_kl_autoencoder = True
        elif "latent" in encode_output:
            latent = encode_output["latents"]
        elif isinstance(encode_output, torch.Tensor):
            latent = encode_output
        else:
            raise ValueError("Unknow encode output")

        # =========
        decode_output = self.model.decode(latent)
        if "sample" in decode_output:
            x_recon = decode_output["sample"]
        elif isinstance(decode_output, torch.Tensor):
            x_recon = decode_output
        else:
            raise ValueError("Unknow decode output")

        recon_loss = torch.nn.functional.mse_loss(x, x_recon)

        # ========
        if is_kl_autoencoder:
            if return_recon_loss_only:
                return {
                    "quantized": latent,
                    "recon": x_recon,
                    "recon_loss": recon_loss,
                }

            encoded_dist = encode_output["latent_dist"]
            ## Improved version
            # log_var = encoded_dist.logvar
            # mean = encoded_dist.mean
            # num_latent_elements = mean[0].numel()
            # num_output_elements = x[0].numel()
            # dimensionality_weight = num_latent_elements / num_output_elements
            # aux_losses = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
            # loss_sum = (
            #     recon_loss
            #     + aux_losses * self.quantizer_aux_loss_weight * dimensionality_weight
            # )

            aux_losses = encoded_dist.kl().mean()  # original
            loss_sum = recon_loss + aux_losses * self.quantizer_aux_loss_weight
            loss_breakdown = {
                "recon": x_recon,
                "recon_loss": recon_loss,
                "aux_loss": aux_losses,
                "quantized": latent,
                "random_latent": random_latent,
            }

        else:
            with torch.no_grad():
                if (
                    hasattr(self.model.quantize, "lookup_from_codebook")
                    and self.model.config.lookup_from_codebook
                ):
                    code = self.model.quantize.get_codebook_entry(latent, None)
                else:
                    raise ValueError("No quantizer found")

            if return_codes:
                return {
                    "codes": code,
                    "quantized": latent,
                }
            if return_recon_loss_only:
                return {
                    "codes": code,
                    "recon": x_recon,
                    "recon_loss": recon_loss,
                    "quantized": latent,
                }

            aux_losses = decode_output.get("commit_loss", self.zero)
            loss_sum = recon_loss + aux_losses * self.quantizer_aux_loss_weight
            loss_breakdown = {
                "codes": code,
                "recon": x_recon,
                "recon_loss": recon_loss,
                "aux_loss": aux_losses,
                "quantized": latent,
            }

        return loss_sum, loss_breakdown

    def get_last_dec_layer(self):
        if hasattr(self.model, "decoder"):
            if hasattr(self.model.decoder, "conv_out"):
                return self.model.decoder.conv_out.weight
            else:
                raise ValueError("No decoder found")
        else:
            raise ValueError("No decoder found")
