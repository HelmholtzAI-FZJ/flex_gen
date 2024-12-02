from torch import nn
import torch
import pickle
from pathlib import Path
import copy
from einops import rearrange


class Basictokenizer(nn.Module):

    @staticmethod
    def capture_init_args(locals_):
        # Remove 'self' and '__class__' from locals
        args = copy.deepcopy(locals_)
        args.pop("self", None)
        args.pop("__class__", None)
        return args

    def __init__(self, **kwargs):
        super().__init__()
        cleaned_kwargs = self.capture_init_args(kwargs)
        # dummy loss
        self._configs = pickle.dumps(cleaned_kwargs)
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @classmethod
    def init_and_load_from(cls, path, strict=True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location="cpu")
        pkg.pop("self", None)
        pkg.pop("__class__", None)

        assert "config" in pkg, "model configs were not found in this saved checkpoint"
        print(f"loading Tokenizer from {str(path)} ....")
        config = pickle.loads(pkg["config"])
        tokenizer = cls(**config)
        tokenizer.load(path, strict=strict)
        return tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def copy_for_eval(self):
        device = self.device
        vae_copy = copy.deepcopy(self.cpu())
        vae_copy.eval()
        return vae_copy.to(device)

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists(), f"{str(path)} already exists"

        model_state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}

        pkg = dict(
            # model_state_dict=self.state_dict(),
            model_state_dict=model_state_dict_cpu,
            # version=__version__,
            config=self._configs,
        )

        print(f"saving model to {str(path)}")
        torch.save(pkg, str(path))

    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cpu", weights_only=True)
        state_dict = pkg.get("model_state_dict")

        assert (
            state_dict is not None
        ), "model state_dict was not found in this saved checkpoint"

        self.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        x = self.encode(video)
        return x["codes"]

    @torch.no_grad()
    def indice_to_code(self, indices):
        return self.quantizers.indice_to_code(indices)

    @torch.no_grad()
    def decode_from_code_indices(
        self,
        codes,
    ):
        quantized = self.quantizers.indice_to_code(codes)
        return self.decode(rearrange(quantized, "b ... c -> b c ..."))

    @torch.no_grad()
    def latent_to_indice(self, latents):
        return self.quantizers.latent_to_indice(latents)[0]

    def encode(self, x):
        """
        return is the latent itself
        """
        raise NotImplementedError

    def decode(self, x):
        """
        return is the reconstracted image
        """
        raise NotImplementedError

    def forward(self, x, return_codes=False):
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
        raise NotImplementedError

    def get_last_dec_layer(self):
        raise NotImplementedError
