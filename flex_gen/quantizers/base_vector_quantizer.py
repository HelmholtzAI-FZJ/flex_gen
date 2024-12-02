from abc import ABC, abstractmethod

from torch import nn

from .utils import compute_dist, pack_one, sample_entropy_loss_fn


class BaseVectorQuantizer(ABC, nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.entropy_temperature = 1.0
        self.entropy_loss_type = "softmax"
        self.normlization_func = lambda x: x

    @property
    def num_embed(self):
        pass

    @abstractmethod
    def init_codebook(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def latent_to_indice(self, latent):
        pass

    @abstractmethod
    def indice_to_code(self, indice):
        pass

    def entropy_loss(self, latent=None, dist=None):
        assert (
            latent is not None or dist is not None
        ), "At least one of latent or dist needs to be specified."
        if dist is None:
            if isinstance(self.codebook, nn.Module):
                self.codebook.weight
            else:
                self.codebook
            # (b, *, d) -> (n, d)
            latent, ps = pack_one(latent, "* d")

            if hasattr(self.codebook, "weight"):
                dist = compute_dist(
                    self.normlization_func(latent),
                    self.normlization_func(self.codebook.weight),
                )
            else:
                # PACK FOR LFQ
                dist = compute_dist(
                    self.normlization_func(latent),
                    self.normlization_func(self.codebook),
                )

        loss = sample_entropy_loss_fn(
            dist, self.entropy_temperature, self.entropy_loss_type
        )

        return loss
