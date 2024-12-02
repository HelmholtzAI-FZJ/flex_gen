import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class GumbelVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        num_embed,
        embed_dim,
        commitment_loss_weight=0.25,
        temp=1,
        kl_loss_weight=5e-4,
        **kwargs
    ):
        super().__init__()
        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.temp = temp
        self.kl_loss_weight = kl_loss_weight

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        # nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)
        codebook = torch.randn(self.num_embed, self.embed_dim)
        codebook = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.codebook.weight.data = codebook

    def forward(self, x):
        # get indice
        indice, kl_loss = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute diff
        # diff = self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x) + kl_loss
        diff = kl_loss

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indice

    def latent_to_indice(self, latent, eps=1e-10):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        # n, m
        logits = -compute_dist(latent, self.codebook.weight)
        # n, m
        hard = False if self.training else True
        indice = F.gumbel_softmax(logits, tau=self.temp, dim=-1, hard=hard)
        indice = unpack_one(indice, ps, "* m")

        if self.kl_loss_weight > 0:
            # + kl divergence to the prior (uniform) loss, increase cb usage
            # Note:
            #       KL(P(x), Q(x)) = sum_x (P(x) * log(P(x) / Q(x)))
            #       in this case: P(x) is p, Q(x) is uniform distribution (1 / num_embed)
            p = F.softmax(logits, dim=-1)
            kl_loss = (
                self.kl_loss_weight
                * torch.sum(p * torch.log(p * self.num_embed + eps), dim=-1).mean()
            )
        else:
            kl_loss = torch.tensor(0.0).cuda().float()

        return indice, kl_loss

    def indice_to_code(self, indice):
        return torch.einsum("... m, m d -> ... d", indice, self.codebook.weight)
