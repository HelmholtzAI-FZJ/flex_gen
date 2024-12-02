import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one
from functools import partial


class VectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        num_embed,
        embed_dim,
        commitment_loss_weight=0.25,
        use_uniform_init=True,
        use_l1_norm=False,
        use_l2_norm=False,
        lambda_tcr_loss=0,
        norm_distirbution_radius=1.0,
        lambda_entropy_loss=0,
        entropy_loss_type="softmax",
        entropy_temperature=1.0,
    ):
        super().__init__()
        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.norm_distirbution_radius = norm_distirbution_radius

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        self.init_codebook(use_uniform_init)

        if use_l1_norm:
            self.normlization_func = partial(F.normalize, p=1, dim=-1)
        elif use_l2_norm:
            self.normlization_func = partial(F.normalize, p=2, dim=-1)
        else:
            self.normlization_func = lambda x: x

        self.lambda_tcr_loss = lambda_tcr_loss
        self.lambda_entropy_loss = lambda_entropy_loss
        self.entropy_temperature = entropy_temperature
        self.entropy_loss_type = entropy_loss_type

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self, use_uniform_init=True):
        if use_uniform_init:
            # This is the default initialization in the Vector Quantizer
            if self.embed_dim > 1:
                nn.init.uniform_(
                    self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed
                )
            else:
                nn.init.uniform_(self.codebook.weight, -1, 1)
        else:
            codebook = torch.randn(self.num_embed, self.embed_dim)
            codebook = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.codebook.weight.data = codebook * self.norm_distirbution_radius

    def forward(self, x):
        # get indice
        indice, dist = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        x = self.normlization_func(x)

        # compute diff
        diff = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        if self.lambda_tcr_loss > 0:
            code, ps = pack_one(x, "* d")
            BS, D = code.shape
            Vocab, D = self.codebook.weight.shape

            tcr_code = -torch.logdet((code.t() @ code).float() * D / BS)
            tcr_codebook = -torch.logdet(
                (self.codebook.weight.t() @ self.codebook.weight).float() * D / Vocab
            )

            diff += (
                self.lambda_tcr_loss * tcr_code + self.lambda_tcr_loss * tcr_codebook
            )

        if self.lambda_entropy_loss > 0:
            entropy_loss = self.entropy_loss(latent=x, dist=dist)
            diff += self.lambda_entropy_loss * entropy_loss

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indice

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        code, ps = pack_one(latent, "* d")
        # n, m
        dist = compute_dist(
            self.normlization_func(code), self.normlization_func(self.codebook.weight)
        )
        # n, 1
        indice = torch.argmin(dist, dim=-1)
        indice = unpack_one(indice, ps, "*")

        return indice, dist

    def indice_to_code(self, indice):
        return self.codebook(indice)
