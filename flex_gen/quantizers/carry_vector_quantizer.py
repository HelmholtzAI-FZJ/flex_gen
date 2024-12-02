import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class CarryVectorQuantizer(BaseVectorQuantizer):
    """
    Similar to the Hierarchical Vector Quantizer, the only difference is that it uses the same level and shares the codebook.
    """

    def __init__(
        self, base, num_levels, embed_dim, commitment_loss_weight=0.25, **kwargs
    ):
        super().__init__()
        self.base = base
        levels = [base] * num_levels
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self._num_embed = self._levels.prod().item()
        self.num_levels = self._levels.shape[0]
        assert embed_dim % self.num_levels == 0
        self.embed_dim = embed_dim // self._levels.shape[0]
        self.commitment_loss_weight = commitment_loss_weight

        # create the codebook of the desired size
        self.codebook = nn.Parameter(
            torch.empty(base, self.embed_dim), requires_grad=True
        )

        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        # nn.init.uniform_(self.codebook, -1 / self.base, 1 / self.base)
        codebook = torch.randn(self.base, self.embed_dim)
        codebook = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.codebook.weight.data = codebook

    def forward(self, x):
        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute diff
        diff = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indice

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        # compute in parallel
        latent = rearrange(latent, "... (g d) -> (... g) d", g=self.num_levels)
        # n, m
        dist = compute_dist(latent, self.codebook)
        # n, 1
        indice = torch.argmin(dist, dim=-1)
        indice = rearrange(indice, "(b g) -> b g", g=self.num_levels)
        indice = (indice * self._basis).sum(dim=-1).to(torch.int32)

        indice = unpack_one(indice, ps, "*")

        return indice

    def indice_to_code(self, indice):
        indice = (indice.unsqueeze(-1) // self._basis) % self._levels
        code_list = []
        for i in range(self.num_levels):
            code = F.embedding(indice[..., i], self.codebook)
            code_list.append(code.unsqueeze(-1))
        code = rearrange(torch.cat(code_list, dim=-1), "... d g -> ... (g d)")

        return code
