import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one
from functools import partial


def random_argmin(dist, dim=-1):
    # Find the minimum value along the last dimension
    min_val = dist.min(dim=dim, keepdim=True).values
    # Get a mask of all minimum values
    min_mask = (dist == min_val)
    # Get random indices from the masked minimums
    random_indices = torch.multinomial(min_mask.float(), 1).squeeze(-1)
    return random_indices


def softmax_sample(dist, tau=1.0, dim=-1):
    """
    Perform softmax sampling over distances.
    tau = 0.1: Strongly favours the lowest distance.
    Probabilities = [0.001, 0.997, 0.002, 0.997, 0.001]
    tau = 1.0: Balanced prioritisation.
    Probabilities = [0.09, 0.42, 0.23, 0.42, 0.04]
    tau = 10.0: Closer to uniform.
    Probabilities = [0.19, 0.21, 0.20, 0.21, 0.18]
    tau = 10e6 : Almost uniform.
    Probabilities = [0.20, 0.20, 0.20, 0.20, 0.20]
    Args:
        dist (torch.Tensor): The distance tensor.
        tau (float): Temperature parameter for softmax.

    Returns:
        torch.Tensor: Indices sampled based on softmax probabilities.
    """
    # Compute softmax probabilities (lower distance = higher probability)
    probabilities = torch.softmax(-dist / tau, dim=dim)
    # Sample indices based on probabilities
    sampled_indices = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    return sampled_indices


class GroupSphereQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        num_embed,
        embed_dim,
        num_group=1,
        use_uniform_init=False,
        commitment_loss_weight=0.25,
        use_l2_norm=False,
        use_l1_norm=False,
        force_1d_clip=False,
        argmin_func='random',
        argmin_softmax_tau=1.0,
        **kwargs,
    ):
        super().__init__()
        self._num_embed = num_embed
        self.num_group = num_group
        self.embed_dim = embed_dim // num_group
        self.commitment_loss_weight = commitment_loss_weight

        if use_l1_norm:
            self.normlization_func = partial(F.normalize, p=1, dim=-1)
        elif use_l2_norm:
            self.normlization_func = partial(F.normalize, p=2, dim=-1)
        else:
            self.normlization_func = lambda x: x

        if use_l1_norm is False and use_l2_norm is False and force_1d_clip:
            if use_uniform_init and self.embed_dim == 1:
                # clip the codebook to [-1, 1] for better performance
                self.normlization_func = lambda x: torch.clamp(x, -1, 1)

        argmin_func = argmin_func.lower() if argmin_func is not None else None
        if argmin_func in [None, "default", "argmin"]:
            self.argmin_func = partial(torch.argmin, dim=-1)
        elif argmin_func == "random":
            self.argmin_func = partial(random_argmin, dim=-1)
        elif argmin_func == "softmax":
            self.argmin_func = partial(softmax_sample, dim=-1, tau=argmin_softmax_tau)


        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        self.init_codebook(use_uniform_init)

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self, use_uniform_init=False):
        # nn.init.normal_(self.codebook.weight, mean=0, std=self.embed_dim**-0.5)
        # nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)

        if use_uniform_init:
            if self.embed_dim > 1:
                nn.init.uniform_(
                    self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed
                )
            else:
                nn.init.uniform_(self.codebook.weight, -1, 1)
        else:
            codebook = torch.randn(self.num_embed, self.embed_dim)
            codebook = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.codebook.weight.data = codebook

    def forward(self, x):
        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        x = rearrange(x, "... (g d) -> ... g d", g=self.num_group)
        x = self.normlization_func(x)
        x = rearrange(x, "... g d -> ... (g d)")

        # compute diff
        diff = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        x_quant = x + (x_quant - x).detach()
        return x_quant, diff, indice

    def latent_to_indice(self, latent):
        # if it is GVQ and codebook is large, e.g. >=256k, we use micrio-batch for better performance
        if self.num_embed >= 256000 and (self.num_group >= 2 or self.embed_dim >= 16):
            indices = []
            for minibatch in range(latent.shape[0]):
                tmp_latent, ps = pack_one(latent[minibatch], "* d")
                tmp_latent = rearrange(
                    tmp_latent, "... (g d) -> (... g) d", g=self.num_group
                )
                # n, m
                dist = compute_dist(
                    self.normlization_func(tmp_latent),
                    self.normlization_func(self.codebook.weight),
                )
                # n, 1
                # indice = torch.argmin(dist, dim=-1)
                indice = self.argmin_func(dist)
                indice = rearrange(indice, "(b g) -> b g", g=self.num_group)
                indice = unpack_one(indice, ps, "* g")
                indices.append(indice.unsqueeze(0))

            return torch.cat(indices, dim=0)

        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        latent = rearrange(latent, "... (g d) -> (... g) d", g=self.num_group)
        # n, m
        # dist = compute_dist(latent, self.codebook.weight)
        dist = compute_dist(
            self.normlization_func(latent), self.normlization_func(self.codebook.weight)
        )
        # n, 1
        # indice = torch.argmin(dist, dim=-1)
        indice = self.argmin_func(dist)
        indice = rearrange(indice, "(b g) -> b g", g=self.num_group)
        indice = unpack_one(indice, ps, "* g")

        return indice

    def indice_to_code(self, indice):
        code = self.codebook(indice)
        code = rearrange(code, "... g d -> ... (g d)")

        return code
