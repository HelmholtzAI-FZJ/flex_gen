import torch
from torch import nn
import copy
from collections import OrderedDict


@torch.no_grad()
def update_ema(ema_model, model, beta=0.9999, **kwargs):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(beta).add_(param.data, alpha=1 - beta)
