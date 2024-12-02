from timm.models import register_model
import diffusers
import transformers
import importlib
from .hf_autoencoder import HFAutoEncoder


@register_model
def hf_transformers(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"]
    del kwargs["pretrained_cfg_overlay"]
    raise NotImplementedError


@register_model
def hf_diffusers(
    diffuser_module="StableDiffusionPipeline", repo=None, pretrained=False, **kwargs
):
    """
    Notice that for pipeline models, it only work for pretrained=True
    """
    del kwargs["pretrained_cfg"]
    del kwargs["pretrained_cfg_overlay"]
    target_ojb = getattr(importlib.import_module(f"diffusers"), diffuser_module)
    quantizer_aux_loss_weight = kwargs.pop("quantizer_aux_loss_weight", 1.0)
    if pretrained:
        if "path" in kwargs:
            return HFAutoEncoder.init_and_load_from(path=kwargs["path"])
        else:
            model = target_ojb.from_pretrained(repo, **kwargs)
    else:
        model = target_ojb(**kwargs)

    # auto wrap StableDiffusionPipeline's vae
    if hasattr(model, "vae"):
        model = model.vae
    return HFAutoEncoder(
        model=model, quantizer_aux_loss_weight=quantizer_aux_loss_weight, **kwargs
    )
