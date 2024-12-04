from timm.models import register_model
from .flexTokenizer import FlexTokenizer
from huggingface_hub import hf_hub_download

"""
The magvit2 here is fully euqal to the (2d Conv) magvit2 model in the paper

The flex_openmagvit and flex_vqgan are not equal to the models in the paper, the difference is :
==== MagViT2 ====
(extra) Residual block : D   -> D//2
(0-N)   Residual blocks: D/2 -> D//2

==== Flex OpenMagViT and Flex Vqgan====
(0)   Residual blocks: D    -> D//2 
(1-N) Residual blocks: D//2 -> D//2 


The MagViT always has few more ResBlocks for channel reduction/increase,

"""

@register_model
def flexTokenizer(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return FlexTokenizer.init_and_load_from(path=kwargs["path"])
        elif 'repo_id' in kwargs:
            # Automatically fetch pretrained weights from Hugging Face Hub
            repo_id = kwargs.get("repo_id")
            filename = kwargs.get("filename", "checkpoint.pt")  # Default file
            # Download from Hugging Face
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
            return FlexTokenizer.init_and_load_from(path=checkpoint_path)            
        else:
            raise ValueError("path is required for pretrained model")

    return FlexTokenizer(**kwargs)


@register_model
def flex_magvit2(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return FlexTokenizer.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return FlexTokenizer(
        use_adaptive_norm=True, use_learnable_up_down_sample=True, **kwargs
    )


@register_model
def flex_openmagvit(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return FlexTokenizer.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return FlexTokenizer(
        use_adaptive_norm=True,
        use_learnable_up_down_sample=True,
        channel_multipliers=(1, 2, 2, 4),
        encoder_layer_configs=(2, 2, 2, 2, 2),
        decoder_layer_configs=(2, 2, 2, 2, 2),
        **kwargs
    )


@register_model
def flex_vqgan(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return FlexTokenizer.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return FlexTokenizer(
        channel_multipliers=(1, 2, 2, 4),
        encoder_layer_configs=(2, 2, 2, 2, 2),
        decoder_layer_configs=(2, 2, 2, 2, 2),
        mid_block_attn=True,
    )