from timm.models import register_model
from flex_gen.autoencoders.baselineConv.baselineConvVAE import BaselineConvVAE
from flex_gen.autoencoders.baselineConv.baselineResConvVAE import BaselineResConvVAE

@register_model
def baseline_conv_vae(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return BaselineConvVAE.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return BaselineConvVAE(
        **kwargs
    )


@register_model
def baseline_res_conv_vae(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return BaselineResConvVAE.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return BaselineResConvVAE(
        **kwargs
    )
