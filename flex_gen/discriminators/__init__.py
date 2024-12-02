from timm.models import register_model
from .no_discriminator import NoDiscriminator
from .styleGanDiscriminator import StyleGanDiscriminator
from .nlayer_discriminator import NLayerDiscriminator
from .dino_Discriminator import DinoDiscriminator


@register_model
def no_discriminator(**kwargs):
    return NoDiscriminator(**kwargs)


@register_model
def styleGanDiscriminator(**kwargs):
    return StyleGanDiscriminator(**kwargs)


@register_model
def magvit_discriminator(**kwargs):
    return StyleGanDiscriminator(
        in_channels=3, dim=128, channels_multiplier=(2, 4, 4, 4, 4)
    )


@register_model
def taming_discriminator(**kwargs):
    return NLayerDiscriminator(in_channels=3, dim=64, num_layers=3)


@register_model
def n_layer_discriminator(**kwargs):
    return NLayerDiscriminator(**kwargs)


@register_model
def dino_discriminator(**kwargs):
    return DinoDiscriminator(**kwargs)
