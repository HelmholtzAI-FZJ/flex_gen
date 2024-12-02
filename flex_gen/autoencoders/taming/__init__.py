from timm.models import register_model
from flex_gen.autoencoders.taming.vq_gan import VQAutoEncoder
from flex_gen.autoencoders.taming.kl_gan import KLAutoEncoder


@register_model
def vq_gan_F8(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return VQAutoEncoder.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return VQAutoEncoder(
        latent_channels=4,
        channel_multipliers=(1, 2, 4, 4),
        quantizer_config={
            "quantize_type": "vq",
            "embed_dim": 4,
            "num_embed": 16384,
        },
    )


@register_model
def vq_gan_taming(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return VQAutoEncoder.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return VQAutoEncoder(**kwargs)


@register_model
def kl_gan_F8(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return VQAutoEncoder.init_and_load_from(path=kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")
    return KLAutoEncoder(
        channel_multipliers=(1, 2, 4, 4),
    )


@register_model
def kl_gan_taming(pretrained=False, **kwargs):
    del kwargs["pretrained_cfg"], kwargs["pretrained_cfg_overlay"]
    if pretrained:
        if "path" in kwargs:
            return KLAutoEncoder.init_and_load_from(kwargs["path"])
        else:
            raise ValueError("path is required for pretrained model")

    return KLAutoEncoder(**kwargs)


# test on main
if __name__ == "__main__":
    import torch

    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    model = vq_gan_F8().cuda()
    loss_sum, output = model(input_tensor)
    print("---- output ----")
    for k, v in output.items():
        print(k, v.shape)

    from flex_gen import discriminators
    from timm.models import create_model

    disc = create_model(
        "taming_discriminator",
    ).cuda()
    loss, gen_loss = disc.get_gan_loss(
        input_tensor, output["recon"], model.get_last_dec_layer()
    )
    print("---- gen_loss ----")
    for k, v in gen_loss.items():
        print(k, v)
    print("---- disc_loss ----")
    loss, disc_loss = disc.get_disc_loss(
        input_tensor,
        output["recon"],
    )
    for k, v in disc_loss.items():
        print(k, v)
