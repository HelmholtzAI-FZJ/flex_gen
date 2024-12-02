from flex_gen.autoencoders.flexTokenizer import FlexTokenizer
import torch

vq_vae_f8 = {
    "input_channels": 3,
    "hidden_channels": 128,
    "encoder_layer_configs": (2, 2, 2, 2, 2),
    "decoder_layer_configs": (2, 2, 2, 2, 2),
    "channel_multipliers": (1, 2, 4, 4),
    "mid_block_attn": False,
    "attention_resolutions": (),
    "embedding_dim": 4,
    "use_adaptive_norm": False,
    "use_learnable_up_down_sample": False,
    "residual_conv_kernel_size": 3,
    "input_conv_kernel_size": 3,
    "output_conv_kernel_size": 3,
    "dropout_probability": 0.0,
    "quantizer_config": {
        "quantize_type": "vq",
        "embed_dim": 4,
        "num_embed": 16384,
    },
}

vq_vae_f8_adaptive_norm_depth2space = {
    "input_channels": 3,
    "hidden_channels": 128,
    "encoder_layer_configs": (2, 2, 2, 2, 2),
    "decoder_layer_configs": (2, 2, 2, 2, 2),
    "channel_multipliers": (1, 2, 4, 4),
    "mid_block_attn": False,
    "attention_resolutions": (),
    "embedding_dim": 4,
    "use_adaptive_norm": True,
    "use_learnable_up_down_sample": True,
    "residual_conv_kernel_size": 3,
    "input_conv_kernel_size": 3,
    "output_conv_kernel_size": 3,
    "dropout_probability": 0.0,
    "quantizer_config": {
        "quantize_type": "vq",
        "embed_dim": 4,
        "num_embed": 16384,
    },
}

vq_vae_f16 = {
    "input_channels": 3,
    "hidden_channels": 128,
    "encoder_layer_configs": (2, 2, 2, 2, 2, 2),
    "decoder_layer_configs": (2, 2, 2, 2, 2, 2),
    "channel_multipliers": (1, 2, 2, 4, 4),
    "mid_block_attn": False,
    "attention_resolutions": (),
    "embedding_dim": 4,
    "use_adaptive_norm": False,
    "use_learnable_up_down_sample": False,
    "residual_conv_kernel_size": 3,
    "input_conv_kernel_size": 3,
    "output_conv_kernel_size": 3,
    "dropout_probability": 0.0,
    "quantizer_config": {
        "quantize_type": "vq",
        "embed_dim": 4,
        "num_embed": 16384,
    },
}


vq_vae_f8_attentoin = {
    "input_channels": 3,
    "hidden_channels": 128,
    "encoder_layer_configs": (2, 2, 2, 2, 2),
    "decoder_layer_configs": (2, 2, 2, 2, 2),
    "channel_multipliers": (1, 2, 4, 4),
    "mid_block_attn": True,
    "attention_resolutions": (1,),
    "embedding_dim": 4,
    "use_adaptive_norm": False,
    "use_learnable_up_down_sample": False,
    "residual_conv_kernel_size": 3,
    "input_conv_kernel_size": 3,
    "output_conv_kernel_size": 3,
    "dropout_probability": 0.0,
    "quantizer_config": {
        "quantize_type": "vq",
        "embed_dim": 4,
        "num_embed": 16384,
    },
}

vq_vae_f8_attentoin_adaptive_norm_depth2space = {
    "input_channels": 3,
    "hidden_channels": 128,
    "encoder_layer_configs": (2, 2, 2, 2, 2),
    "decoder_layer_configs": (2, 2, 2, 2, 2),
    "channel_multipliers": (1, 2, 4, 4),
    "mid_block_attn": True,
    "attention_resolutions": (1,),
    "embedding_dim": 4,
    "use_adaptive_norm": True,
    "use_learnable_up_down_sample": True,
    "residual_conv_kernel_size": 3,
    "input_conv_kernel_size": 3,
    "output_conv_kernel_size": 3,
    "dropout_probability": 0.0,
    "quantizer_config": {
        "quantize_type": "vq",
        "embed_dim": 4,
        "num_embed": 16384,
    },
}


magvit_f8 = {
    "input_channels": 3,
    "hidden_channels": 128,
    "encoder_layer_configs": (4, 3, 4, 3, 4),
    "decoder_layer_configs": (4, 3, 4, 3, 4),
    "channel_multipliers": (1, 2, 2, 4),
    "mid_block_attn": False,
    "attention_resolutions": (),
    "embedding_dim": 4,
    "use_adaptive_norm": True,
    "use_learnable_up_down_sample": True,
    "residual_conv_kernel_size": 3,
    "input_conv_kernel_size": 3,
    "output_conv_kernel_size": 3,
    "dropout_probability": 0.0,
    "quantizer_config": {
        "quantize_type": "vq",
        "embed_dim": 4,
        "num_embed": 16384,
    },
}

# 4,3,4,3,4
# 4,2,2,1,0


def test_vq_vae():
    input = torch.randn(2, 3, 256, 256).cuda()
    print("Testing VQ-VA-8 ======================")
    model = FlexTokenizer(**vq_vae_f8).cuda()
    print(f"Parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    loss, ret = model(input)
    print("shape", ret["recon"].shape)
    del model

    # print("Testing VQ-VAE-16 ======================", input.shape)
    # model = FlexTokenizer(**vq_vae_f16).cuda()
    # print(f"Parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    # loss, ret = model(input)
    # print("shape", ret["recon"].shape)
    # del model

    # print("Testing VQ-VAE with Adaptive Norm and DepthToSpace ======================")
    # model = FlexTokenizer(**vq_vae_f8_adaptive_norm_depth2space).cuda()
    # loss, ret = model(input)
    # print("shape", ret["recon"].shape)
    # print(f"Parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # print("Testing MagViT ======================")
    # model = FlexTokenizer(**magvit_f8).cuda()
    # print(model)
    # print(f"Parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    # loss, ret = model(input)
    # print("shape", ret["recon"].shape)
    # del model


if __name__ == "__main__":
    test_vq_vae()
