from .carry_vector_quantizer import CarryVectorQuantizer
from .ema_vector_quantizer import EMAVectorQuantizer
from .finite_scalar_quantizer import FiniteScalarQuantizer
from .group_vector_quantizer import GroupVectorQuantizer
from .group_sphere_quantizer import GroupSphereQuantizer
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .hierachical_vector_quantizer import HierachicalVectorQuantizer
from .residual_vector_quantizer import ResidualVectorQuantizer
from .lookup_free_quantizer import LookUpFreeQuantizer
from .vector_quantizer import VectorQuantizer
from omegaconf import DictConfig
from einops import rearrange

QuantizeModelDict = {
    "fsq": FiniteScalarQuantizer,
    "lfq": LookUpFreeQuantizer,
    "gumbelvq": GumbelVectorQuantizer,
    "vq": VectorQuantizer,
    "emavq": EMAVectorQuantizer,
    "gvq": GroupVectorQuantizer,
    'gsq': GroupSphereQuantizer,
    "hvq": HierachicalVectorQuantizer,
    "cvq": CarryVectorQuantizer,
    "rvq": ResidualVectorQuantizer,
}


def patch_vq_forward(vq_forward):
    if getattr(vq_forward, "_is_patched", False):
        # If the function is already patched, return it as is
        return vq_forward
    def wrapper(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        try_get_results = vq_forward(self=self, x=x)
        assert len(try_get_results) == 3
        # the return of vq_forward should follows
        # x_quant, diff, indice -> x_quant, loss, codes
        x_quant, loss, codes = try_get_results
        return {
            "quantized": rearrange(x_quant, "b h w c -> b c h w").contiguous(),
            "aux_loss": loss,
            "codes": codes,
        }
    wrapper._is_patched = True
    return wrapper


class AnyQuantizer:
    @staticmethod
    def build_quantizer(config):
        if isinstance(config, dict):
            config = DictConfig(config)
        quantize_type = config.pop("quantize_type").lower()
        assert (
            quantize_type in QuantizeModelDict
        ), f"quantize_type {quantize_type} not found"
        config = AnyQuantizer.check_quantize_config(config)
        vq_model = QuantizeModelDict[quantize_type]
        vq_model.forward = patch_vq_forward(vq_model.forward)
        return vq_model(**config)

    @staticmethod
    def check_quantize_config(config):
        return config
