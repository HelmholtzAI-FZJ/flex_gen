from flex_gen.utils.scheduler import ConstantLRScheduler
from omegaconf import OmegaConf, DictConfig
import importlib
from flex_gen.utils.utils import get_obj_from_str


def build_lr_scheduler(optimizer, lr_scheduler_cfg=None):
    lr_scheduler_cfg = lr_scheduler_cfg or DictConfig({})
    lr_scheduler_type = lr_scheduler_cfg.pop("name", "constant").lower()

    if lr_scheduler_type == "constant":
        return ConstantLRScheduler(optimizer)

    elif lr_scheduler_type == "annealing":
        from flex_gen.utils.scheduler import AnnealingLR

        return AnnealingLR(optimizer, **lr_scheduler_cfg)
    elif lr_scheduler_type == "warmup_milestone":
        from flex_gen.utils.scheduler import WarmupMileStoneLR

        return WarmupMileStoneLR(optimizer, **lr_scheduler_cfg)

    else:
        raise ValueError(f"Unknown scheduler type {lr_scheduler_type}")


def build_image_tokenizer(model_name, path):
    from timm import create_model

    model = create_model(model_name, pretrained=False)
    # get class_obj of the model
    model_class = model.__class__
    return model_class.init_and_load_from(path)


def build_noise_scheduler(noise_scheduler_cfg):
    noise_scheduler_cfg_name = noise_scheduler_cfg.pop("name")
    scheduler_obj = get_obj_from_str(noise_scheduler_cfg_name)
    return scheduler_obj(**noise_scheduler_cfg)
