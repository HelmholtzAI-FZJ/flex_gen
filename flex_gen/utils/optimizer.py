# https://github.com/lucidrains/magvit2-pytorch/blob/main/magvit2_pytorch/optimizer.py
from torch.optim import AdamW, Adam
import os


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-2,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=False,
    # group_wd_params=True,
    **kwargs,
):
    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    opt_kwargs = dict(lr=lr, betas=betas, eps=eps)
    opt_kwargs = {"weight_decay": wd, **opt_kwargs}
    return AdamW(params, **opt_kwargs)


def apply_lr_scale(
    optim_config,
    train_batch_size=1,
    grad_accum_every=1,
):
    if "base_lr" not in optim_config and "lr" not in optim_config:
        raise ValueError("Learning rate not provided in optimizer config")

    lr_scale = optim_config.pop("lr_scale", "linear")

    if (
        "base_lr" in optim_config
        and "WORLD_SIZE" in os.environ
        and lr_scale == "linear"
    ):
        if int(os.environ["RANK"]) == 0:
            print(optim_config)
        base_lr = optim_config.pop("base_lr")
        world_size = int(os.environ["WORLD_SIZE"])
        lr = base_lr * train_batch_size * grad_accum_every * world_size
        optim_config.lr = lr
        if int(os.environ["RANK"]) == 0:
            print(f"! Applying Linear Learning Rate Scaling !")
            print(
                f" LR  {lr:.6f} = Base_LR * train_BS * GA * World SIZE: {base_lr:.6f} * {train_batch_size} * {grad_accum_every} * {world_size}"
            )
    elif lr_scale == "fixed":
        base_lr = optim_config.pop("base_lr")
        optim_config.lr = base_lr
        if int(os.environ["RANK"]) == 0:
            print(f"! Applying Constact Rate Scaling !")
            print(f" LR  {base_lr:.6f} = Base_LR : {base_lr:.6f} ")
    else:
        assert "lr" in optim_config, "Learning rate not provided in optimizer config"
    return optim_config
