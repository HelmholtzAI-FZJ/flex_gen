from flex_gen.trainer.VideoTokenizerTrainer import VideoTokenizerTrainer
from flex_gen.utils.logger import Logger
from flex_gen.utils.config import setup_configs
from flex_gen.utils.optimizer import get_optimizer, apply_lr_scale
from flex_gen.utils.builder import build_lr_scheduler
from flex_gen.utils.metrics import Metrics
from flex_gen.algorithms.ema.ema import update_ema
import flex_gen.utils.distributed as distributed
from flex_gen.datasets import build_dataset, build_dataloader, build_transforms
from timm import create_model
import copy
from omegaconf import DictConfig
from accelerate.utils import set_seed
import argparse
import os
from functools import partial
import torch


def main(configs, checkpoint_path=None, auto_resume=False):
    copy_of_configs = copy.deepcopy(configs)
    set_seed(configs.pop("seed", 42))

    logger_configs = configs.pop("logger", {})
    logger = Logger(
        logger_configs,
        global_rank=os.environ.get("RANK", 0),  # since dist is not yet inited
        full_configs=copy_of_configs,
    )

    # training configs
    train_configs = configs.pop("train", DictConfig({}))
    num_train_steps = train_configs.pop("num_train_steps", 100)
    grad_accum_every = train_configs.pop("grad_accum_every", 1)
    apply_gradient_penalty_every = train_configs.pop("apply_gradient_penalty_every", 1)
    max_grad_norm = train_configs.pop("max_grad_norm", 1.0)
    discr_start_after_step = train_configs.pop("discr_start_after_step", 0)
    train_batch_size = train_configs.pop("train_batch_size", 1)

    # validation configs
    evaluation_configs = configs.pop("evaluation", DictConfig({}))
    eval_every_step = evaluation_configs.get("eval_every_step", -1)
    eval_every_step = evaluation_configs.get("eval_every_step", -1)
    eval_for_steps = evaluation_configs.get("eval_for_steps", 1)
    eval_metrics = Metrics(
        metrics_list=evaluation_configs.get("metrics", ["mse"]),
        dataset_name=evaluation_configs.get("dataset_name", None),
        device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}",
    )

    # model and ema model
    model_configs = configs.pop("model", {})
    model_name = model_configs.pop("name")
    assert model_name is not None, "Model name is not provided"

    model = create_model(
        model_name,
        **model_configs,
    )
    print(f" *** Model: {model} ***")
    print(
        f" *** Parameters {(model_name)}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M ***"
    )

    ema_configs = configs.pop("ema", {})
    ema_update_fn = partial(update_ema, **ema_configs)

    # discriminator
    disc_configs = configs.pop("discriminator", {})
    disc_name = disc_configs.pop("name")
    assert disc_name is not None, "Discriminator name is not provided"
    disc_model = create_model(disc_name, **disc_configs)
    # generally we don't need EMA for discriminator
    # ema_disc = EMA(disc_model, **ema_configs)

    print(f" *** Model: {disc_model} ***")

    # optimizer
    optim_configs = configs.pop("optim", {})
    disc_optim_configs = configs.pop("disc_optim", {})
    optim_configs = apply_lr_scale(optim_configs, train_batch_size, grad_accum_every)
    disc_optim_configs = apply_lr_scale(
        disc_optim_configs, train_batch_size, grad_accum_every
    )
    optimizer = get_optimizer(model.parameters(), **optim_configs)
    disc_optimizer = get_optimizer(disc_model.parameters(), **disc_optim_configs)

    # lr scheduler
    lr_scheduler_configs = configs.pop("lr_scheduler", None)
    disc_lr_scheduler_configs = configs.pop("disc_lr_scheduler", None)
    lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler_configs)
    disc_lr_scheduler = build_lr_scheduler(disc_optimizer, disc_lr_scheduler_configs)

    # data loaders
    train_data_configs = cfg.pop("train_data")
    dataset_name = train_data_configs.dataset.get("path")
    transforms_config = train_data_configs.pop("transforms", None)
    data_loader_configs = train_data_configs.pop("dataloader")

    data_transform = build_transforms(
        transforms_config,
        dataset_name=dataset_name,
    )
    dataset = build_dataset(
        train_data_configs.pop("dataset"),
        transforms=data_transform,
    )
    collate_fn = None
    train_loader = build_dataloader(
        dataset, collate_fn=collate_fn, dataloader_config=data_loader_configs
    )
    print(f" *** Train data loader: {len(train_loader)} ***")

    # build eval data loader
    eval_data_configs = cfg.pop("eval_data", None)
    dataset_name = eval_data_configs.dataset.get("path")
    eval_transforms_config = eval_data_configs.pop("transforms", None)
    eval_data_loader_configs = eval_data_configs.pop("dataloader")

    eval_data_transform = build_transforms(
        eval_transforms_config, dataset_name=dataset_name
    )
    eval_dataset = build_dataset(
        eval_data_configs.pop("dataset"),
        transforms=eval_data_transform,
    )
    eval_collate_fn = None
    eval_loader = build_dataloader(
        eval_dataset,
        collate_fn=eval_collate_fn,
        dataloader_config=eval_data_loader_configs,
        # use_distributed_sampler=True,
    )
    print(f" *** Eval data loader: {len(eval_loader)} ***")

    # extra configs for acceleration
    accelerate_configs = configs.pop("accelerate", DictConfig({}))

    trainer = VideoTokenizerTrainer(
        model=model,
        ema_update_fn=ema_update_fn,
        discriminator=disc_model,
        optimizer=optimizer,
        disc_optimizer=disc_optimizer,
        lr_scheduler=lr_scheduler,
        disc_lr_scheduler=disc_lr_scheduler,
        train_loader=train_loader,
        logger=logger,
        eval_metrics=eval_metrics,
        num_train_steps=num_train_steps,
        grad_accum_every=grad_accum_every,
        apply_gradient_penalty_every=apply_gradient_penalty_every,
        max_grad_norm=max_grad_norm,
        discr_start_after_step=discr_start_after_step,
        eval_loader=eval_loader,
        eval_every_step=eval_every_step,
        eval_for_steps=eval_for_steps,
        accelerate_configs=accelerate_configs,
        **train_configs,
    )

    if checkpoint_path is not None:
        trainer.load(checkpoint_path)
    elif auto_resume:
        raise NotImplementedError
        trainer.resume_checkpoint()

    trainer.fit()


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("F-MAE training", add_help=add_help)
    parser.add_argument(
        "--config-file",
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint-path",
        default=None,
        help="manually restore from a specific checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="logs_TEST",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    distributed.enable(overwrite=True, dist_init=False)
    cfg = setup_configs(args)
    main(cfg, checkpoint_path=args.checkpoint_path, auto_resume=args.auto_resume)
