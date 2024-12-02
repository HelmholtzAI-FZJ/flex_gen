from flex_gen.utils.metrics import Metrics
import flex_gen.utils.distributed as distributed
from flex_gen.datasets import (
    build_dataset,
    build_dataloader,
    build_transforms,
    build_collate_fn,
)
from flex_gen.utils import utils
from timm import create_model
from omegaconf import OmegaConf, DictConfig
from accelerate.utils import set_seed
import argparse
import os
import tqdm
from accelerate import Accelerator
import torch
from flex_gen.utils.metrics.torchmetric_codebook import CodeBookMetric
from flex_gen import autoencoders
import sys
from PIL import Image
import numpy as np


class Evaluater:
    checkpoint_path = None

    def __init__(
        self,
        image_tokenizer,
        eval_metrics,
        eval_loader,
        image_logger_config,
        accelerate_configs=None,
        num_samples=5_000,
        spatial_scale=1,
    ):
        self.eval_metrics = eval_metrics
        self.image_tokenizer = image_tokenizer
        self.num_samples = num_samples
        self.fished_samples = 0
        self.image_logger_config = image_logger_config

        accelerate_configs = {} if accelerate_configs is None else accelerate_configs
        accelerate_configs["mixed_precision"] = "bf16"
        self.accelerator = Accelerator(
            **accelerate_configs,
        )
        self.len_of_loader = len(eval_loader)
        eval_loader = self.accelerator.prepare(eval_loader)
        self.eval_loader = iter(eval_loader)

        self.image_tokenizer.to(self.accelerator.device)

        if hasattr(self.image_tokenizer, "quantizers"):
            self.codebook_metric = CodeBookMetric(
                self.image_tokenizer.quantizers.num_embed
            )
        else:
            self.codebook_metric = None

        self.spatial_scale = spatial_scale

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @torch.no_grad()
    def valid_step(
        self,
        dl_iter,
    ):

        batch = next(dl_iter)
        if len(batch["image"].shape) == 5:
            B, T, C, H, W = batch["image"].shape
            new_sample = B * T
        elif len(batch["image"].shape) == 4:
            T = 1
            new_sample = batch["image"].shape[0]

        if self.fished_samples + new_sample > self.num_samples:
            rest_samples = (self.num_samples - self.fished_samples) // T
            rest_samples = max(1, rest_samples)
            for k in batch:
                batch[k] = batch[k][:rest_samples]
        self.fished_samples += new_sample
        valid_video = batch["image"].cuda()
        with self.accelerator.autocast():
            # model_return = self.image_tokenizer(
            #     valid_video, return_recon_loss_only=True
            # )
            
            if self.spatial_scale == 1:
                quantize_ret = self.image_tokenizer.encode(valid_video)
                codes = quantize_ret["codes"]
                recon = self.image_tokenizer.decode(quantize_ret['quantized'])

            # else:
            #     scale = self.spatial_scale
            #     x = valid_video
            #     B, C, H, W = x.shape
            #     x = x.view(B, C, H // scale, scale, W // scale, scale)  # Split H and W into groups of 2
            #     x = x.permute(0, 3, 5, 1, 2, 4)
            #     x = x.contiguous().view(B*scale*scale, C , H // scale, W // scale)  # Merge channels with interleaved positions

            #     quantize_ret = self.image_tokenizer.encode(x[0:B])
            #     codes = quantize_ret["codes"]
            #     recon = self.image_tokenizer.decode(quantize_ret['quantized'])
            #     valid_video = x[0:B]
            #     # recon = recon.view(B, scale, scale, C, H // scale, W // scale)
            #     # recon = recon.permute(0, 3, 4, 1, 5, 2)
            #     # recon = recon.contiguous().view(B, C, H, W)

        real = valid_video.clamp(-1, 1)
        fake = recon.clamp(-1, 1)

        # real = utils.rescale_image_tensor(
        #     real,
        #     self.image_logger_config.rescale_mean,
        #     self.image_logger_config.rescale_std,
        # )
        # fake = utils.rescale_image_tensor(
        #     fake,
        #     self.image_logger_config.rescale_mean,
        #     self.image_logger_config.rescale_std,
        # )
        self.eval_metrics.update(real=real, fake=fake)

        # self.save_to_local(real, fake, step=self.fished_samples, save=True)

        if self.codebook_metric is not None:
            self.codebook_metric.update(codes)

    def save_to_local(self, real, fake, step=0, save=False):
        if not save:
            return

        if self.is_main:
            real_img = real[0].cpu().float().numpy().transpose(1, 2, 0)
            fake_img = fake[0].cpu().float().numpy().transpose(1, 2, 0)
            real_img = (real_img * 255).astype(np.uint8)
            fake_img = (fake_img * 255).astype(np.uint8)
            Image.fromarray(real_img).save(f"./tmp/real_{step}.png")
            Image.fromarray(fake_img).save(f"./tmp/fake_{step}.png")

        self.accelerator.wait_for_everyone()

    def evaluate(self,):
        GB = 1024.0 * 1024.0 * 1024.0
        len_of_loader = self.len_of_loader
        self.eval_metrics.reset()
        self.image_tokenizer.eval()
        last_finished = 0
        with tqdm.trange(
            0,
            self.num_samples,
            desc="Validating",
            dynamic_ncols=True,
            disable=not self.is_main,
        ) as pbar:
            for step in pbar:
                self.valid_step(self.eval_loader)
                pbar.set_postfix(
                    memory=f"{torch.cuda.max_memory_allocated() / GB :.2f} GB",
                    samples=f"{self.fished_samples}/{self.num_samples}",
                )
                new_samples = self.fished_samples - last_finished
                last_finished = self.fished_samples
                pbar.update(new_samples)
                if self.fished_samples >= self.num_samples:
                    pbar.close()
                    break

        eval_results = self.eval_metrics.compute_and_reduce()

        to_print = f"!FLAG {self.checkpoint_path} "
        for k, v in eval_results.items():
            to_print += f"|{k}:{v:.4f}"

        if self.codebook_metric is not None:
            codebook_results = self.codebook_metric.get_result()
            for k, v in codebook_results.items():
                to_print += f"|{k}:{v:.4f}"

        if self.is_main:
            print(to_print + "|")


def main(
    configs,
    checkpoint_path,
    num_samples=5_000,
    batch_size=128,
    force_resolution=-1,
    spatial_scale=1,
):
    set_seed(configs.seed)

    train_configs = configs.pop("train", DictConfig({}))
    evaluation_configs = configs.pop("evaluation", DictConfig({}))
    # evaluation_configs["metrics"] = ["ssim", "mse", "psnr", "fid", "is"]
    eval_metrics = Metrics(
        metrics_list=evaluation_configs.get("metrics", ["mse"]),
        dataset_name=evaluation_configs.get("dataset_name", None),
        device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}",
    )

    model_configs = configs.pop("model", {})
    model_name = model_configs.pop("name")
    model_configs["path"] = checkpoint_path
    model_configs["pretrained"] = True
    image_tokenizer = create_model(model_name=model_name, **model_configs)

    # build eval data loader
    eval_data_configs = cfg.pop("eval_data", None)
    dataset_name = eval_data_configs.dataset.get("path")
    eval_transforms_config = eval_data_configs.pop("transforms", None)
    eval_data_loader_configs = eval_data_configs.pop("dataloader")
    eval_data_loader_configs.batch_size = batch_size

    if force_resolution is not None and force_resolution > 0:
        eval_transforms_config.input_size = force_resolution

    eval_data_transform = build_transforms(
        eval_transforms_config, dataset_name=dataset_name
    )
    eval_dataset = build_dataset(
        eval_data_configs.pop("dataset"),
        transforms=eval_data_transform,
    )
    print(f"Toal samples: {len(eval_dataset)}")
    eval_collate_fn = build_collate_fn(eval_data_configs.pop("collate_fn", None))
    eval_loader = build_dataloader(
        eval_dataset,
        collate_fn=eval_collate_fn,
        dataloader_config=eval_data_loader_configs,
    )

    logger_configs = configs.pop("logger", {})
    image_logger_config = logger_configs.pop("image_logger", {None})

    accelerate_configs = configs.pop("accelerate", DictConfig({}))

    evaluater = Evaluater(
        image_tokenizer,
        eval_metrics,
        eval_loader,
        accelerate_configs=accelerate_configs,
        num_samples=num_samples,
        image_logger_config=image_logger_config,
        spatial_scale=spatial_scale,
    )

    res = evaluater.evaluate()
    print("done!")
    sys.exit(0)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Diffusion Evaluate", add_help=add_help)
    parser.add_argument(
        "--config-file",
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint-path",
        default=None,
        help="manually restore from a specific checkpoint directory",
    )
    parser.add_argument(
        "--num_samples",
        "--num-samples",
        default=1_000,
        type=int,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--batch_size",
        "--batch_size",
        default=128,
        type=int,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--resolution",
        default=-1,
        type=int,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--spatial_scale",
        default=1,
        type=int,
        help="number of samples to generate",
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
    return parser


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    distributed.enable(overwrite=True, dist_init=False)

    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    cfg.args = OmegaConf.create(vars(args))
    cfg = OmegaConf.create(cfg)

    main(
        cfg,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        force_resolution=args.resolution,
        spatial_scale=args.spatial_scale,
    )
