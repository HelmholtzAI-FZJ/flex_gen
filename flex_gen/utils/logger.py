import os
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
import wandb
from pathlib import Path
from flex_gen.utils import utils
from PIL import Image
from einops import rearrange
import torch
import torchvision
import numpy as np
from flex_gen.datasets import DATASET_CONFIG
import os

DEFAULT_IMAGE_LOGGER_CONFIG = DictConfig(
    {
        "img_log_iter_frequency": 100,
        "max_images_to_log": 1,
        "clamp_log_images": True,
        "rescale_log_images": False,
        "save_on_local": False,
        "save_on_wandb": False,
    }
)


class Logger:
    _lazy_init = True
    image_save_dir = None
    models_to_wach = []

    def __init__(self, logger_cfg, global_rank=0, full_configs=None):
        self.output_dir = Path(logger_cfg.get("output_dir", "logs_test"))
        self.global_rank = int(global_rank)
        self.full_configs = full_configs
        self.checkpoint_every_step = logger_cfg.get("checkpoint_every_step", 1000)

        # Image logger
        self.image_logger_config = OmegaConf.merge(
            DEFAULT_IMAGE_LOGGER_CONFIG, logger_cfg.image_logger
        )
        self.enable_image_logging = (
            self.image_logger_config.save_on_local
            or self.image_logger_config.save_on_wandb
        )
        self.save_images_on_local = self.image_logger_config.save_on_local
        self.max_images_to_log = self.image_logger_config.max_images_to_log
        self.setup_rescale_factor_for_images(logger_cfg)
        self.logger_cfg = logger_cfg

        self.metrics_writter_buffer = ""

    def sync_output_dir(self):
        if not (dist.is_available() and dist.is_initialized()):
            raise ValueError("Distributed is not initialized")

        if self.global_rank == 0:
            broadcast_output_dir = [self.output_dir]
        else:
            broadcast_output_dir = [None]
        dist.broadcast_object_list(
            broadcast_output_dir,
            src=0,
        )
        assert broadcast_output_dir[0] is not None
        self.output_dir = broadcast_output_dir[0]
        self.logger_cfg.output_dir = self.output_dir

    def setup_logger(self):
        self._lazy_init = False
        self.sync_output_dir()
        if self.global_rank == 0:
            # create the output directory
            os.makedirs(self.output_dir, exist_ok=True)
            # save the config file
            with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
                OmegaConf.save(
                    config=self.full_configs,
                    f=f,
                )
            # create the wandb directory

            wandb_dir = os.path.join(self.output_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)

            self.metrics_save_dir = os.path.join(self.output_dir, "metrics.csv")
            self.metrics_writer_header = open(self.metrics_save_dir, "w")

        dist.barrier()
        self.setup_tracker()
        dist.barrier()

    def setup_tracker(self):
        if self.global_rank == 0:
            if "wandb" in self.logger_cfg:
                default_name = os.path.basename(self.get_output_dir())
                wandb_cache_dir = self.output_dir / "wandb"
                wandb_config_dict = self.logger_cfg.wandb
                wandb_config_dict.name = wandb_config_dict.get("name", default_name)
                wandb_config_dict.dir = wandb_config_dict.get("dir", wandb_cache_dir)
                wandb.init(**wandb_config_dict)
                wandb.config.update(
                    OmegaConf.to_container(self.full_configs, resolve=True)
                )

    def get_output_dir(self):
        if self._lazy_init:
            self.setup_logger()
        return self.output_dir

    def get_directory(self, query_dir):
        if self._lazy_init:
            self.setup_logger()
        dir = self.output_dir / query_dir
        os.makedirs(dir, exist_ok=True)
        return dir

    def setup_rescale_factor_for_images(self, logger_cfg):
        dataset_name = logger_cfg.get("dataset_name", "").lower()
        if "rescale_mean" not in self.image_logger_config:
            self.image_logger_config.rescale_mean = (
                DATASET_CONFIG.query_mean_from_dataset_name(dataset_name)
            )
        if "rescale_std" not in self.image_logger_config:
            self.image_logger_config.rescale_std = (
                DATASET_CONFIG.query_std_from_dataset_name(dataset_name)
            )

    def log(self, data_kwargs, step, commit=False):
        if self.metrics_writter_buffer == "":
            self.metrics_writter_buffer += f"Step: {step}, "

        for k, v in data_kwargs.items():
            if isinstance(v, torch.Tensor):
                # v = (v.detach() * 1.0).mean()
                data_kwargs[k] = torch.mean(v * 1.0).item()

            if k != "step":
                self.metrics_writter_buffer += f"{k}: {data_kwargs[k]}, "
        # self.accelerator.log(data_kwargs, step=step)
        if self.global_rank == 0:
            wandb.log(data_kwargs, step=step, commit=commit)
            if commit:
                self.metrics_writter_buffer = self.metrics_writter_buffer[:-1] + "\n"
                if int(step) % 100 == 0:
                    self.metrics_writer_header.write(self.metrics_writter_buffer)
                    self.metrics_writer_header.flush()
                self.metrics_writter_buffer = ""

    def log_image(self, images, img_name, step, is_video=False, prefix="", suffix=""):
        # image_np: (B, C, H, W) in [0, 1]
        images = images.clamp(0, 1)
        grid = torchvision.utils.make_grid(images, nrow=4)  # c,h,w
        grid = grid.permute(1, 2, 0).numpy() * 255  # h,w,c
        gird = Image.fromarray(grid.astype(np.uint8), mode="RGB")
        if self.image_logger_config.save_on_local:
            gird.save(
                self.image_save_dir / f"{prefix}{step:06d}_{img_name}{suffix}.png"
            )

        if (
            self.image_logger_config.save_on_wandb
            and self.global_rank == 0
            and not is_video
        ):
            wandb.log({img_name: wandb.Image(gird)}, step=step, commit=False)

    def log_heatmap(self, img, img_name, step, prefix="", suffix=""):
        heat_map_img = torch.Tensor(
            utils.convert_tensor_to_heatmap(img)
        )  # get heatmap, but B, H, W, C
        self.log_image(
            rearrange(heat_map_img, "B H W C -> B C H W"),
            img_name,
            step,
            prefix,
            suffix,
        )

    def rescale_image_tensor(self, img):
        return utils.rescale_image_tensor(
            img,
            self.image_logger_config.rescale_mean,
            self.image_logger_config.rescale_std,
        )

    def log_ae_training_images(
        self, images_dict, step, prefix="", suffix="", rescale=True
    ):
        """
        data, recon, codes, sample
        """
        if not self.enable_image_logging:
            return
        if self.image_save_dir is None:
            self.image_save_dir = self.get_directory("images")
        if step % self.image_logger_config.img_log_iter_frequency == 0:
            for img_name, img in images_dict.items():
                if img_name == "codes":
                    pass
                    # self.log_heatmap(
                    #     img[0 : self.max_images_to_log].detach().cpu(),
                    #     img_name,
                    #     step,
                    #     prefix,
                    #     suffix,
                    # )
                elif img_name in ["data", "recon", "sample"]:
                    img = img[0 : self.max_images_to_log].detach().cpu().float()
                    if rescale:
                        img = utils.rescale_image_tensor(
                            img[0 : self.max_images_to_log],
                            self.image_logger_config.rescale_mean,
                            self.image_logger_config.rescale_std,
                        )

                    self.log_image(
                        img,
                        img_name,
                        step,
                        False,
                        prefix,
                        suffix,
                    )

    def log_ldm_training_images(
        self, images_dict, step, prefix="", suffix="", rescale=True
    ):
        """
        data, recon, codes, sample
        """
        if not self.enable_image_logging:
            return
        if self.image_save_dir is None:
            self.image_save_dir = self.get_directory("images")
        if step % self.image_logger_config.img_log_iter_frequency == 0:
            for img_name, img in images_dict.items():
                if len(img.shape) == 4:
                    img = img[0 : self.max_images_to_log].detach().cpu().float()

                    if rescale:
                        img = utils.rescale_image_tensor(
                            img,
                            self.image_logger_config.rescale_mean,
                            self.image_logger_config.rescale_std,
                        )

                    self.log_image(img, img_name, step, False, prefix, suffix)

                elif len(img.shape) == 5:
                    # other wise, we only show 1 sample
                    img = img.detach().cpu().float()
                    # We only show and upload the first 2 videos to wandb
                    img = [
                        utils.rescale_image_tensor(
                            single_video,
                            self.image_logger_config.rescale_mean,
                            self.image_logger_config.rescale_std,
                        )
                        for single_video in img[:2]
                    ]

                    if self.image_logger_config.save_on_wandb and self.global_rank == 0:
                        wandb.log(
                            {
                                img_name: wandb.Video(
                                    (np.stack(img) * 255).astype(np.uint8), fps=1
                                )
                            },
                            step=step,
                            commit=False,
                        )
                    # we only plot the first video on the local disk
                    self.log_image(img[0], img_name, step, True, prefix, suffix)
