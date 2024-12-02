# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import os
import re
from omegaconf import OmegaConf, DictConfig
from collections.abc import Iterable
import flex_gen.utils.distributed as distributed
import torch.distributed as dist


def name_output_dir(output_dir):
    if output_dir == "":
        output_dir = "logs_default"

    if os.path.exists(output_dir):
        output_dir_parent = os.path.dirname(output_dir)
        output_dir_name = os.path.basename(output_dir)
        if "_EXP_" not in output_dir:
            output_dir_name = output_dir_name + "_EXP_2"

        prefix, suffix = output_dir_name.split("_EXP_")
        try:
            all_folders = os.listdir(output_dir_parent)
            all_folders = [re.search(rf"{prefix}_EXP_(\d+)", _) for _ in all_folders]
            all_folders = [1 if _ is None else int(_.group(1)) for _ in all_folders]
            suffix = max(all_folders) + 1
        except:
            suffix = 2
        output_dir = prefix + "_EXP_" + str(suffix)
        output_dir = os.path.join(output_dir_parent, output_dir)
        if os.path.exists(output_dir):
            prefix, suffix = output_dir.split("_EXP_")
            output_dir = prefix + "_EXP_" + str(int(suffix) + 1)

    return output_dir


def get_cfg_from_args(args):
    args.output_dir = name_output_dir(args.output_dir)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    cfg.args = OmegaConf.create(vars(args))
    return OmegaConf.create(cfg)


def default_setup(args, cfg):
    # === To broadcase the output_dir to all nodes ===
    # if distributed.is_main_process() and distributed.get_local_rank() == 0:
    #     broadcast_output_dir = [args.output_dir]
    # else:
    #     broadcast_output_dir = [None]
    # dist.broadcast_object_list(
    #     broadcast_output_dir,
    #     src=0,
    # )
    # assert broadcast_output_dir[0] is not None
    # args.output_dir = broadcast_output_dir[0]
    # If we use Acclerator, we need to move this "broadcast" inside the Logger.init

    if "logger" not in cfg or cfg.logger is None:
        cfg.logger = DictConfig({"output_dir": args.output_dir})
    else:
        cfg.logger.output_dir = args.output_dir
    # os.makedirs(args.output_dir, exist_ok=True)
    # === To broadcase the output_dir to all nodes ===

    if hasattr(args, "seed"):
        seed = int(
            getattr(
                args,
                "seed",
            )
        )
        cfg.seed = seed
    elif not hasattr(cfg, "seed"):
        cfg.seed = -1

    if cfg.seed < 0:
        from time import time

        seed = int(time())
    else:
        seed = cfg.seed
    cfg.seed = seed


def setup_configs(args):
    cfg = get_cfg_from_args(args)
    # os.makedirs(args.output_dir, exist_ok=True)
    # Note - I moved the output_dir creation to default_setup to ensure all nodes create the same folder
    default_setup(args, cfg)
    return cfg
