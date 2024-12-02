import torch
from accelerate import Accelerator
from flex_gen.utils.logger import Logger
import os
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class BasicTrainer:
    accelerator: Accelerator = None
    logger: Logger = None

    def __init__(self) -> None:
        pass

    @property
    def device(self):
        # if isinstance(self.model, FSDP):
        #     return self.model.module.device
        # else:
        #     return self.model.device
        return next(self.model.parameters()).device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)
