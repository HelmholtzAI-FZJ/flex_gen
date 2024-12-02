from functools import partial
from typing import List
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
import math

ConstantLRScheduler = partial(LambdaLR, lr_lambda=lambda step: 1.0)


class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ["linear", "cosine", "exponential", "constant", "None"]

    def __init__(
        self,
        optimizer,
        start_lr,
        warmup_iter,
        num_iters,
        decay_style=None,
        last_iter=-1,
        decay_ratio=0.5,
        restart_iter=0,
    ):
        self.restart_iter = restart_iter
        assert warmup_iter <= num_iters
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None

        if decay_ratio == 0:
            self.decay_ratio = 1e12
            self.real_decay_ratio = 0
        else:
            self.decay_ratio = 1 / decay_ratio
            self.real_decay_ratio = decay_ratio
        self.step(self.num_iters)

    def get_last_lr(self):
        return [self.get_lr()]

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        real_num_iters = self.num_iters - self.restart_iter
        real_end_iter = self.end_iter - self.restart_iter
        if self.warmup_iter > 0 and real_num_iters <= self.warmup_iter:
            return float(self.start_lr) * real_num_iters / self.warmup_iter
        else:
            if self.decay_style == self.DECAY_STYLES[0]:
                lr = self.start_lr * (
                    (real_end_iter - (real_num_iters - self.warmup_iter))
                    / real_end_iter
                )

                return lr
            elif self.decay_style == self.DECAY_STYLES[1]:
                decay_step_ratio = min(
                    1.0, (real_num_iters - self.warmup_iter) / real_end_iter
                )

                if (
                    self.real_decay_ratio == 0
                    and math.cos(math.pi * decay_step_ratio) == -1
                ):
                    return 0

                lr = (
                    self.start_lr
                    / self.decay_ratio
                    * (
                        (math.cos(math.pi * decay_step_ratio) + 1)
                        * (self.decay_ratio - 1)
                        / 2
                        + 1
                    )
                )
                return lr
            elif self.decay_style == self.DECAY_STYLES[2]:
                # TODO: implement exponential decay
                return self.start_lr
            else:
                return self.start_lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def state_dict(self):
        sd = {
            # 'start_lr': self.start_lr,
            "warmup_iter": self.warmup_iter,
            "num_iters": self.num_iters,
            "decay_style": self.decay_style,
            "end_iter": self.end_iter,
            "decay_ratio": self.decay_ratio,
        }
        return sd

    def load_state_dict(self, sd):
        # self.start_lr = sd['start_lr']
        self.warmup_iter = sd["warmup_iter"]
        self.num_iters = sd["num_iters"]
        self.end_iter = sd["end_iter"]
        self.decay_style = sd["decay_style"]
        if "decay_ratio" in sd:
            self.decay_ratio = sd["decay_ratio"]
        self.step(self.num_iters)


class WarmupMileStoneLR(_LRScheduler):
    def __init__(self, optimizer, start_lr, milestones, warmup_iter=0, gamma=0.1):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.milestones = milestones
        self.gamma = gamma
        self.num_iters = 0
        self.step(0)

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def get_lr(self):
        real_num_iters = self.num_iters

        if self.warmup_iter > 0 and real_num_iters <= self.warmup_iter:
            return float(self.start_lr) * real_num_iters / self.warmup_iter
        else:
            return self.start_lr * self.gamma ** sum(
                [real_num_iters > m for m in self.milestones]
            )

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        sd = {
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "milestones": self.milestones,
            "gamma": self.gamma,
            "num_iters": self.num_iters,
        }
        return sd

    def load_state_dict(self, sd):
        self.warmup_iter = sd["warmup_iter"]
        self.milestones = sd["milestones"]
        self.gamma = sd["gamma"]
        self.start_lr = sd["start_lr"]
        self.num_iters = sd["num_iters"]
        self.step(self.num_iters)
