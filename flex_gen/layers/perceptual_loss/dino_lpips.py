"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import hashlib
import os
from collections import namedtuple
from torchvision.transforms import RandomCrop, Normalize
import requests
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

CKPT_MAP = {"vgg_lpips": "vgg.pth"}

MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


"""
CHANGELOG:
From magvit and maskbit paper, seems like to take intermidiate features is not necessary?
"""


class DINO_LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(
        self,
        base_model="dinov2_vits14_reg",
    ):
        super().__init__()

        dino = torch.hub.load("facebookresearch/dinov2", base_model)
        self.dino = dino.eval().requires_grad_(False)
        self.img_resolution = self.dino.patch_embed.img_size
        # normlize by imagenet mean and std
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        # LPIPS is always in eval mode
        self.dino.train(False)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, input, target):
        input = self.normalize(input)
        target = self.normalize(target)

        in0_input = nn.functional.interpolate(input, self.img_resolution, mode="area")
        in1_input = nn.functional.interpolate(target, self.img_resolution, mode="area")

        outs0 = self.dino(in0_input)
        outs1 = self.dino(in1_input)
        sum_distance = torch.nn.functional.mse_loss(outs0, outs1)
        return sum_distance
