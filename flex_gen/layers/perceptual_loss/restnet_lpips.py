"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from collections import namedtuple
from torchvision.transforms import RandomCrop, Normalize
import requests
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from timm import create_model

"""
CHANGELOG:
From magvit and maskbit paper, seems like to take intermidiate features is not necessary?
"""


class ResNet_LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(
        self,
        base_model="resnet50.a1_in1k",
    ):
        super().__init__()

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        resnet = create_model(base_model, pretrained=True)

        self.backbone = resnet.eval().requires_grad_(False)
        self.img_resolution = 224

        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        # LPIPS is always in eval mode
        self.backbone.train(False)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, input, target):

        input = self.normalize(input)
        target = self.normalize(target)
        in0_input = nn.functional.interpolate(input, self.img_resolution, mode="area")
        in1_input = nn.functional.interpolate(target, self.img_resolution, mode="area")

        outs0 = self.backbone(in0_input)
        outs1 = self.backbone(in1_input)
        sum_distance = torch.nn.functional.mse_loss(outs0, outs1)
        return sum_distance
