# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Network and image transforms for the room classifier (shared by training and inference)."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics

ARCHS = {
    # name: (constructor, pretrained weights enum, attribute path of the final layer)
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained backbone with a fresh classification layer for our classes."""
    ctor, weights = ARCHS[arch]
    m = ctor(weights=weights if pretrained else None)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch.startswith("efficientnet"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch.startswith("convnext"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


class RandomRot90:
    """Floor plans have no 'up': a room rotated by 90 degrees is still the same room."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img if k == 0 else img.transpose([None, Image.Transpose.ROTATE_90,
                                                 Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270][k])


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        RandomRot90(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.3),   # Danish listing plans are often grey/beige; CubiCasa ones are colourful
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_checkpoint(path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device), ckpt
