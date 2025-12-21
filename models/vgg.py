from __future__ import annotations

from collections import OrderedDict, defaultdict
import torch
from torch import nn


class VGG(nn.Module):
    """
    VGG-style CNN used for CIFAR-10-like inputs (32x32 RGB).
    Backbone output: [N, 512, 2, 2] -> avgpool -> flatten -> linear classifier.
    """
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != "M":
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                add("pool", nn.MaxPool2d(2))

        add("avgpool", nn.AvgPool2d(2))
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)        # [N, 512, 2, 2]
        x = x.view(x.shape[0], -1)  # [N, 512]
        x = self.classifier(x)      # [N, 10]
        return x
