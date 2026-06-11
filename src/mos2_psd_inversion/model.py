from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.avg_pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )
        self.se = SELayer(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.body(x))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SEResNet(nn.Module):
    def __init__(
        self,
        block: type[SEBottleneck],
        layers: list[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * block.expansion, output_dim)

    def _make_layer(
        self,
        block: type[SEBottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class GeM1d(nn.Module):
    def __init__(self, kernel_size: int, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        return F.avg_pool1d(x, self.kernel_size).pow(1.0 / self.p)


class LowFrequencyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(p=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MidFrequencyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            GeM1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            GeM1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=4),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(p=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HighFrequencyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            GeM1d(kernel_size=5),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            GeM1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            GeM1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=4),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(p=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PSDDefectRegressor(nn.Module):
    """Estimate shallow and deep defect activation energies from PSD curves."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.low_encoder = LowFrequencyEncoder()
        self.mid_encoder = MidFrequencyEncoder()
        self.high_encoder = HighFrequencyEncoder()
        self.backbone = SEResNet(SEBottleneck, [3, 4, 6, 3], self.config.output_dim)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        x = torch.clamp(x * cfg.scale_factor, min=cfg.log_epsilon)
        x = torch.log(x)
        x = (x - self.mean) / (self.std + cfg.norm_epsilon)

        low_start, low_end = cfg.low_slice
        mid_start, mid_end = cfg.mid_slice
        high_start, high_end = cfg.high_slice

        low = self.low_encoder(x[:, :, low_start:low_end]).view(-1, 64, 64)
        mid = self.mid_encoder(x[:, :, mid_start:mid_end]).view(-1, 64, 64)
        high = self.high_encoder(x[:, :, high_start:high_end]).view(-1, 64, 64)
        rgb_like = torch.stack([low, mid, high], dim=1)
        return self.backbone(rgb_like)
