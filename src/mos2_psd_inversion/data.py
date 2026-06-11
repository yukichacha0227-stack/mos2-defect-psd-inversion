from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, ModelConfig, TrainConfig


class NoisyPSDDataset(Dataset):
    """Apply multiplicative noise to clean PSD curves on each access."""

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        config: DataConfig | None = None,
        augment: bool = True,
    ) -> None:
        self.inputs = inputs.float()
        self.targets = targets.float()
        self.config = config or DataConfig()
        self.augment = augment
        self.frequency_axis = torch.linspace(1, self.config.sequence_length, self.config.sequence_length)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        psd = self.inputs[index]
        if self.augment:
            psd = psd * (1.0 + self._processed_noise().to(psd.device))
            psd = torch.clamp(psd, min=1e-30)
        return psd, self.targets[index]

    def _processed_noise(self) -> torch.Tensor:
        cfg = self.config
        variance = cfg.noise_variance_base + cfg.noise_variance_slope * self.frequency_axis / cfg.noise_variance_scale
        variance = torch.clamp(variance, max=cfg.noise_variance_max)
        noise = torch.normal(mean=0.0, std=torch.sqrt(variance))
        clipped = torch.tanh(noise / cfg.noise_clip_range) * cfg.noise_clip_range
        return clipped * (1.0 - cfg.noise_smoothing_factor) + noise * cfg.noise_smoothing_factor


def split_train_val(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: DataConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = config or DataConfig()
    train_size = int(len(inputs) * cfg.train_fraction)
    return inputs[:train_size], targets[:train_size], inputs[train_size:], targets[train_size:]


def compute_input_stats(
    train_inputs: torch.Tensor,
    model_config: ModelConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_cfg = model_config or ModelConfig()
    train_inputs = train_inputs.float()
    transformed = torch.clamp(
        train_inputs * model_cfg.scale_factor,
        min=model_cfg.log_epsilon,
    )
    transformed = torch.log(transformed)
    mean = transformed.mean(dim=(0, 2), keepdim=True)
    std = transformed.std(dim=(0, 2), keepdim=True)
    return mean, std


def make_dataloaders(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    data_config: DataConfig | None = None,
    train_config: TrainConfig | None = None,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    data_cfg = data_config or DataConfig()
    train_cfg = train_config or TrainConfig()
    train_x, train_y, val_x, val_y = split_train_val(inputs, targets, data_cfg)
    mean, std = compute_input_stats(train_x)
    train_dataset = NoisyPSDDataset(train_x, train_y, data_cfg, augment=True)
    val_dataset = NoisyPSDDataset(val_x, val_y, data_cfg, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    return train_loader, val_loader, mean, std
