from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Frequency layout and augmentation settings for PSD samples."""

    sequence_length: int = 3000
    train_fraction: float = 0.875
    noise_variance_base: float = 0.2
    noise_variance_slope: float = 0.1
    noise_variance_scale: float = 1000.0
    noise_variance_max: float = 0.3
    noise_clip_range: float = 0.5
    noise_smoothing_factor: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    """Model constants inherited from the research prototype."""

    low_slice: tuple[int, int] = (0, 80)
    mid_slice: tuple[int, int] = (30, 300)
    high_slice: tuple[int, int] = (300, 3000)
    output_dim: int = 2
    scale_factor: float = 2.5e24
    log_epsilon: float = 1e-30
    norm_epsilon: float = 1e-8


@dataclass(frozen=True)
class TrainConfig:
    """Training defaults for small and medium runs."""

    batch_size: int = 64
    epochs: int = 100
    num_workers: int = 0
    base_lr: float = 1e-5
    target_lr: float = 1e-3
    warmup_epochs: int = 20
    patience: int = 50
    grad_clip_norm: float = 1.0
    ea_alpha_weight: float = 10.0
    ea_beta_weight: float = 1.0
