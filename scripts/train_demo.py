from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mos2_psd_inversion.config import TrainConfig
from mos2_psd_inversion.data import make_dataloaders
from mos2_psd_inversion.model import PSDDefectRegressor
from mos2_psd_inversion.synthetic import generate_synthetic_psd
from mos2_psd_inversion.train import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MoS2 PSD inversion demo.")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs, targets = generate_synthetic_psd(samples=args.samples, seed=args.seed)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_epochs=min(2, args.epochs),
        patience=max(args.epochs, 5),
    )
    train_loader, val_loader, mean, std = make_dataloaders(inputs, targets, train_config=train_config)
    model = PSDDefectRegressor(mean=mean, std=std)
    device = torch.device(args.device) if args.device else None
    fit(model, train_loader, val_loader, output_dir=args.output_dir, config=train_config, device=device)


if __name__ == "__main__":
    main()
