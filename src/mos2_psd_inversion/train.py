from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .config import TrainConfig


class WeightedMSELoss(nn.Module):
    def __init__(self, ea_alpha_weight: float = 10.0, ea_beta_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor([ea_alpha_weight, ea_beta_weight]).float())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((predictions - targets).pow(2) * self.weights.to(predictions.device))


class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        target_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        current_epoch = self.last_epoch + 1
        if current_epoch <= self.warmup_epochs:
            factor = current_epoch / max(self.warmup_epochs, 1)
            lr = self.base_lr + (self.target_lr - self.base_lr) * factor
        else:
            denom = max(self.total_epochs - self.warmup_epochs, 1)
            progress = (current_epoch - self.warmup_epochs) / denom
            lr = self.target_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return [lr for _ in self.base_lrs]


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return
        self.counter += 1
        self.should_stop = self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss = criterion(model(inputs), targets)
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, list[float] | float]:
    cfg = config or TrainConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = WeightedMSELoss(cfg.ea_alpha_weight, cfg.ea_beta_weight)
    optimizer = Adam(model.parameters(), lr=cfg.base_lr)
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
        base_lr=cfg.base_lr,
        target_lr=cfg.target_lr,
    )
    early_stopper = EarlyStopping(cfg.patience)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg.grad_clip_norm,
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"epoch={epoch + 1:03d}/{cfg.epochs:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.6g}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
            },
            output_dir / "checkpoint.pth",
        )

        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print("early stopping triggered")
            break

    _save_loss_curve(history, output_dir / "loss_curve.png")
    return {**history, "best_val_loss": best_val_loss}


def _save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("weighted MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
