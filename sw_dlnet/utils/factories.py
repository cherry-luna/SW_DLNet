"""Optimizer, scheduler and criterion factories."""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import mape_loss


def create_optimizer(
    optim_type: str,
    model_params,
    lr: float,
    **kwargs,
) -> torch.optim.Optimizer:
    """Build an optimizer from a config node."""
    name = optim_type.lower()
    momentum = kwargs.get("momentum", 0.9)
    weight_decay = kwargs.get("weight_decay", 1e-4)

    if name == "sgd":
        return torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            nesterov=kwargs.get("nesterov", True),
            weight_decay=weight_decay,
        )

    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "radam": torch.optim.RAdam}
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optim_type}")
    return optimizers[name](
        model_params,
        lr=lr,
        betas=kwargs.get("betas", (0.9, 0.999)),
        weight_decay=weight_decay,
    )


def create_criterion(criterion_type: str):
    """Build a loss function from a config node."""
    name = criterion_type.lower()
    if name == "mape":
        return mape_loss
    if name == "mse":
        return nn.MSELoss(reduction="mean")
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unsupported criterion: {criterion_type}")


def create_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    **kwargs,
):
    """Build a learning-rate scheduler from a config node."""
    name = scheduler_type.lower()

    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs["max_lr"],
            total_steps=kwargs["total_steps"],
            pct_start=kwargs.get("pct_start", 0.3),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs["T_max"],
            eta_min=kwargs.get("eta_min", 1e-6),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    if name == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=5e-5,
            max_lr=1e-3,
            step_size_up=2000,
            mode="triangular2",
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_type}")
