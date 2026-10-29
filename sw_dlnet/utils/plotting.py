"""Matplotlib helpers for training curves and test reports."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _figure_path(save_dir: str, stem: str, prefix: str) -> str:
    """Join a directory and a file name, honouring an optional prefix."""
    name = f"{prefix}_{stem}.png" if prefix else f"{stem}.png"
    return os.path.join(save_dir, name)


def _title(text: str, prefix: str) -> str:
    """Prefix a figure title when one is supplied."""
    return f"{prefix} - {text}" if prefix else text


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
    prefix: str = "",
) -> str:
    """Plot the training and validation loss of every epoch seen so far."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, color="blue", label="Training Loss")
    ax.plot(val_losses, color="red", label="Validation Loss")
    ax.set_title(_title("Training and Validation Loss Curve", prefix))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    save_dir: str,
    prefix: str = "",
) -> list[str]:
    """Save the prediction / residual / error-distribution figures."""
    os.makedirs(save_dir, exist_ok=True)
    relative_error = (targets - predictions) / np.abs(targets)
    saved = []

    # 1. predictions coloured by SW score
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        targets,
        predictions,
        c=scores,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="w",
        s=100,
    )
    low, high = float(targets.min()), float(targets.max())
    ax.plot([low, high], [low, high], "r--", label="Ideal Prediction")
    ax.set_title(
        _title("Predicted vs Actual Values (coloured by SW score)", prefix)
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="SW score")
    path = _figure_path(save_dir, "predicted_vs_actual", prefix)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 2. relative error distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(relative_error, bins=50, color="skyblue", edgecolor="black")
    ax.set_title(_title("Relative Error Distribution", prefix))
    ax.set_xlabel("(Actual - Predicted) / |Actual|")
    ax.set_ylabel("Frequency")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 12000)
    path = _figure_path(save_dir, "relative_error_distribution", prefix)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 3. residuals coloured by SW score
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        targets,
        relative_error,
        c=scores,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="w",
        s=100,
    )
    ax.axhline(0, color="black", linestyle="--", label="Zero Error Line")
    ax.set_title(_title("Error vs Actual Values (coloured by SW score)", prefix))
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Residuals (Error)")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="SW score")
    path = _figure_path(save_dir, "error_vs_actual", prefix)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    return saved