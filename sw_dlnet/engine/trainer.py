"""Training and evaluation loops."""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from ..utils.factories import create_criterion, create_optimizer, create_scheduler
from ..utils.metrics import evaluate_sw_score, sw_scores
from ..utils.plotting import plot_loss_curve, plot_report


class Trainer:
    """Run the multi-stage schedule and keep the best SW checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        dataset_manager,
        stages: list[dict[str, Any]],
        batch_size: int,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.dataset_manager = dataset_manager
        self.batch_size = batch_size
        self.use_amp = use_amp and device.type == "cuda"

        self.train_loader, self.val_loader, self.test_loader = (
            dataset_manager.get_dataloaders()
        )
        self.stage_configs = stages
        self.history: dict[str, Any] = {"stages": []}
        self.best_sw_score = -math.inf

        self.scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=2**16,
            growth_factor=1.5,
            backoff_factor=0.5,
            enabled=self.use_amp,
        )
        self._configure_stages()

    def _configure_stages(self) -> None:
        """Instantiate optimizer, scheduler and criterion for every stage."""
        for config in self.stage_configs:
            optimizer_cfg = dict(config["optimizer"])
            optimizer_type = optimizer_cfg.pop("type")
            lr = optimizer_cfg.pop("lr")
            optimizer = create_optimizer(
                optimizer_type, self.model.parameters(), lr, **optimizer_cfg
            )

            scheduler_cfg = dict(config["scheduler"])
            scheduler_type = scheduler_cfg.pop("type")
            if scheduler_type.lower() == "onecycle":
                epochs = scheduler_cfg.pop("total_steps_epochs")
                scheduler_cfg["total_steps"] = epochs * len(self.train_loader)
            scheduler = create_scheduler(scheduler_type, optimizer, **scheduler_cfg)

            self.history["stages"].append(
                {
                    "name": config["name"],
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "criterion": create_criterion(config["criterion"]["type"]),
                    "epochs": config["epochs"],
                    "early_stop": config["early_stop"],
                    "current_epoch": 0,
                    "best_sw_score": -math.inf,
                }
            )

    def _autocast(self):
        return torch.amp.autocast("cuda", enabled=self.scaler.is_enabled())

    def _move_batch(self, batch: dict):
        iq = batch["iq"].to(self.device, non_blocking=True)
        widths = batch["sym_width"].to(self.device, non_blocking=True)
        data_len = batch["data_len"].to(self.device, non_blocking=True)
        return iq, widths, data_len

    # ------------------------------------------------------------------ one step
    def train_step(self, stage_idx: int, batch: dict) -> float:
        """Forward, backward and optimiser step for a single batch."""
        stage = self.history["stages"][stage_idx]
        self.model.train()
        stage["optimizer"].zero_grad(set_to_none=True)

        iq, widths, data_len = self._move_batch(batch)

        with self._autocast():
            predictions = self.model(iq, data_len)
            if torch.isnan(predictions).any():
                print("NaN detected in model output.")
                raise RuntimeError("NaN prediction encountered, training aborted.")
            loss = stage["criterion"](predictions.squeeze(), widths)

        if torch.isnan(loss).any():
            print("NaN detected in loss.")
            raise RuntimeError("NaN loss encountered, training aborted.")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(stage["optimizer"])
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(stage["optimizer"])
        self.scaler.update()

        scheduler = stage["scheduler"]
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        return loss.item()

    # ----------------------------------------------------------------- validation
    @torch.inference_mode()
    def validate(self, stage_idx: int) -> dict[str, float]:
        """Average validation loss and SW score over the validation split."""
        criterion = self.history["stages"][stage_idx]["criterion"]
        total_loss = 0.0
        predictions: list[float] = []
        targets: list[float] = []

        progress = tqdm(self.val_loader, desc="Validation", position=1, leave=False)
        for batch in progress:
            iq, widths, data_len = self._move_batch(batch)
            with self._autocast():
                outputs = self.model(iq, data_len)
                loss = criterion(outputs.squeeze(), widths)
            total_loss += loss.item()
            predictions.extend(outputs.flatten().cpu().numpy().tolist())
            targets.extend(widths.flatten().cpu().numpy().tolist())

        return {
            "val_loss": total_loss / len(self.val_loader),
            "sw_score": evaluate_sw_score(targets, predictions),
        }

    # --------------------------------------------------------------------- epochs
    def _save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_arch": "HybridModel",
                "scaler_state": self.scaler.state_dict(),
                "dataset_stats": {
                    "mean": self.dataset_manager.mean,
                    "std": self.dataset_manager.std,
                },
            },
            path,
        )

    def _step_scheduler(self, stage: dict) -> None:
        scheduler = stage["scheduler"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(stage["val_loss"])
        elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

    def fit(
        self,
        checkpoint_path: str,
        curve_path: str,
        curve_prefix: str = "",
    ) -> float:
        """Train every stage in order and return the best validation SW score."""
        train_losses: list[float] = []
        val_losses: list[float] = []
        total_epochs = sum(stage["epochs"] for stage in self.history["stages"])

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)

        outer = tqdm(total=total_epochs, desc="Overall Training", position=0)
        for stage_idx, stage in enumerate(self.history["stages"]):
            epochs = stage["epochs"]
            early_stop = stage["early_stop"]
            no_improve = 0

            for epoch in range(epochs):
                stage["current_epoch"] = epoch + 1
                epoch_loss = 0.0

                inner = tqdm(
                    self.train_loader, desc=stage["name"], position=1, leave=False
                )
                for batch in inner:
                    loss = self.train_step(stage_idx, batch)
                    epoch_loss += loss
                    inner.set_postfix(
                        loss=f"{loss:.4f}",
                        lr=f"{stage['optimizer'].param_groups[0]['lr']:.2e}",
                    )
                inner.close()

                train_loss = epoch_loss / len(self.train_loader)
                train_losses.append(train_loss)
                stage.setdefault("train_losses", []).append(train_loss)

                metrics = self.validate(stage_idx)
                val_losses.append(metrics["val_loss"])
                stage.setdefault("val_losses", []).append(metrics["val_loss"])
                stage["val_loss"] = metrics["val_loss"]

                outer.update(1)
                outer.set_postfix(
                    val_loss=f"{metrics['val_loss']:.4f}",
                    sw=f"{metrics['sw_score']:.4f}",
                )

                self._step_scheduler(stage)

                if metrics["sw_score"] > self.best_sw_score:
                    self.best_sw_score = metrics["sw_score"]
                    stage["best_sw_score"] = metrics["sw_score"]
                    no_improve = 0
                    self._save_checkpoint(checkpoint_path)
                    print(
                        f"Stage '{stage['name']}' epoch {epoch + 1}: new best SW "
                        f"score {self.best_sw_score:.4f}, checkpoint saved."
                    )
                else:
                    no_improve += 1

                if no_improve >= early_stop:
                    print(
                        f"Stage '{stage['name']}' early stopping at epoch "
                        f"{epoch + 1} (best SW {self.best_sw_score:.4f})"
                    )
                    outer.update(epochs - (epoch + 1))
                    break

            plot_loss_curve(
                train_losses, val_losses, curve_path, prefix=curve_prefix
            )

        outer.close()
        print(f"Training finished. Best SW score: {self.best_sw_score:.4f}")
        return self.best_sw_score


class Tester:
    """Evaluate a trained checkpoint on the test split."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        test_loader,
        save_dir: str,
        prefix: str = "",
    ) -> dict[str, float]:
        """Run the test split and write the metrics and figures."""
        self.model.eval()
        targets: list[float] = []
        predictions: list[float] = []

        for batch in tqdm(test_loader, desc="Testing", leave=False):
            iq = batch["iq"].to(self.device)
            data_len = batch["data_len"].to(self.device)
            outputs = self.model(iq, data_len)
            targets.extend(batch["sym_width"].numpy().tolist())
            predictions.extend(outputs.cpu().numpy().flatten().tolist())

        targets_array = np.asarray(targets)
        predictions_array = np.asarray(predictions)

        metrics = {
            "mae": float(mean_absolute_error(targets_array, predictions_array)),
            "mse": float(mean_squared_error(targets_array, predictions_array)),
            "r2": float(r2_score(targets_array, predictions_array)),
            "sw_score": float(evaluate_sw_score(targets_array, predictions_array)),
        }

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        plot_report(
            targets_array,
            predictions_array,
            sw_scores(targets_array, predictions_array),
            save_dir,
            prefix=prefix,
        )

        _print_report(metrics)
        return metrics


def _print_report(metrics: dict[str, float]) -> None:
    """Print the test report in the layout used by the original release."""
    print("Test report:")
    print(f"  MAE      : {metrics['mae']:.4f}")
    print(f"  MSE      : {metrics['mse']:.4f}")
    print(f"  R2       : {metrics['r2']:.4f}")
    print(f"  SW score : {metrics['sw_score']:.4f}")