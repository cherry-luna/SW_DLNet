"""End-to-end training and evaluation on the bundled sample."""

from __future__ import annotations

import json
import math

import pytest
import torch

from sw_dlnet.engine import Tester as ModelTester
from sw_dlnet.engine import Trainer
from sw_dlnet.models import HybridModel

CHECKPOINT_KEYS = {"model_state", "model_arch", "scaler_state", "dataset_stats"}


def _sample_stages(epochs: int = 2) -> list[dict]:
    return [
        {
            "name": "Sample",
            "epochs": epochs,
            "early_stop": epochs,
            "optimizer": {"type": "RAdam", "lr": 1e-4, "weight_decay": 1e-4},
            "scheduler": {"type": "onecycle", "max_lr": 1e-3, "total_steps_epochs": epochs},
            "criterion": {"type": "mape"},
        }
    ]


@pytest.mark.slow
def test_train_evaluate_and_reload(manager, tmp_path, device):
    torch.manual_seed(0)
    model = HybridModel(use_amp=False)
    trainer = Trainer(model, device, manager, _sample_stages(), batch_size=8, use_amp=False)

    checkpoint = tmp_path / "best.pth"
    curve = tmp_path / "loss_curve.png"
    best_score = trainer.fit(str(checkpoint), str(curve))

    assert checkpoint.is_file() and curve.is_file()
    assert math.isfinite(best_score) and 0.0 <= best_score <= 100.0
    assert len(trainer.history["stages"][0]["train_losses"]) == 2

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert set(payload) == CHECKPOINT_KEYS
    assert payload["model_arch"] == "HybridModel"
    np_stats = payload["dataset_stats"]
    assert set(np_stats) == {"mean", "std"}

    metrics = ModelTester(model, device).evaluate(trainer.test_loader, str(tmp_path / "reports"))
    assert set(metrics) == {"mae", "mse", "r2", "sw_score"}
    assert all(math.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["sw_score"] <= 100.0

    stored = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert stored == pytest.approx(metrics)
    for name in (
        "predicted_vs_actual.png",
        "error_vs_actual.png",
        "relative_error_distribution.png",
    ):
        assert (tmp_path / "reports" / name).is_file()
