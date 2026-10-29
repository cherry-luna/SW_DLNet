"""The command-line entry point runs end to end."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_train_script_writes_reports(tmp_path):
    config_text = (REPO_ROOT / "configs" / "sample.yaml").read_text(encoding="utf-8")
    output_root = str(tmp_path / "sample").replace(os.sep, "/")
    config_text = config_text.replace("outputs/sample", output_root)

    config_path = tmp_path / "sample.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    environment = dict(
        os.environ,
        KMP_DUPLICATE_LIB_OK="TRUE",
        PYTHONPATH=str(REPO_ROOT),
    )
    process = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert process.returncode == 0, process.stderr[-3000:]
    reports = tmp_path / "sample" / "reports"
    assert (reports / "metrics.json").is_file()
    assert (tmp_path / "sample" / "weights" / "best_model_8.pth").is_file()
    assert (tmp_path / "sample" / "figures" / "loss_curve_8.png").is_file()
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"mae", "mse", "r2", "sw_score"}
