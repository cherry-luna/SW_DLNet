"""Configuration loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _resolve(path: str) -> str:
    """Resolve a path relative to the repository root."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return str((REPO_ROOT / expanded).resolve())


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve the data/output paths it contains."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key in ("root", "cache", "indices", "stats"):
        if key in config["data"]:
            config["data"][key] = _resolve(config["data"][key])
    for key in ("weights", "figures", "reports"):
        if key in config["output"]:
            config["output"][key] = _resolve(config["output"][key])
    return config
