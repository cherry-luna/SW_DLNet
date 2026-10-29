"""Shared fixtures for the test-suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sw_dlnet.data import DatasetManager

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
SAMPLE_MODULATIONS = [
    "BPSK",
    "QPSK",
    "8PSK",
    "MSK",
    "8QAM",
    "16QAM",
    "32QAM",
    "8APSK",
    "16APSK",
    "32APSK",
]


@pytest.fixture(scope="session")
def sample_root() -> Path:
    """Directory holding three real CSV files per modulation class."""
    if not SAMPLE_DIR.is_dir():
        pytest.skip("bundled sample dataset is missing")
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def manager(sample_root: Path, tmp_path_factory) -> DatasetManager:
    """A DatasetManager built on the sample dataset with a temporary cache."""
    cache_dir = tmp_path_factory.mktemp("cache")
    return DatasetManager(
        root_dir=str(sample_root),
        dataset_path=str(cache_dir / "dataset.h5"),
        index_path=str(cache_dir / "indices.pkl"),
        batch_size=8,
        seed=42,
    )
