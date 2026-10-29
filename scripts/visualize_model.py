"""Print a layer-by-layer summary of the hybrid model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchinfo import summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sw_dlnet.models import HybridModel  # noqa: E402


def main() -> None:
    """Build the model on CPU and print its architecture and parameter count."""
    model = HybridModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    sample = {
        "iq": torch.randn(2, 2, 1024),
        "data_len": torch.tensor([1024, 2048]),
    }
    summary(
        model,
        input_data=(sample["iq"], sample["data_len"]),
        depth=3,
        col_names=("input_size", "output_size", "num_params"),
    )


if __name__ == "__main__":
    main()