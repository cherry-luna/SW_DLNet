"""Global random seed helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int | None = None) -> None:
    """Seed Python, NumPy and PyTorch and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False