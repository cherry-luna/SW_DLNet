"""Raw CSV dataset for I/Q recordings."""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import correct_iq_phase

PAD_TOKEN = -1

MODULATION_LABELS = {
    "BPSK": 1,
    "QPSK": 2,
    "8PSK": 3,
    "MSK": 4,
    "8QAM": 5,
    "16QAM": 6,
    "32QAM": 7,
    "8APSK": 8,
    "16APSK": 9,
    "32APSK": 10,
}


class SignalDataset(Dataset):
    """One sample per CSV file: I/Q samples, symbols and the symbol width."""

    def __init__(
        self,
        root_dir: str = "",
        phase_corrector=correct_iq_phase,
        max_data_len: int | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.phase_corrector = phase_corrector
        self.max_data_len = max_data_len
        self.max_seq_len = max_seq_len

        self.signal_data: list[tuple[list[float], list[float]]] = []
        self.symbol_seq: list[list[int]] = []
        self.modulation_type: list[int] = []
        self.symbol_width: list[float] = []
        self.cache: dict[int, dict] = {}

        if root_dir:
            self._load_from_raw()

    def _csv_files(self) -> list[tuple[str, str]]:
        """Return (modulation, path) pairs, sorted for a stable sample order."""
        pairs: list[tuple[str, str]] = []
        for mod_dir in sorted(os.listdir(self.root_dir)):
            mod_path = os.path.join(self.root_dir, mod_dir)
            if not os.path.isdir(mod_path):
                continue
            for name in sorted(os.listdir(mod_path)):
                if name.endswith(".csv"):
                    pairs.append((mod_dir, os.path.join(mod_path, name)))
        return pairs

    @staticmethod
    def _read_rows(path: str) -> list[list[str]]:
        """Read a CSV file, dropping empty lines."""
        with open(path, "r", encoding="utf-8") as handle:
            return [row for row in csv.reader(handle) if any(row)]

    def _load_from_raw(self) -> None:
        """Scan the CSV tree twice: first for lengths, then for content."""
        files = self._csv_files()

        data_lens, seq_lens = [], []
        for _, path in files:
            rows = self._read_rows(path)
            data_lens.append(len(rows))
            seq_lens.append(sum(1 for row in rows if row[2].strip()))

        if not self.max_data_len:
            self.max_data_len = int(np.percentile(data_lens, 95) * 1.1)
        if not self.max_seq_len:
            self.max_seq_len = int(np.percentile(seq_lens, 95) * 1.1)

        for mod_dir, path in files:
            rows = self._read_rows(path)
            i_data = [float(row[0]) for row in rows]
            q_data = [float(row[1]) for row in rows]
            symbols = [int(row[2]) for row in rows if row[2].strip()]

            self.signal_data.append((i_data, q_data))
            self.symbol_seq.append(symbols)
            self.modulation_type.append(self.modulation_to_label(mod_dir))
            self.symbol_width.append(float(rows[0][4]))

    @staticmethod
    def modulation_to_label(mod_dir: str) -> int:
        """Map a modulation folder name to its integer label."""
        return MODULATION_LABELS[mod_dir]

    def __len__(self) -> int:
        return len(self.signal_data)

    def __getitem__(self, idx: int) -> dict:
        """Return one phase-corrected sample; results are memoised."""
        cached = self.cache.get(idx)
        if cached is not None:
            return cached

        iq_raw = self.signal_data[idx]
        i_corrected, q_corrected = self.phase_corrector(iq_raw)

        sample = {
            "iq": (
                np.array(i_corrected, dtype=np.float32),
                np.array(q_corrected, dtype=np.float32),
            ),
            "symbol": torch.tensor(self.symbol_seq[idx], dtype=torch.long),
            "mod_type": self.modulation_type[idx],
            "sym_width": self.symbol_width[idx],
            "data_len": len(iq_raw[0]),
        }
        self.cache[idx] = sample
        return sample
