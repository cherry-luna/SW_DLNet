"""Raw CSV parsing and sample construction."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from sw_dlnet.data import PAD_TOKEN, SignalDataset
from sw_dlnet.data.dataset import MODULATION_LABELS

MODULATIONS = [
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


def test_one_sample_per_csv(sample_root):
    dataset = SignalDataset(str(sample_root))
    assert len(dataset) == 3 * len(MODULATIONS)


def test_modulation_labels_are_complete(sample_root):
    dataset = SignalDataset(str(sample_root))
    assert set(dataset.modulation_type) == set(MODULATION_LABELS.values())


def test_modulation_label_mapping():
    assert SignalDataset.modulation_to_label("BPSK") == 1
    assert SignalDataset.modulation_to_label("32APSK") == 10
    with pytest.raises(KeyError):
        SignalDataset.modulation_to_label("OFDM")


def test_symbol_width_matches_csv_header(sample_root):
    dataset = SignalDataset(str(sample_root))
    files = dataset._csv_files()
    index = next(
        i
        for i, (modulation, path) in enumerate(files)
        if modulation == "QPSK" and path.endswith("data_1.csv")
    )
    with open(files[index][1], "r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert dataset.symbol_width[index] == float(rows[0][4])


def test_symbols_are_integer_levels_and_pad_token_is_minus_one(sample_root):
    dataset = SignalDataset(str(sample_root))
    sample = dataset[0]
    assert sample["symbol"].dtype == torch.long
    assert sample["symbol"].numel() > 0
    assert int(sample["symbol"].min()) >= 0
    assert PAD_TOKEN == -1


def test_iq_is_float32_and_unpadded(sample_root):
    dataset = SignalDataset(str(sample_root))
    sample = dataset[0]
    assert sample["iq"][0].dtype == np.float32
    assert sample["iq"][1].shape == sample["iq"][0].shape
    assert sample["data_len"] == sample["iq"][0].size


def test_sample_access_is_cached(sample_root):
    dataset = SignalDataset(str(sample_root))
    assert dataset[2] is dataset[2]


def test_csv_files_are_sorted(sample_root):
    dataset = SignalDataset(str(sample_root))
    paths = [path for _, path in dataset._csv_files()]
    assert paths == sorted(paths)


def test_empty_dataset_is_allowed():
    dataset = SignalDataset()
    assert len(dataset) == 0
