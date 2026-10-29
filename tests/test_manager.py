"""Dataset splitting, caching and batching."""

from __future__ import annotations

import numpy as np
import torch

from sw_dlnet.data import DatasetManager


def test_split_sizes(manager):
    assert manager.get_dataset_sizes() == (18, 6, 6)


def test_split_is_disjoint_and_complete(manager):
    train = set(manager.train_dataset.indices)
    valid = set(manager.valid_dataset.indices)
    test = set(manager.test_dataset.indices)
    assert not (train & valid) and not (train & test) and not (valid & test)
    assert train | valid | test == set(range(len(manager.dataset)))
    manager.check_split_disjoint()


def test_split_is_reproducible(sample_root, tmp_path):
    def build(folder: str) -> tuple:
        dataset_manager = DatasetManager(
            root_dir=str(sample_root),
            dataset_path=str(tmp_path / folder / "dataset.h5"),
            index_path=str(tmp_path / folder / "indices.pkl"),
            batch_size=8,
            seed=42,
        )
        return tuple(
            np.asarray(part.indices).tolist()
            for part in (
                dataset_manager.train_dataset,
                dataset_manager.valid_dataset,
                dataset_manager.test_dataset,
            )
        )

    assert build("a") == build("b")


def test_hdf5_cache_roundtrip(sample_root, tmp_path):
    cache = tmp_path / "dataset.h5"
    first = DatasetManager(
        root_dir=str(sample_root),
        dataset_path=str(cache),
        index_path=str(tmp_path / "indices.pkl"),
        batch_size=8,
    )
    second = DatasetManager(
        root_dir=str(sample_root),
        dataset_path=str(cache),
        index_path=str(tmp_path / "indices.pkl"),
        batch_size=8,
    )
    np.testing.assert_array_equal(first.dataset.signal_data[0][0], second.dataset.signal_data[0][0])
    np.testing.assert_array_equal(first.dataset.symbol_seq[3], second.dataset.symbol_seq[3])
    assert first.dataset.max_data_len == second.dataset.max_data_len
    assert list(first.dataset.modulation_type) == list(second.dataset.modulation_type)


def test_statistics_are_finite(manager):
    assert np.all(np.isfinite(manager.mean))
    assert np.all(np.isfinite(manager.std))
    assert np.all(manager.std > 0)


def test_collate_shapes_and_dtypes(manager):
    batch = next(iter(manager.train_loader))
    assert batch["iq"].ndim == 3 and batch["iq"].shape[1] == 2
    assert batch["iq"].dtype == torch.float32
    assert batch["symbol"].dtype == torch.long
    assert batch["mod_type"].dtype == torch.int64
    assert batch["sym_width"].dtype == torch.float32
    assert batch["data_len"].shape == batch["sym_width"].shape
    assert batch["symbol"].shape[0] == batch["iq"].shape[0]


def test_collate_pads_with_pad_token(manager):
    batch = next(iter(manager.train_loader))
    for row, seq_len in zip(batch["symbol"], batch["seq_len"]):
        assert torch.all(row[seq_len:] == -1)


def test_collate_normalises_with_dataset_statistics(manager):
    sample = manager.dataset[0]
    batch = manager._collate([sample])
    raw_i = np.asarray(sample["iq"][0], dtype=np.float32)
    expected = (raw_i - manager.mean[0]) / manager.std[0]
    np.testing.assert_allclose(batch["iq"][0, 0].numpy(), expected, rtol=1e-5, atol=1e-6)
