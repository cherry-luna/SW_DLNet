"""Dataset loading, splitting and batching."""
from __future__ import annotations

import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import PAD_TOKEN, SignalDataset
from .seeding import set_seed

STATS_FILENAME = "dataset_stats_v3.pkl"


class DatasetManager:
    """Build the train / validation / test loaders of I/Q recordings.

    The raw CSV tree is converted to an HDF5 cache on first use; the split
    indices and the per-channel normalisation statistics are cached next to
    it, so later runs start from the same split.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_path: str,
        index_path: str,
        batch_size: int,
        seed: int = 42,
        stats_path: str | None = None,
    ) -> None:
        self.root_dir = os.fspath(root_dir)
        self.dataset_path = os.fspath(dataset_path)
        self.index_path = os.fspath(index_path)
        self.batch_size = batch_size
        self.seed = seed
        self.stats_path = stats_path or os.path.join(
            os.path.dirname(self.dataset_path), STATS_FILENAME
        )

        self.dataset: SignalDataset | None = None
        self.train_dataset: Subset | None = None
        self.valid_dataset: Subset | None = None
        self.test_dataset: Subset | None = None
        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        set_seed(self.seed)
        self._init_dataset()
        self.create_dataloaders()
        self._calculate_stats()

        if self.mean is None or self.std is None:
            raise RuntimeError("Dataset mean/std failed to initialize!")

    # --------------------------------------------------------------- dataset i/o
    def _init_dataset(self) -> None:
        if os.path.exists(self.dataset_path):
            print(f"Loading cached dataset from {self.dataset_path} ...")
            self.dataset = self._load_h5()
        else:
            print(f"Building dataset cache at {self.dataset_path} ...")
            self.dataset = SignalDataset(self.root_dir)
            self._save_h5()
        print("Dataset is ready.")

    def _save_h5(self) -> None:
        os.makedirs(os.path.dirname(self.dataset_path) or ".", exist_ok=True)
        with h5py.File(self.dataset_path, "w") as handle:
            handle.attrs["max_data_len"] = self.dataset.max_data_len
            handle.attrs["max_seq_len"] = self.dataset.max_seq_len

            iq_group = handle.create_group("iq")
            sym_group = handle.create_group("symbol")
            for i in range(len(self.dataset)):
                i_data, q_data = self.dataset.signal_data[i]
                iq_group.create_dataset(f"{i}_i", data=i_data)
                iq_group.create_dataset(f"{i}_q", data=q_data)
                sym_group.create_dataset(str(i), data=self.dataset.symbol_seq[i])

            handle.create_dataset("mod_type", data=self.dataset.modulation_type)
            handle.create_dataset("sym_width", data=self.dataset.symbol_width)

    def _load_h5(self) -> SignalDataset:
        dataset = SignalDataset()
        with h5py.File(self.dataset_path, "r") as handle:
            dataset.max_data_len = handle.attrs["max_data_len"]
            dataset.max_seq_len = handle.attrs["max_seq_len"]

            iq_group = handle["iq"]
            sym_group = handle["symbol"]
            for i in range(len(iq_group) // 2):
                dataset.signal_data.append(
                    (iq_group[f"{i}_i"][:], iq_group[f"{i}_q"][:])
                )
                dataset.symbol_seq.append(sym_group[str(i)][:])

            dataset.modulation_type = handle["mod_type"][:]
            dataset.symbol_width = handle["sym_width"][:]
        return dataset

    # ------------------------------------------------------------------- splitting
    def _split_indices(self):
        if os.path.exists(self.index_path):
            print(f"Loading split indices from {self.index_path} ...")
            with open(self.index_path, "rb") as handle:
                return pickle.load(handle)

        print("Creating new dataset splits ...")
        set_seed(self.seed)
        total_size = len(self.dataset)
        train_size = int(0.6 * total_size)
        valid_test_size = total_size - train_size

        permutation = np.random.permutation(total_size)
        train_indices = permutation[:train_size]
        valid_test_indices = permutation[train_size:]

        test_size = valid_test_size - int(0.5 * valid_test_size)
        test_indices = valid_test_indices[:test_size]
        valid_indices = valid_test_indices[test_size:]

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "wb") as handle:
            pickle.dump((train_indices, valid_indices, test_indices), handle)
        return train_indices, valid_indices, test_indices

    def create_dataloaders(self) -> None:
        train_indices, valid_indices, test_indices = self._split_indices()

        self.train_dataset = Subset(self.dataset, train_indices)
        self.valid_dataset = Subset(self.dataset, valid_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        common = {
            "batch_size": self.batch_size,
            "collate_fn": self._collate,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": False,
        }
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, **common)
        self.valid_loader = DataLoader(self.valid_dataset, shuffle=False, **common)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **common)

    def get_dataloaders(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def get_dataset_sizes(self) -> tuple[int, int, int]:
        return (
            len(self.train_dataset),
            len(self.valid_dataset),
            len(self.test_dataset),
        )

    def check_split_disjoint(self) -> None:
        """Fail loudly if the cached split ever leaks samples across sets."""
        train = set(self.train_dataset.indices)
        valid = set(self.valid_dataset.indices)
        test = set(self.test_dataset.indices)
        overlaps = (train & valid, train & test, valid & test)
        print(
            "Overlapping samples (train-valid, train-test, valid-test): "
            f"{[len(o) for o in overlaps]} (must be 0)"
        )
        assert not any(overlaps), "Data split leaks samples across train/valid/test!"

    # -------------------------------------------------------------------- batching
    def _collate(self, batch: list[dict]) -> dict:
        data_lens = [len(item["iq"][0]) for item in batch]
        seq_lens = [len(item["symbol"]) for item in batch]
        max_data_len = max(data_lens)
        max_seq_len = max(seq_lens)

        batch_iq = torch.zeros((len(batch), 2, max_data_len), dtype=torch.float32)
        batch_symbol = torch.full(
            (len(batch), max_seq_len), PAD_TOKEN, dtype=torch.long
        )
        mod_types, widths = [], []

        for i, item in enumerate(batch):
            valid_len = item["data_len"]
            i_norm = (item["iq"][0][:valid_len] - self.mean[0]) / self.std[0]
            q_norm = (item["iq"][1][:valid_len] - self.mean[1]) / self.std[1]

            batch_iq[i, 0, :valid_len] = torch.from_numpy(np.asarray(i_norm))
            batch_iq[i, 1, :valid_len] = torch.from_numpy(np.asarray(q_norm))

            symbols = item["symbol"]
            batch_symbol[i, : len(symbols)] = symbols

            mod_types.append(item["mod_type"])
            widths.append(item["sym_width"])

        return {
            "iq": batch_iq,
            "symbol": batch_symbol,
            "mod_type": torch.tensor(mod_types),
            "sym_width": torch.tensor(widths, dtype=torch.float32),
            "data_len": torch.tensor(data_lens),
            "seq_len": torch.tensor(seq_lens),
        }

    # ----------------------------------------------------------------- statistics
    def _calculate_stats(self) -> None:
        """Estimate per-channel mean/std over the valid samples of every file."""
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "rb") as handle:
                stats = pickle.load(handle)
            self.mean, self.std = stats["mean"], stats["std"]
            print(f"Loaded existing stats: mean={self.mean}, std={self.std}")
            return

        print("Calculating dataset statistics using actual data lengths ...")
        sum_i = sum_q = 0.0
        sum_sq_i = sum_sq_q = 0.0
        total_points = 0

        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            i_data = item["iq"][0][: item["data_len"]]
            q_data = item["iq"][1][: item["data_len"]]
            sum_i += float(np.sum(i_data))
            sum_q += float(np.sum(q_data))
            sum_sq_i += float(np.sum(np.square(i_data)))
            sum_sq_q += float(np.sum(np.square(q_data)))
            total_points += len(i_data)

        mean_i, mean_q = sum_i / total_points, sum_q / total_points
        self.mean = np.array([mean_i, mean_q], dtype=np.float32)
        self.std = np.array(
            [
                np.sqrt(sum_sq_i / total_points - mean_i**2),
                np.sqrt(sum_sq_q / total_points - mean_q**2),
            ],
            dtype=np.float32,
        ) + 1e-8

        os.makedirs(os.path.dirname(self.stats_path) or ".", exist_ok=True)
        with open(self.stats_path, "wb") as handle:
            pickle.dump({"mean": self.mean, "std": self.std}, handle)
        print(f"Saved new stats: mean={self.mean}, std={self.std}.")