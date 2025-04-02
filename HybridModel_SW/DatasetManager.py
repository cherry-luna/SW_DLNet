import os
import pickle
import h5py
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from SignalDataset import SignalDataset
from random_seed import set_seed
from IQPhase import correct_iq_phase

PAD_TOKEN = -1


class DatasetManager:
    _current_instance = None

    def __init__(self, root_dir, dataset_dir, index_dir, batch_size, seed=42):
        """
        初始化 DatasetManager，管理数据集的加载、划分和数据加载。

        :param root_dir: 原始数据集路径
        :param dataset_dir: 保存数据集的路径
        :param index_dir: 保存数据划分索引的路径
        :param batch_size: 批量大小
        :param seed: 随机种子
        """
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        self.index_dir = index_dir
        self.batch_size = batch_size
        self.seed = seed

        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        # 新增标准化参数存储
        self.stats_path = os.path.join(os.path.dirname(dataset_dir), './dataset_stats_v3.pkl')
        self.mean = None
        self.std = None
        self.mean_tensor = None
        self.std_tensor = None

        # 设置随机种子
        set_seed(self.seed)
        # 初始化数据集
        self._init_dataset()
        # 创建数据加载器
        self.create_dataloaders()
        # **计算统计参数**
        self._calculate_stats()
        # **检查是否正确计算**
        if self.mean is None or self.std is None:
            raise RuntimeError("Dataset mean/std failed to initialize!")

    def _calculate_stats(self):
        """计算训练集的均值和标准差（仅使用有效数据部分）"""
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'rb') as f:
                stats = pickle.load(f)
                self.mean, self.std = stats['mean'], stats['std']
                self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32)
                self.std_tensor = torch.tensor(self.std, dtype=torch.float32)
            print(f"Loaded existing stats: mean={self.mean}, std={self.std}")
            return

        print("Calculating dataset statistics using ACTUAL data lengths...")
        sum_i = 0.0
        sum_q = 0.0
        sum_sq_i = 0.0
        sum_sq_q = 0.0
        total_points = 0


        for idx in tqdm(range(len(self.dataset)), desc="Processing training data..."):
            item = self.dataset[idx]
            i_data = item['iq'][0][:item['data_len']]  # 使用实际数据长度
            q_data = item['iq'][1][:item['data_len']]
            sum_i += np.sum(i_data)
            sum_q += np.sum(q_data)
            sum_sq_i += np.sum(np.square(i_data))
            sum_sq_q += np.sum(np.square(q_data))
            total_points += len(i_data)

        self.mean = np.array([sum_i/total_points, sum_q/total_points], dtype=np.float32)
        self.std = np.array([
            np.sqrt(sum_sq_i/total_points - (sum_i/total_points)**2),
            np.sqrt(sum_sq_q/total_points - (sum_q/total_points)**2)
        ], dtype=np.float32) + 1e-8

        # 保存统计量
        with open(self.stats_path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)
        print(f"Saved new stats: mean={self.mean}, std={self.std}.")

    def _normalize(self, data):
        """应用标准化"""
        if self.mean is None or self.std is None:
            raise ValueError("Standardized parameters must be calculated first!")
        return (data - self.mean) / (self.std + 1e-8)

    def _init_dataset(self):
        if os.path.exists(self.dataset_dir):
            print(f"Loading saved signal dataset from {self.dataset_dir}...")
            self.dataset = self._load_h5()
        else:
            print(f"HDF5 dataset not found. Creating and saving signal dataset to {self.dataset_dir}...")
            self.dataset = SignalDataset(self.root_dir, correct_iq_phase_func=correct_iq_phase)
            self._save_h5()
        print("Loading or creating dataset is successful.")

    def _save_h5(self):
        with h5py.File(self.dataset_dir, 'w') as f:
            # 保存元数据
            f.attrs['max_data_len'] = self.dataset.max_data_len
            f.attrs['max_seq_len'] = self.dataset.max_seq_len

            # 变长存储
            iq_group = f.create_group('iq')
            sym_group = f.create_group('symbol')
            for i in range(len(self.dataset)):
                iq = self.dataset.signal_data[i]
                iq_group.create_dataset(f'{i}_i', data=iq[0])
                iq_group.create_dataset(f'{i}_q', data=iq[1])
                sym_group.create_dataset(str(i), data=self.dataset.symbol_seq[i])

            f.create_dataset('mod_type', data=self.dataset.modulation_type)
            f.create_dataset('sym_width', data=self.dataset.symbol_width)

    def _load_h5(self):
        with h5py.File(self.dataset_dir, 'r') as f:
            dataset = SignalDataset('')
            dataset.max_data_len = f.attrs['max_data_len']
            dataset.max_seq_len = f.attrs['max_seq_len']

            # 加载变长数据
            iq_group = f['iq']
            sym_group = f['symbol']
            for i in range(len(iq_group) // 2):
                dataset.signal_data.append((
                    iq_group[f'{i}_i'][:],
                    iq_group[f'{i}_q'][:]
                ))
                dataset.symbol_seq.append(sym_group[str(i)][:])

            dataset.modulation_type = f['mod_type'][:]
            dataset.symbol_width = f['sym_width'][:]
        return dataset

    def create_dataloaders(self):
        """
        创建训练集、验证集和测试集的 DataLoader。
        """
        # 根据CPU核心数设置workers
        num_workers = min(4, os.cpu_count())
        pin_memory = torch.cuda.is_available()

        # **确保索引文件是新的**
        if os.path.exists(self.index_dir):
            print(f"Loading saved indices from {self.index_dir}...")
            with open(self.index_dir, "rb") as f:
                train_indices, valid_indices, test_indices = pickle.load(f)
            # os.remove(self.index_dir)  # **删除旧的索引文件，确保不会加载错误的索引**
        else:
            print("Creating new dataset splits...")

            # 固定随机种子，确保数据划分一致
            set_seed(self.seed)
            # 获取数据集的长度
            total_size = len(self.dataset)
            train_size = int(0.6 * total_size)
            valid_test_size = total_size - train_size
            # 获取随机打乱的索引
            all_indices = np.random.permutation(total_size)
            train_indices = all_indices[:train_size]
            valid_test_indices = all_indices[train_size:]

            valid_size = int(0.5 * valid_test_size)  # 50% 用作验证集
            test_size = valid_test_size - valid_size  # 剩余 50% 用作测试集
            test_indices = valid_test_indices[:test_size]
            valid_indices = valid_test_indices[test_size:]

            # 保存索引
            with open(self.index_dir, "wb") as f:
                pickle.dump((train_indices, valid_indices, test_indices), f)

        self.train_dataset = Subset(self.dataset, train_indices)
        self.valid_dataset = Subset(self.dataset, valid_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
        print("Data split successful. No leakage detected.")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       collate_fn=self.dynamic_collate, prefetch_factor=2,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=pin_memory, persistent_workers=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                       collate_fn=self.dynamic_collate,
                                       shuffle=False, num_workers=num_workers,
                                       pin_memory=pin_memory, persistent_workers=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      collate_fn=self.dynamic_collate,
                                      shuffle=False, num_workers=num_workers,
                                      pin_memory=pin_memory, persistent_workers=True)

    def get_dataloaders(self):
        """
        获取训练集、验证集和测试集的 DataLoader。

        :return: 训练集、验证集和测试集的 DataLoader
        """
        return self.train_loader, self.valid_loader, self.test_loader

    def get_dataset_sizes(self):
        """
        获取训练集、验证集和测试集的大小。

        :return: 训练集、验证集和测试集的大小
        """
        return len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset)

    @staticmethod
    def dynamic_collate(batch):
        """动态填充collate函数"""
        # 获取全局标准化参数
        manager = DatasetManager.__get_current_manager()
        if manager is None:
            raise RuntimeError("DatasetManager instance is None! Make sure it is used in a `with` statement.")

        if manager.mean is None or manager.std is None:
            raise RuntimeError("DatasetManager mean or std is not initialized!")

        # 计算批次最大长度
        data_lens = [len(item['iq'][0]) for item in batch]
        seq_lens = [len(item['symbol']) for item in batch]
        max_data_len = max(data_lens)
        max_seq_len = max(seq_lens)

        # 初始化填充容器
        batch_iq = torch.zeros((len(batch), 2, max_data_len), dtype=torch.float32)
        batch_symbol = torch.full((len(batch), max_seq_len), PAD_TOKEN, dtype=torch.long)
        batch_mod = []
        batch_width = []

        for i, item in enumerate(batch):
            # 获取原始有效数据
            valid_len = item['data_len']
            i_data = item['iq'][0][:valid_len]  # 直接使用tensor
            q_data = item['iq'][1][:valid_len]  # 直接使用tensor

            # 应用标准化
            i_norm = (i_data - manager.mean_tensor[0]) / manager.std_tensor[0]
            q_norm = (q_data - manager.mean_tensor[1]) / manager.std_tensor[1]

            # 填充标准化后的数据
            batch_iq[i, 0, :valid_len] = i_norm
            batch_iq[i, 1, :valid_len] = q_norm

            # 码序列填充
            symbols = item['symbol']
            batch_symbol[i, :len(symbols)] = symbols

            batch_mod.append(item['mod_type'])
            batch_width.append(item['sym_width'])

        return {
            'iq': batch_iq, 
            'symbol': batch_symbol,
            'mod_type': torch.tensor(batch_mod),
            'sym_width': torch.tensor(batch_width, dtype=torch.float32),
            'data_len': torch.tensor(data_lens),
            'seq_len': torch.tensor(seq_lens)
        }

    @classmethod
    def __get_current_manager(cls):
        """获取当前活动的DatasetManager实例"""
        return cls._current_instance

    def __enter__(self):
        DatasetManager._current_instance = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        DatasetManager._current_instance = None
