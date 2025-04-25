import os
import csv
import numpy as np
from torch.utils.data import Dataset
import torch

PAD_TOKEN = -1


class SelfSignalDataset(Dataset):

    def __init__(self, root_dir, transform=None, correct_iq_phase_func=None,
                 max_data_len=None, max_seq_len=None):

        self.root_dir = root_dir
        self.transform = transform
        # if correct_iq_phase_func is None:
        #     self.correct_iq_phase_func = self.default_phase_correction
        # else:
        #     self.correct_iq_phase_func = correct_iq_phase_func
        self.correct_iq_phase_func = correct_iq_phase_func
        self.max_data_len = max_data_len
        self.max_seq_len = max_seq_len

        self.signal_data = []
        self.symbol_seq = []
        self.modulation_type = []
        self.symbol_width = []

        self.cache = {}  # 添加缓存

        if root_dir:
            self._load_from_raw()

    def _load_from_raw(self):
        """从原始CSV加载数据并计算动态长度"""
        data_lens, seq_lens = [], []
        mod_folders = [d for d in os.listdir(self.root_dir)
                       if os.path.isdir(os.path.join(self.root_dir, d))]

        # 第一遍扫描获取长度分布
        for mod_dir in mod_folders:
            mod_path = os.path.join(self.root_dir, mod_dir)
            for fname in os.listdir(mod_path):
                if not fname.endswith('.csv'):
                    continue
                with open(os.path.join(mod_path, fname), 'r') as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader if any(row)]
                data_lens.append(len(rows))
                seq_lens.append(sum(1 for row in rows if row[2].strip()))

        # 动态计算最大长度
        self.max_data_len = int(np.percentile(data_lens, 95) * 1.1) if not self.max_data_len else self.max_data_len
        self.max_seq_len = int(np.percentile(seq_lens, 95) * 1.1) if not self.max_seq_len else self.max_seq_len

        # 第二遍加载数据
        for mod_dir in mod_folders:
            mod_type = self.modulation_to_label(mod_dir)
            mod_path = os.path.join(self.root_dir, mod_dir)
            for fname in os.listdir(mod_path):
                if not fname.endswith('.csv'):
                    continue
                with open(os.path.join(mod_path, fname), 'r') as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader if any(row)]

                # 提取原始数据
                i_data = [float(row[0]) for row in rows]
                q_data = [float(row[1]) for row in rows]
                symbols = [int(row[2]) for row in rows if row[2].strip()]
                sym_width = float(rows[0][4])

                self.signal_data.append((i_data, q_data))
                self.symbol_seq.append(symbols)
                self.modulation_type.append(mod_type)
                self.symbol_width.append(sym_width)

    @staticmethod
    def modulation_to_label(mod_dir):
        """目录名转标签编码"""
        return {
            'BPSK': 1, 'QPSK': 2, '8PSK': 3, 'MSK': 4,
            '8QAM': 5, '16QAM': 6, '32QAM': 7,
            '8APSK': 8, '16APSK': 9, '32APSK': 10
        }[mod_dir]

    @staticmethod
    def default_phase_correction(iq_signal):
        """默认IQ相位校正：利用信号均值进行相位旋转"""
        i_data, q_data = iq_signal
        complex_signal = np.array(i_data) + 1j * np.array(q_data)

        if np.abs(complex_signal).max() < 1e-6:
            return iq_signal
        avg_phase = np.angle(np.mean(complex_signal))
        rot_matrix = np.array([[np.cos(avg_phase), np.sin(avg_phase)],
                               [-np.sin(avg_phase), np.cos(avg_phase)]])
        return rot_matrix @ np.stack([i_data, q_data])

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, idx):
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        # 实时相位校正
        iq_raw = self.signal_data[idx]
        # # 转换为tensor
        # iq_tensor = (torch.tensor(iq_raw[0], dtype=torch.float64),
        #             torch.tensor(iq_raw[1], dtype=torch.float64))
        # # 进行相位校正
        # iq_corrected = self.correct_iq_phase_func(iq_tensor)
        iq_corrected = self.correct_iq_phase_func(iq_raw)

        # 返回原始数据（标准化将在collate中处理）
        result = {
            'iq': (np.array(iq_corrected[0], dtype=np.float32),
                   np.array(iq_corrected[1], dtype=np.float32)),
            'symbol':  torch.tensor(self.symbol_seq[idx], dtype=torch.long),
            'mod_type': self.modulation_type[idx],
            'sym_width': self.symbol_width[idx],
            'data_len': len(iq_raw[0])
        }
        
        # 更新缓存
        self.cache[idx] = result
        
        return result
