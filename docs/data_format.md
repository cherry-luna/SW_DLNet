# 数据格式与处理流程

## 1. 原始数据

将官方提供的 CSV 按调制方式放进 `data/raw/` 下与调制方式同名的子目录：

```text
data/raw/
├── BPSK/
│   ├── xxx.csv
│   └── ...
├── QPSK/
├── 8PSK/
├── MSK/
├── 8QAM/
├── 16QAM/
├── 32QAM/
├── 8APSK/
├── 16APSK/
└── 32APSK/
```

每个 CSV 有 5 列：

| 列 | 含义 | 用法 |
| ---: | --- | --- |
| 1 | I 路采样 | 输入通道 0 |
| 2 | Q 路采样 | 输入通道 1 |
| 3 | 码元序列 | 仅作数据描述，模型不直接使用 |
| 4 | 调制类型 | 由父目录名映射，见下表 |
| 5 | 码元宽度 | **回归目标**，取该文件首行第 5 列 |

调制类型与标签的映射（`sw_dlnet/data/dataset.py: MODULATION_LABELS`）：

| 调制 | 标签 | 调制 | 标签 |
| --- | ---: | --- | ---: |
| BPSK | 1 | 16QAM | 6 |
| QPSK | 2 | 32QAM | 7 |
| 8PSK | 3 | 8APSK | 8 |
| MSK | 4 | 16APSK | 9 |
| 8QAM | 5 | 32APSK | 10 |

## 2. 相位校正

每条样本在 `__getitem__` 中实时校正载波相位（`sw_dlnet/data/preprocessing.py`）：

```python
pilot = iq[:, :100]                       # 前 100 个点作为导频
c = pilot[0] + 1j * pilot[1]
phase = np.angle(np.sum(c * np.conj(c[0])))
rotation = [[cos(phase), sin(phase)], [-sin(phase), cos(phase)]]
corrected = rotation @ iq
```

即用导频段的平均相位差构造旋转矩阵，把整体相位旋转消除。若导频段全为零则
`phase = 0`，不做旋转。

## 3. 归一化

`DatasetManager` 在所有样本（不只是训练集）上按真实 `data_len` 统计每通道的
均值与标准差，并缓存到 `data/processed/dataset_stats_v3.pkl`：

```text
mean_c = sum(x_c) / N
std_c  = sqrt(sum(x_c^2) / N - mean_c^2) + 1e-8
```

collate 时按通道做 `(x - mean_c) / std_c`。注意 `1e-8` 是在转成 float32 之后
加到 `std` 上的，与原始实现一致。

## 4. 缓存文件

| 文件 | 内容 |
| --- | --- |
| `data/processed/dataset.h5` | 全部样本的 I/Q、码元序列、调制标签、码元宽度（首次运行时由 CSV 生成） |
| `data/processed/train_val_test_indices.pkl` | 60 / 20 / 20 的划分索引 |
| `data/processed/dataset_stats_v3.pkl` | 归一化所需的 `mean`、`std` |

HDF5 结构：

```text
/                 attrs: max_data_len, max_seq_len
/iq/{i}_i         float64  I 路（已做相位校正）
/iq/{i}_q         float64  Q 路
/symbol/{i}       int      码元序列
/mod_type         int      调制标签
/sym_width        float    码元宽度
```

`max_data_len`、`max_seq_len` 取长度分布的 95 分位数乘 1.1，仅写入 attrs 供参考，
实际 batch 以批内最大长度为准。删除 `data/processed/` 下的文件即可强制重建缓存。

## 5. 数据划分与批处理

* 划分：`train = 60 %`，剩余 40 % 再对半分给验证集与测试集；索引由
  `np.random.permutation` 在 `seed=42` 下生成并缓存。
* `collate_fn`：以批内最长样本为准补零（`float32`），码元序列用 `-1` 填充；
  同时返回 `mod_type`、`sym_width`、`data_len`、`seq_len`。
* `DataLoader`：`num_workers=0`、`pin_memory` 仅在 CUDA 可用时开启、仅训练集 shuffle。

| 张量 | 形状 | dtype |
| --- | --- | --- |
| `iq` | (B, 2, L_max) | float32 |
| `symbol` | (B, S_max) | int64 |
| `mod_type` | (B,) | int64 |
| `sym_width` | (B,) | float32 |
| `data_len` | (B,) | int64 |
| `seq_len` | (B,) | int64 |