# 测试与持续集成

本工程用 `pytest` 覆盖数据管线、模型结构、指标实现和端到端训练流程，
并通过 GitHub Actions 在每次提交、每个 PR 和每个发布 tag 上自动执行。

## 快速运行

```bash
pip install -e ".[dev]"   # 安装 pytest 与 ruff
pytest -q                 # 全部测试，CPU 上约 15 秒
pytest -m "not slow" -q   # 只跑单元测试，约 3 秒
pytest -m slow -q         # 只跑端到端训练与命令行测试
ruff check .              # 风格检查
ruff format --check .     # 格式检查
```

Windows 上运行 PyTorch 前需要设置环境变量，否则 OpenMP 可能报
`OMP Error #15`：

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## 测试数据

`data/sample/` 是从比赛训练集中按调制方式各取 3 个 CSV 得到的真实子集
（10 类共 30 个文件，约 1 MB），随仓库一起提交，因此 CI 不需要下载数据集：

```text
data/sample/
├── BPSK/data_1.csv, data_10.csv, data_100.csv
├── QPSK/...
└── 32APSK/...
```

它与 `data/raw/` 结构完全一致，30 个样本按 60/20/20 划分得到 18 / 6 / 6，
足以覆盖缓存、归一化、批处理和一次完整的前向、反向与评估。样本仅用于测试，
不参与任何结果复现。

## 测试文件

| 文件 | 覆盖内容 |
| --- | --- |
| `tests/test_preprocessing.py` | 相位校正的旋转矩阵、幅度不变性、零导频、输入输出类型 |
| `tests/test_dataset.py` | CSV 解析、调制映射、码元序列、浮点精度、样本缓存、目录排序 |
| `tests/test_manager.py` | 60/20/20 划分、可复现性、HDF5 缓存往返、归一化与填充、统计量 |
| `tests/test_models.py` | 参数量 40,735,353、565 个 state_dict 键、三路下采样率、前向形状、初始化可复现、最近邻插值 |
| `tests/test_metrics.py` | SW Score 分段函数、逐样本与整体一致性、MAPE 分母约定 |
| `tests/test_factories.py` | 优化器 / 调度器 / 损失函数的默认参数与非法输入 |
| `tests/test_trainer.py` | 两轮训练 + 验证 + checkpoint 结构 + 测试报告与三张图（`slow`） |
| `tests/test_cli.py` | 以子进程方式跑通 `scripts/train.py`（`slow`） |

标记 `slow` 的测试会构建完整模型并做数次前向反向，在 4 核 CPU 上约 10 秒，
因此单独成组：提交级 CI 只跑快速测试，端到端任务再跑慢测试。

## 持续集成

`.github/workflows/ci.yml` 在 push 到 `main`、PR 和手动触发时运行：

| 任务 | 内容 |
| --- | --- |
| `lint` | `ruff check` 与 `ruff format --check` |
| `test` | Ubuntu / Windows × Python 3.10 / 3.12（Windows 只跑 3.12）上的快速测试 |
| `end-to-end` | 慢测试 + 用 `configs/sample.yaml` 在真实样本上训练 2 个 epoch，上传 `outputs/sample` 产物 |

`.github/workflows/sample-training.yml` 可手动触发一次完整的小规模训练，
用于在没有 GPU 的环境里验证流水线。

`.github/workflows/release.yml` 在推送 `v*` tag 时先跑完整测试，再构建
sdist / wheel 与 `sample-dataset.zip`，最后用自动生成的 release notes 发布
GitHub Release。

## 新增测试的约定

* 需要真实 CSV 的用例统一使用 `sample_root` fixture，避免依赖 `data/raw/`；
* 会写文件的用例使用 `tmp_path`，不要污染仓库内的缓存目录；
* 构建模型前调用 `torch.manual_seed(0)`，保证初始化可复现；
* 单次训练超过 5 秒的用例加 `@pytest.mark.slow`。
