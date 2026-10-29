# 重构记录

本工程在**不改变算法与数值行为**的前提下，对原始实现做了目录重组、命名清理与
重复代码合并。下面记录改动内容与验证方式。

## 1. 目录对照

| 原始文件 | 新位置 |
| --- | --- |
| `main.py` | `scripts/train.py` + `configs/default.yaml` |
| `data/SignalDataset.py` | `sw_dlnet/data/dataset.py` |
| `data/DatasetManager.py` | `sw_dlnet/data/manager.py` |
| `data/IQPhase.py` | `sw_dlnet/data/preprocessing.py` |
| `data/random_seed.py` | `sw_dlnet/data/seeding.py` |
| `models/model_arch.py` | `sw_dlnet/models/blocks.py` + `sw_dlnet/models/hybrid.py` |
| `train/trainer.py` | `sw_dlnet/engine/trainer.py` |
| `utils/MAPE_Loss_func.py` | `sw_dlnet/utils/losses.py` |
| `utils/sw_scores.py` | `sw_dlnet/utils/metrics.py` |
| `utils/optim_criter_sched_factory.py` | `sw_dlnet/utils/factories.py` |
| `utils/画图.py` | `sw_dlnet/utils/plotting.py` |
| `draw/draw.py` | `scripts/visualize_model.py` |
| `Visualization/*.png` | `assets/results/` |
| `EX/*.png` | `assets/logs/` |

## 2. 命名调整

| 原名 | 新名 |
| --- | --- |
| `SelfSignalDataset` | `SignalDataset` |
| `CosmicTrainer` / `CosmicTester` | `Trainer` / `Tester` |
| `cosmic_train_step` | `train_step` |
| `cosmic_validate` | `validate` |
| `launch` | `fit` |
| `quantum_test` | `evaluate` |
| `correct_iq_phase_func` | `correct_iq_phase` |
| `sw_score_i` | `sw_scores` |
| `OptimizerFactory.create` 等 | `create_optimizer`、`create_scheduler`、`create_criterion` |

## 3. 删除的死代码

以下内容在原实现中从未生效，删除后不影响结果：

* `Trainer.enable_cuda_graph`、CUDA Graph 相关占位逻辑；
* `Trainer.amp_dtype`（bf16 判断链）——实际始终使用 `GradScaler` 默认的 fp16；
* `DatasetManager._normalize`、`SignalDataset.transform`、`default_phase_correction`；
* `configure_stage` 中注入的 `pct_start=0.3`、`anneal_strategy="linear"`——
  `pct_start` 本就是 `OneCycleLR` 的默认值，`anneal_strategy` 从未被工厂转发；
* 注释掉的历史代码和未使用的 import。

被保留的“看似冗余”项：`PositionAwareTransformer.input_norm` / `final_norm` 未参与
forward，但属于 state dict，保留以保证旧 checkpoint 可直接加载。

## 4. 等价性关键点

| 细节 | 处理方式 |
| --- | --- |
| `F.interpolate(x2, size=...)` | 保持默认 `mode="nearest"` |
| `std` 统计 | 使用默认无偏估计，先转 float32 再加 `1e-8` |
| MAPE | 真值在分母：`|(y_true - y_pred) / (y_true + 1e-8)| * 100` |
| SW 分数 | 对全部样本一次计算，而非逐 batch 求平均 |
| 早停 | 每个阶段独立计数，触发时补齐进度条剩余步数 |
| OneCycleLR | 每个 batch 更新一次，Phase 结束后不再 step |
| 随机性 | 建模前先 `next(iter(train_loader))`；模块创建顺序不变 |
| checkpoint | 键名仍为 `model_state`、`model_arch`、`scaler_state`、`dataset_stats` |

设备适配是唯一的行为变化：`amp.autocast` / `GradScaler` / `pin_memory` 改为按
`device` 决定，从而允许在没有 GPU 的机器上跑通；在 CUDA 上与此前完全一致。

## 5. 验证结果

* 以相同随机种子分别构造新旧模型：`state_dict` 共 **565** 个键，键名完全一致，
  参数逐位相同；
* 同一输入（`B=2, L=4096`）下前向输出最大绝对误差 **0.0**；
* 参数量均为 **40,735,353**；
* 在合成小数据集上跑通 `DatasetManager → Trainer.fit → Tester.evaluate` 全流程，
  产物（checkpoint、loss 曲线、指标 json、三张报告图）均按预期生成。