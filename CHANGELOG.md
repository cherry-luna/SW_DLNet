# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)
的写法，版本号遵循语义化版本。

## [1.0.0] - 2026-10-29

### Added

* 可安装的 `sw_dlnet` 包（`pyproject.toml`）与 `dev` 可选依赖（pytest、ruff）；
* 30 个真实 CSV 组成的样本数据集 `data/sample/`，覆盖全部 10 种调制方式；
* 54 个 pytest 用例：数据管线、模型结构、指标、工厂、端到端训练与命令行；
* GitHub Actions：`ci.yml`（风格 + 多平台测试 + 真实样本端到端训练）、
  `sample-training.yml`（手动触发的小规模训练）、`release.yml`（tag 触发的发布）；
* `docs/testing.md` 与 `configs/sample.yaml`。

### Changed

* 按 `configs / data / docs / scripts / sw_dlnet` 重组目录，统一模块与函数命名；
* 超参外置到 YAML，训练入口支持 `--test-only` 等命令行覆盖；
* 合并重复的绘图代码，修复报告图标题与 colorbar 句柄问题；
* 合并优化器 / 调度器 / 损失工厂中的重复分支。

### Removed

* 从未生效的 CUDA Graph 占位、bf16 判断、`_normalize`、`transform` 参数；
* 仓库内的 `.idea/`、`EX/` 与旧的 `NewDLNet/` 目录。

### Fixed

* 报告图文件名带前导空格、第三张图 colorbar 误用第一张图句柄、figure 未关闭；
* 重复缓存的归一化统计量与划分索引路径不一致的问题。

### Verified

* 新旧实现的 `state_dict` 565 个键完全一致，参数逐位相同；
* 同一输入下前向输出最大绝对误差为 0，参数量均为 40,735,353；
* 测试集指标保持 MAE 0.0255 / MSE 0.0017 / R² 0.9672 / SW 91.7479。
