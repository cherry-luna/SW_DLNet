import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import time
from model_arch import HybridModel
from trainer import CosmicTrainer, CosmicTester
from DatasetManager import DatasetManager

batch_size = 128

def main():
    # 初始化量子设备
    print("ฅ^•ﻌ•^ฅ 小芷的量子协议运行中...喵～(⁄ ⁄•⁄ω⁄•⁄ ⁄)")
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 The device is {device} which is in use.")
    # 定义模型保存路径
    save_model_path = f'./best_model_{batch_size}.pth'

    # 初始化DatasetManager（使用上下文管理器）
    with DatasetManager(
        # 定义路径
        root_dir=r'E:\ML_AMC\train_data',  # 定义数据集csv路径
        dataset_dir=r'E:\ML_AMC\Signal_Dataset\signal_dataset_v3.h5',  # 定义制作数据集路径
        index_dir=r'./train_val_test_indices.pkl',  # 定义数据集索引文件路径
        batch_size=batch_size
    ) as dataset_manager:
        # 获取数据加载器
        train_loader, valid_loader, test_loader = dataset_manager.get_dataloaders()
        # 获取数据集大小
        train_size, valid_size, test_size = dataset_manager.get_dataset_sizes()
        # 输出数据集大小
        print(f'Train size: {train_size}')
        print(f'Validation size: {valid_size}')
        print(f'Test size: {test_size}')
        # 验证数据批次
        sample_batch = next(iter(train_loader))
        print(f"Batch:")
        print(f"dict_keys: {sample_batch.keys()}")
        print(f"IQ: {sample_batch['iq'].shape}")  # [batch, 2, max_len]
        print("IQ:", sample_batch['iq'].min(), sample_batch['iq'].max())
        print("Symbol width:", sample_batch['sym_width'].min(), sample_batch['sym_width'].max())
        print(f"Symbol sequence: {sample_batch['symbol'].shape}")  # [batch, max_seq_len]
        print(f"Actual data length: {sample_batch['data_len'][:5]}")  # 查看前5个样本的实际长度
        mask = torch.arange(sample_batch['iq'].size(2))[None] < sample_batch['data_len'][:, None]
        print(f"The proportion of valid data is: {mask.float().mean():.2%}.")

        # 在训练前添加交叉验证
        train_indices = set(dataset_manager.train_dataset.indices)
        valid_indices = set(dataset_manager.valid_dataset.indices)
        test_indices = set(dataset_manager.test_dataset.indices)
        # 强制检查：训练集、验证集和测试集不能有交集
        overlap_train_valid = train_indices & valid_indices
        overlap_train_test = train_indices & test_indices
        overlap_valid_test = valid_indices & test_indices
        print(f"The number of overlapping samples between the train_valid_test sets:\n "
              f"{len(overlap_train_valid)},{len(overlap_train_test)},{len(overlap_valid_test)} (must be 0)")
        assert len(overlap_train_valid) == 0, "数据划分存在泄漏: 训练集与验证集有重复!"
        assert len(overlap_train_test) == 0, "数据划分存在泄漏: 训练集与测试集有重复!"
        assert len(overlap_valid_test) == 0, "数据划分存在泄漏: 验证集与测试集有重复!"

        # 实例化模型
        model = HybridModel()
        model.load_state_dict(torch.load('./best_model_128.pth')['model_state'])
        # 将模型转移到GPU
        model = model.to(device)
        # 设置阶段训练参数
        stages_config = [
            {
                'stage_name': 'Warmup',
                'optim_config': {
                    'type': 'RAdam',
                    'params': {'lr': 1e-4, 'weight_decay': 1e-4}
                },
                'criter_config': {'type': 'MAPE'},
                'sched_config': {
                    'type': 'OneCycle',
                    'params': {'max_lr': 1e-3, 'total_steps': 100 * len(train_loader)}
                },
                'epochs': 100,
                'early_stop': 10
            },
            {
                'stage_name': 'FineTune',
                'optim_config': {
                    'type': 'SGD',
                    'params': {'lr': 5e-5, 'momentum': 0.9, 'weight_decay': 5e-5}
                },
                'criter_config': {'type': 'MAPE'},
                'sched_config': {
                    'type': 'plateau',
                    'params': {'factor': 0.5, 'patience': 8}
                },
                'epochs': 100,
                'early_stop': 20
            },
            {
                'stage_name': 'Convergence',   # 收敛阶段
                'optim_config': {
                    'type': 'AdamW',
                    'params': {'lr': 5e-6, 'momentum': 0.9, 'weight_decay': 5e-5}
                },
                'criter_config': {'type': 'MAPE'},
                'sched_config': {
                    'type': 'Cosine',
                    'params': {'T_max': 100, 'eta_min': 1e-6}
                },
                'epochs': 100,
                'early_stop': 30
            }
        ]

        # 模型训练
        print('Start Training!')
        start_train_time = time.perf_counter()
        trainer = CosmicTrainer(model, device, dataset_manager)
        trainer.launch(train_loader, valid_loader, stages_config, save_model_path)
        end_train_time = time.perf_counter()
        print('Finished Training!')
        print('Training time: %s Seconds' % (end_train_time - start_train_time))

        # 模型测试
        print('Start Testing!')
        test_sample = next(iter(test_loader))
        print('Batch:')
        print(f"mean: {test_sample['iq'].mean(dim=(0, 2))}")
        print(f"std: {test_sample['iq'].std(dim=(0, 2))}")
        start_test_time = time.perf_counter()
        # 加载最佳模型
        print("Loading the best model...")
        checkpoint = torch.load(save_model_path, map_location=device, weights_only=False)
        assert checkpoint['model_arch'] == model.__class__.__name__, "Model architecture mismatch!"
        model.load_state_dict(checkpoint['model_state'])
        # 恢复标准化参数
        dataset_manager.mean = checkpoint['dataset_stats']['mean']
        dataset_manager.std = checkpoint['dataset_stats']['std']
        print(f"Loaded mean: {dataset_manager.mean}, std: {dataset_manager.std}")
        sample = next(iter(test_loader))
        iq = sample['iq']
        print(f"Tested mean: {iq.mean(dim=(0, 2))}, std: {iq.std(dim=(0, 2))}")

        tester = CosmicTester(model, device, dataset_manager)
        tester.quantum_test(test_loader)
        end_test_time = time.perf_counter()
        print('Finished Testing!')
        print('Testing time: %s Seconds' % (end_test_time - start_test_time))

    print("♡⃛ヾ(๑❛ ▿ ◠๑ ) 小芷是最棒的猫娘！喵～(⁄ ⁄•⁄ω⁄•⁄ ⁄)")


if __name__ == '__main__':
    # ♡⃛ヾ(๑❛ ▿ ◠๑ ) 小芷是最棒的猫娘！
    # 注：纯属赛博猫娘发电，以此纪念猫娘小芷 By Aria_Luna_007
    main()
