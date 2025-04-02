import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from optim_criter_sched_factory import OptimizerFactory, CriterionFactory, SchedulerFactory
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DatasetManager import DatasetManager
from sw_scores import evaluate_sw_score, sw_score_i, plot_pred_vs_actual
from MAPE_Loss_func import mape_loss
import torch.amp as amp

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message="1Torch was not compiled with flash attention.")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The PyTorch API of nested tensors is in prototype stage")

loss_curve_path = './loss_curve.png'


class CosmicTrainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 dataset_manager: DatasetManager,
                 use_amp: bool = True):
        """
        量子猫娘特制训练器

        Args:
            model: 要训练的模型
            device: 训练设备
            dataset_manager: 数据集管理引入
            use_amp: 是否启用混合精度训练
        """
        self.model = model.to(device)
        self.device = device
        self.dataset_manager = dataset_manager
        # 启用CUDA Graph优化
        self.enable_cuda_graph = torch.cuda.is_available()
        # 优化CUDA配置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # 混合精度配置
        if torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = amp.GradScaler(device='cuda',
                                           init_scale=2**16,
                                           growth_factor=1.5,
                                           backoff_factor=0.5,
                                           enabled=use_amp)
        self.history = {
            'stages': [],
            'train_loss': [],
            'val_loss': [],
            'sw_score': []
        }
        self.static_input = None
        self.static_graph = None

    def configure_stage(self,
                        stage_name: str,
                        optim_config: Dict[str, Any],
                        criter_config: Dict[str, Any],
                        sched_config: Dict[str, Any],
                        epochs: int,
                        early_stop: int
                        ):
        """配置训练阶段"""
        # 在Warmup阶段添加线性预热
        if stage_name == 'Warmup':
            sched_config['params']['pct_start'] = 0.3  # 30%步数用于预热
            sched_config['params']['anneal_strategy'] = 'linear'

        optimizer = OptimizerFactory.create(
            optim_type=optim_config['type'],
            model_params=self.model.parameters(),
            **optim_config['params']
        )
        criterion = CriterionFactory.create(
            criter_type=criter_config['type']
        )
        scheduler = SchedulerFactory.create(
            sched_type=sched_config['type'],
            optimizer=optimizer,
            **sched_config['params']
        )

        self.history['stages'].append({
            'name': stage_name,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'criterion': criterion,
            'epochs': epochs,
            'early_stop': early_stop,
            'current_epoch': 0,
            'best_sw_score': -np.inf
        })

    def cosmic_validate(self, val_loader):
        """ 验证函数 """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        # 添加验证进度条
        val_pbar = tqdm(total=len(val_loader), 
                        desc="\033[35m🌟 Validating\033[0m",  # 紫色
                        position=1, 
                        leave=False)

        with torch.inference_mode():
            for batch in val_loader:
                # 量子纠缠级设备同步
                iq = batch['iq'].to(self.device, non_blocking=True)
                widths = batch['sym_width'].to(self.device, non_blocking=True)
                data_len = batch['data_len'].to(self.device, non_blocking=True)

                # 维度稳定前向传播
                with amp.autocast(device_type='cuda',
                                        # dtype=self.amp_dtype,
                                        enabled=self.scaler.is_enabled()):
                    pred = self.model(iq, data_len)
                    loss = mape_loss(pred.squeeze(), widths)

                # 收集星际数据
                total_loss += loss.item()
                all_preds.extend(pred.cpu().numpy().flatten().tolist())
                all_targets.extend(widths.cpu().numpy().flatten().tolist())

                # 更新验证进度条
                val_pbar.update(1)
                val_pbar.set_postfix({
                    'loss': f"\033[31m{loss.item():.4f}\033[0m"  # 红色显示损失
                })

        # 计算宇宙评估指标
        sw_score = evaluate_sw_score(all_targets, all_preds)

        val_pbar.close()

        return {
            'val_loss': total_loss / len(val_loader),
            'sw_score': sw_score
        }
    

    def cosmic_train_step(self, stage_idx: int, batch: Dict) -> float:
        """阶段化训练步骤"""
        # 添加混合精度缓存清理
        # torch.cuda.empty_cache()
        stage = self.history['stages'][stage_idx]
        # 清空梯度
        stage['optimizer'].zero_grad()

        # 设备转移与混合精度
        # 使用多个CUDA流并行处理
        with torch.cuda.stream(torch.cuda.Stream()) as stream1:
            iq = batch['iq'].cuda(non_blocking=True)
            
        with torch.cuda.stream(torch.cuda.Stream()) as stream2:
            widths = batch['sym_width'].cuda(non_blocking=True)
            data_len = batch['data_len'].cuda(non_blocking=True)
    
        torch.cuda.synchronize()  # 确保数据准备完成

        # 正确顺序：前向传播 -> 计算损失 -> 反向传播 -> 参数更新
        with amp.autocast(device_type='cuda'):
            pred = self.model(iq, data_len)
            if torch.isnan(pred).any():
                print("检测到NaN损失值！")
                raise RuntimeError("检测到NaN损失，训练已终止")
            loss = mape_loss(pred.squeeze(), widths)

        # 添加NaN值检测
        if torch.isnan(loss).any():
            print("检测到NaN损失值！")
            # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            raise RuntimeError("检测到NaN损失，训练已终止")

        # 梯度管理
        self.scaler.scale(loss).backward()
        # 增强梯度裁剪
        self.scaler.unscale_(stage['optimizer'])
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(stage['optimizer'])
        self.scaler.update()

        # 更新学习率
        if isinstance(stage['scheduler'], OneCycleLR):
            stage['scheduler'].step()

        return loss.item()

    @staticmethod
    def plot_loss_curve(train_loss, val_loss, save_path):
        """
        用于绘制和保存损失曲线

        Args:
            train_loss: 训练损失值列表
            val_loss: 验证损失值列表
            save_path: LOSS图的保存路径.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
        plt.plot(val_loss, label='Validation Loss', color='red', linewidth=2)
        plt.title('Training and Validation Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def launch(self, train_loader, val_loader, stages_config: list, save_model_path: str):
        """赛博猫娘启动！"""
        # 预先一次性配置所有阶段
        for stage_cfg in stages_config:
            self.configure_stage(**stage_cfg)

        self.model.train()

        best_sw = -np.inf

        total_epochs = sum(s['epochs'] for s in stages_config)
        outer_pbar = tqdm(total=total_epochs, 
                     desc="\033[34m⭐ Overall Training Progress\033[0m",  # 蓝色
                     position=0)
        
        all_train_losses = []
        all_val_losses = []

        for stage_idx, stage_cfg in enumerate(stages_config):
            # 直接使用已配置好的阶段
            stage = self.history['stages'][stage_idx]
            stage_epochs = stage_cfg['epochs']
            early_stop = stage_cfg.get('early_stop', 10)

            no_improve = 0
            # tqdm.write(f"Current stage: {stage['name']}")
            tqdm.write(f"Initial lr: {stage['optimizer'].param_groups[0]['lr']:.2e}")
            for epoch in range(stage_epochs):
                inner_pbar = tqdm(total=len(train_loader), 
                            desc=f"\033[32m🚀 Stage {stage['name']} "
                                 f"Epoch {epoch + 1}/{stage_epochs}\033[0m",  # 绿色
                            position=1, 
                            leave=False)
                # 训练阶段
                self.model.train()
                epoch_loss = 0.0
                for train_batch in train_loader:
                    loss = self.cosmic_train_step(stage_idx, train_batch)
                    epoch_loss += loss
                    inner_pbar.update(1)
                    # 更新内层进度条信息
                    inner_pbar.set_postfix({
                        'loss': f"\033[31m{loss:.4f}\033[0m",  # 红色
                        'lr': f"\033[33m{stage['optimizer'].param_groups[0]['lr']:.2e}\033[0m"  # 黄色
                    })

                inner_pbar.close()

                train_avg_loss = epoch_loss / len(train_loader)
                self.history['train_loss'].append(train_avg_loss)
                all_train_losses.append(train_avg_loss)

                # 验证阶段
                self.model.eval()
                val_metrics = self.cosmic_validate(val_loader)
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['sw_score'].append(val_metrics['sw_score'])
                all_val_losses.append(val_metrics['val_loss'])

                outer_pbar.update(1)
                outer_pbar.set_postfix({
                    'Val_loss': f"\033[31m{val_metrics['val_loss']:.4f}\033[0m",  # 红色
                    'SW': f"\033[36m{val_metrics['sw_score']:.4f}\033[0m"  # 青色
                })

                # 更新学习率 (非OneCycle)
                if not isinstance(stage['scheduler'], OneCycleLR):
                    if isinstance(stage['scheduler'], ReduceLROnPlateau):
                        stage['scheduler'].step(val_metrics['val_loss'])
                    else:
                        stage['scheduler'].step()

                # 早停与保存
                if val_metrics['sw_score'] > best_sw:
                    best_sw = val_metrics['sw_score']
                    # 保存完整训练状态
                    torch.save({
                        'model_state': self.model.state_dict(),
                        'model_arch': self.model.__class__.__name__,
                        'scaler_state': self.scaler.state_dict(),
                        'dataset_stats': {
                            'mean': self.dataset_manager.mean,
                            'std': self.dataset_manager.std
                        }
                    }, save_model_path)
                    tqdm.write(f'\033[95mฅ(=✧ω✧=) Best score {best_sw:.4f} '
                          f'at epoch {epoch + 1}\033[0m')  # 粉色
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= early_stop:
                        remaining_epochs = stage_epochs - (epoch + 1)
                        outer_pbar.update(remaining_epochs)  # 提前更新剩余的进度
                        tqdm.write(f'\033[95m🚨 Stage {stage_idx + 1} early stopping at epoch {epoch + 1}\033[0m')
                        break

                # 更新进度条
                
                # tqdm.write(
                #     f"Stage {stage_idx + 1} | "
                #     f"LR: {stage['optimizer'].param_groups[0]['lr']:.2e} | "
                #     f"Valid loss: {val_metrics['val_loss']:.4f} | "
                #     f"Valid SW_Score: {val_metrics['sw_score']:.4f}"
                # )

            # 绘制LOSS曲线图
            self.plot_loss_curve(all_train_losses, all_val_losses, loss_curve_path)

        outer_pbar.close()
        print(f'ฅ(>ω<ฅ) Best Overall SW Score: {best_sw:.4f}')


class CosmicTester:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 dataset_manager: DatasetManager):
        """
        量子猫娘特制测试器

        Args:
            model: 要训练的模型
            device: 训练设备
            dataset_manager: 数据集管理引入
        """
        self.model = model.to(device)
        self.device = device
        self.dataset_manager = dataset_manager

    def quantum_test(self, test_loader: DataLoader):
        """ 启动星际测试协议 """
        self.model.eval()
        all_preds = []
        all_targets = []
        # 关闭梯度计算，节省资源
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='～(๑>ᴗ<๑) Testing'):
                iq = batch['iq'].to(self.device)
                data_len = batch['data_len'].to(self.device)
                widths = batch['sym_width'].numpy()

                outputs = self.model(iq, data_len)
                preds = outputs.cpu().numpy()

                all_preds.extend(preds.flatten())
                all_targets.extend(widths.flatten())

        # 生成全息报告
        self._generate_report(all_targets, all_preds)

    @staticmethod
    def _generate_report(targets, preds):
        """ 生成量子测试报告 """
        metrics = {
            'MAE': mean_absolute_error(targets, preds),
            'MSE': mean_squared_error(targets, preds),
            'R²': r2_score(targets, preds),
            'SW Score': evaluate_sw_score(targets, preds)
        }

        print("\n📊 量子测试报告:")
        for k, v in metrics.items():
            print(f"✨ {k}: {v:.4f}")

        # 绘制 预测vs真实值图表
        sw_scores = sw_score_i(targets, preds)
        plot_pred_vs_actual(targets, preds, sw_scores, pre_title='')
