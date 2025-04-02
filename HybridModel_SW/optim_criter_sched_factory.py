import torch
import torch.optim as optim
import torch.nn as nn
from MAPE_Loss_func import mape_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, CyclicLR, OneCycleLR


class OptimizerFactory:
    @staticmethod
    def create(optim_type: str, model_params, lr: float, **kwargs):
        """优化器量子工厂"""
        optim_type = optim_type.lower()
        if optim_type == 'sgd':
            return optim.SGD(
                model_params, lr=lr,
                momentum=kwargs.get('momentum', 0.9),
                nesterov=kwargs.get('nesterov', True),
                weight_decay=kwargs.get('weight_decay', 1e-4)
            )
        elif optim_type == 'adam':
            return torch.optim.Adam(
                model_params, lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                weight_decay=kwargs.get('weight_decay', 1e-4)
            )
        elif optim_type == 'adamw':
            return optim.AdamW(
                model_params, lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                weight_decay=kwargs.get('weight_decay', 1e-4)
            )
        elif optim_type == 'radam':
            return optim.RAdam(
                model_params, lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                weight_decay=kwargs.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optim_type}")


class CriterionFactory:
    @staticmethod
    def create(criter_type: str):
        """损失函数量子工厂"""
        criter_type = criter_type.lower()
        if criter_type == 'mape':
            return mape_loss
        elif criter_type == 'mse':
            return nn.MSELoss(reduction='mean')
        elif criter_type == 'huber':
            return nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unknown criterion type: {criter_type}")


class SchedulerFactory:
    @staticmethod
    def create(sched_type: str, optimizer, **kwargs):
        """调度器量子工厂"""
        sched_type = sched_type.lower()
        if sched_type == 'onecycle':
            return OneCycleLR(
                optimizer,
                max_lr=kwargs['max_lr'],
                total_steps=kwargs['total_steps'],
                pct_start=kwargs.get('pct_start', 0.3)
            )
        elif sched_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=kwargs['T_max'],
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif sched_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10)
            )
        elif sched_type == 'step':
            return StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.2)
            )
        elif sched_type == 'cyclic':
            return CyclicLR(
                optimizer,
                base_lr=kwargs.get('base_lr', 5e-5),
                max_lr=kwargs.get('max_lr', 1e-3),
                step_size_up=kwargs.get('step_size_up', 2000),
                mode=kwargs.get('mode', 'triangular2')
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")