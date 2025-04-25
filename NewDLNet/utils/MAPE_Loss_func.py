import torch


def mape_loss(y_pred, y_true, epsilon=1e-8):
    """
    自定义 MAPE 损失函数

    :param y_pred: 模型输出，PyTorch 张量
    :param y_true: 真实标签，PyTorch 张量
    返回: MAPE 损失值
    """
    # 避免除以 0
    absolute_percentage_error = torch.abs((y_true - y_pred) / (y_true + epsilon))

    return torch.mean(absolute_percentage_error) * 100  # 转换为百分比

