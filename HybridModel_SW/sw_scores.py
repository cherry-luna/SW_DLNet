import numpy as np
import matplotlib.pyplot as plt


def compute_sw_score(y_true, y_pred):
    """计算单个样本i的SW_score_i"""
    # 计算误差比 (ER)
    er = np.abs(y_true - y_pred) / y_true

    # 根据ER值计算SW_score_i
    if er < 0.05:
        SW_score_i = 100
    elif er > 0.20:
        SW_score_i = 0
    else:
        SW_score_i = 100 - (er - 0.05) / 0.15 * 100

    return SW_score_i


def evaluate_sw_score(y_true, y_pred):
    """计算所有样本的SW_score"""
    SW_scores = []

    # 将 y_true 和 y_pred 转换为 numpy 数组，防止传入非 numpy 数组类型
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 检查y_true和y_pred是否是标量
    if y_true.ndim == 1 and y_pred.ndim == 1:  # 一维数组
        for true, pred in zip(y_true, y_pred):
            SW_score_i = compute_sw_score(true, pred)
            SW_scores.append(SW_score_i)
    elif y_true.ndim == 0 and y_pred.ndim == 0:  # 标量
        SW_score_i = compute_sw_score(y_true, y_pred)
        SW_scores.append(SW_score_i)
    else:
        raise ValueError("y_true 和 y_pred 的维度不匹配")

    # 返回SW_score的平均值
    return np.mean(SW_scores)


def sw_score_i(y_true, y_pred):
    """构造所有样本的SW_score_i数组"""
    sw_scores = [compute_sw_score(true, pred) for true, pred in zip(y_true, y_pred)]
    return np.array(sw_scores)


def plot_pred_vs_actual(targets, predictions, sw_scores, pre_title):
    """
        绘制预测值 vs 实际值的对比图，并显示 residual 分布

        :param targets: 实际值
        :param predictions: 预测值
        :param sw_scores: 对应的 SW 分数
        :param pre_title: 图例标签，区分不同的预测回归类型
    """
    # 计算绝对误差
    residuals = np.array(targets) - np.array(predictions)
    residuals = residuals/np.abs(np.array(targets))

    # 确保 sw_scores 是一维数组
    sw_scores = np.array(sw_scores)

    # 绘制 Predicted vs Actual 图
    plt.figure(figsize=(12, 6))

    # 使用 SW_score 为每个点上色 (更高的 SW_score 用更深的颜色表示)
    scatter = plt.scatter(targets, predictions, c=sw_scores, cmap='coolwarm', alpha=0.6, edgecolors='w', s=100)

    # 理想预测直线
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color='red', linestyle='--',
             label='Ideal Prediction')

    # 添加色条，表示 SW_score
    plt.colorbar(scatter, label='SW_score')

    plt.title(f'{pre_title} - Predicted vs Actual Values(Colored by SW_score)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(f'{pre_title} - Predicted vs Actual Values.png')

    # 绘制 Residuals 分布图
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'{pre_title} - Absolute Error Distribution')
    plt.xlabel('(Actual - Predicted)/|Actual|')
    plt.ylabel('Frequency')
    plt.savefig(f'{pre_title} - Absolute Error Distribution.png')

    # 绘制 Error vs Actual 图
    plt.figure(figsize=(12, 6))
    plt.scatter(targets, residuals, c=sw_scores, cmap='coolwarm', alpha=0.6, edgecolors='w', s=100)
    plt.axhline(0, color='black', linestyle='--', label='Zero Error Line')
    plt.title(f'{pre_title} - Error vs Actual Values(Colored by SW_score)')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals (Error)')
    plt.legend()
    plt.colorbar(scatter, label='SW_score')
    plt.savefig(f'{pre_title} - Error vs Actual Values.png')

