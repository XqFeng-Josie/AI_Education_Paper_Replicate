"""
评估指标计算工具
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score
)


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    计算评估指标
    
    Args:
        y_true: 真实标签 (0/1)
        y_pred: 预测标签 (0/1)
        y_pred_proba: 预测概率 (0-1)
        
    Returns:
        dict: 包含各种评估指标
    """
    metrics = {}
    
    # 基础分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # 如果有概率预测，计算AUC
    if y_pred_proba is not None:
        try:
            # ROC-AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # PR-AUC (最重要的指标)
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        except ValueError as e:
            # 如果只有一个类别，AUC无法计算
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    # 样本数量
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(y_true.sum())
    metrics['n_negative'] = int(len(y_true) - y_true.sum())
    metrics['positive_rate'] = float(y_true.mean())
    
    return metrics


def compare_metrics(baseline_metrics, augmented_metrics):
    """
    对比baseline和augmented的指标
    
    Args:
        baseline_metrics: baseline的指标字典
        augmented_metrics: augmented的指标字典
        
    Returns:
        dict: 包含改进幅度的字典
    """
    comparison = {}
    
    key_metrics = ['pr_auc', 'roc_auc', 'f1', 'precision', 'recall']
    
    for metric in key_metrics:
        baseline_val = baseline_metrics.get(metric, 0)
        augmented_val = augmented_metrics.get(metric, 0)
        
        if baseline_val > 0:
            improvement = (augmented_val - baseline_val) / baseline_val * 100
            comparison[f'{metric}_improvement_%'] = improvement
        else:
            comparison[f'{metric}_improvement_%'] = 0.0
        
        comparison[f'{metric}_baseline'] = baseline_val
        comparison[f'{metric}_augmented'] = augmented_val
    
    return comparison


def print_metrics(metrics, title="Metrics"):
    """
    打印指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"PR-AUC:      {metrics.get('pr_auc', 0):.4f}")
    print(f"ROC-AUC:     {metrics.get('roc_auc', 0):.4f}")
    print(f"F1:          {metrics.get('f1', 0):.4f}")
    print(f"Precision:   {metrics.get('precision', 0):.4f}")
    print(f"Recall:      {metrics.get('recall', 0):.4f}")
    print(f"Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"Samples:     {metrics.get('n_samples', 0)}")
    print(f"Positive:    {metrics.get('n_positive', 0)} ({metrics.get('positive_rate', 0):.1%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 模拟数据
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 100)  # 30%正样本
    y_pred_proba = np.random.rand(100)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print_metrics(metrics, title="测试指标")
    
    # 对比指标
    baseline = {'pr_auc': 0.65, 'roc_auc': 0.75, 'f1': 0.55}
    augmented = {'pr_auc': 0.70, 'roc_auc': 0.78, 'f1': 0.60}
    comparison = compare_metrics(baseline, augmented)
    
    print(f"\n改进对比:")
    for key, val in comparison.items():
        if 'improvement' in key:
            print(f"  {key}: {val:+.2f}%")

