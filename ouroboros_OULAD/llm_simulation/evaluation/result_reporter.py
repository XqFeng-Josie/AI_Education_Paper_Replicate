"""
Step 6: 结果报告 - 生成metrics_sim.csv和可视化

严格遵循instruction.txt:
- 输出格式: condition, seed, pr_auc, f1, precision, recall
- 对比baseline和增强版本的性能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ResultReporter:
    """结果报告器"""
    
    def __init__(self):
        """初始化报告器"""
        sns.set_style('whitegrid')
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def generate_metrics_csv(self, results, output_path):
        """
        生成metrics_sim.csv
        
        Args:
            results: 实验结果列表
            output_path: 输出路径
        """
        print(f"\n{'='*60}")
        print("生成metrics_sim.csv")
        print(f"{'='*60}")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 选择必需列（按instruction要求）
        required_cols = ['condition', 'seed', 'pr_auc', 'f1', 'precision', 'recall']
        optional_cols = ['model', 'train_size', 'test_size', 'roc_auc']
        
        # 检查必需列
        for col in required_cols:
            if col not in df.columns:
                print(f"警告: 缺少必需列 {col}")
        
        # 选择输出列
        output_cols = [col for col in required_cols + optional_cols if col in df.columns]
        df_output = df[output_cols]
        
        # 排序（按condition和model）
        df_output = df_output.sort_values(['condition', 'model'], ignore_index=True)
        
        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"✓ 已保存 {len(df_output)} 行结果")
        print(f"✓ 输出路径: {output_path}")
        print(f"{'='*60}")
        
        return df_output
    
    def print_summary_table(self, results_df):
        """
        打印汇总表格
        
        Args:
            results_df: 结果DataFrame
        """
        print(f"\n{'='*80}")
        print("实验结果汇总")
        print(f"{'='*80}")
        
        # 按condition和model分组
        summary = results_df.groupby(['condition', 'model']).agg({
            'pr_auc': 'mean',
            'f1': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'train_size': 'first'
        }).reset_index()
        
        # 打印表格
        print(f"\n{summary.to_string(index=False, float_format='%.4f')}")
        
        print(f"\n{'='*80}")
        
        return summary
    
    def analyze_improvement(self, results_df):
        """
        分析性能提升
        
        Args:
            results_df: 结果DataFrame
        """
        print(f"\n{'='*80}")
        print("性能提升分析")
        print(f"{'='*80}")
        
        # 按模型分组
        for model_name in results_df['model'].unique():
            print(f"\n模型: {model_name}")
            print("-" * 60)
            
            model_df = results_df[results_df['model'] == model_name]
            
            # 获取baseline性能
            baseline = model_df[model_df['condition'] == 'baseline']
            if len(baseline) == 0:
                print("  警告: 缺少baseline结果")
                continue
            
            baseline_prauc = baseline['pr_auc'].values[0]
            
            # 计算各条件的提升
            for condition in ['plus_200', 'plus_500', 'plus_1000']:
                cond_df = model_df[model_df['condition'] == condition]
                if len(cond_df) == 0:
                    continue
                
                cond_prauc = cond_df['pr_auc'].values[0]
                improvement = cond_prauc - baseline_prauc
                improvement_pct = (improvement / baseline_prauc) * 100 if baseline_prauc > 0 else 0
                
                status = "↑" if improvement > 0 else ("↓" if improvement < 0 else "→")
                
                print(f"  {condition:12s}: PR-AUC={cond_prauc:.4f} "
                      f"({status} {improvement:+.4f}, {improvement_pct:+.2f}%)")
        
        print(f"\n{'='*80}")
    
    def plot_comparison(self, results_df, output_dir):
        """
        生成对比图表
        
        Args:
            results_df: 结果DataFrame
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("生成可视化图表")
        print(f"{'='*60}")
        
        # 图1: PR-AUC对比（按condition）
        fig, ax = plt.subplots(figsize=(12, 6))
        
        conditions = ['baseline', 'plus_200', 'plus_500', 'plus_1000']
        x_labels = ['Baseline', '+200', '+500', '+1000']
        
        for model_name in results_df['model'].unique():
            model_df = results_df[results_df['model'] == model_name]
            
            prauc_values = []
            for cond in conditions:
                cond_df = model_df[model_df['condition'] == cond]
                if len(cond_df) > 0:
                    prauc_values.append(cond_df['pr_auc'].values[0])
                else:
                    prauc_values.append(np.nan)
            
            ax.plot(x_labels, prauc_values, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('PR-AUC', fontsize=12)
        ax.set_title('PR-AUC Comparison Across Conditions', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plot_path = output_dir / 'prauc_comparison.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  ✓ {plot_path.name}")
        
        # 图2: 性能提升热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算相对baseline的提升
        improvement_data = []
        models = results_df['model'].unique()
        conditions_aug = ['plus_200', 'plus_500', 'plus_1000']
        
        for model_name in models:
            model_df = results_df[results_df['model'] == model_name]
            baseline_df = model_df[model_df['condition'] == 'baseline']
            
            if len(baseline_df) == 0:
                improvement_data.append([0, 0, 0])
                continue
            
            baseline_prauc = baseline_df['pr_auc'].values[0]
            
            improvements = []
            for cond in conditions_aug:
                cond_df = model_df[model_df['condition'] == cond]
                if len(cond_df) > 0:
                    cond_prauc = cond_df['pr_auc'].values[0]
                    imp = ((cond_prauc - baseline_prauc) / baseline_prauc * 100)
                    improvements.append(imp)
                else:
                    improvements.append(0)
            
            improvement_data.append(improvements)
        
        improvement_df = pd.DataFrame(
            improvement_data,
            index=models,
            columns=['+200', '+500', '+1000']
        )
        
        sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'PR-AUC Improvement (%)'}, ax=ax)
        ax.set_title('Performance Improvement vs Baseline (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Synthetic Data Size', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        plot_path = output_dir / 'improvement_heatmap.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  ✓ {plot_path.name}")
        
        print(f"{'='*60}")
        
    def generate_full_report(self, results, output_dir):
        """
        生成完整报告
        
        Args:
            results: 实验结果列表
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("生成完整实验报告")
        print(f"{'='*80}")
        
        # 1. 保存CSV
        csv_path = output_dir / 'metrics_sim.csv'
        results_df = self.generate_metrics_csv(results, csv_path)
        
        # 2. 打印汇总
        summary_df = self.print_summary_table(results_df)
        
        # 3. 分析提升
        self.analyze_improvement(results_df)
        
        # 4. 生成图表
        self.plot_comparison(results_df, output_dir)
        
        # 5. 保存汇总表
        summary_path = output_dir / 'summary_table.csv'
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"\n✓ 汇总表已保存: {summary_path}")
        
        print(f"\n{'='*80}")
        print("报告生成完成！")
        print(f"{'='*80}")
        print(f"输出目录: {output_dir}")
        print(f"  - metrics_sim.csv (详细结果)")
        print(f"  - summary_table.csv (汇总表)")
        print(f"  - prauc_comparison.png (对比图)")
        print(f"  - improvement_heatmap.png (提升热力图)")
        print(f"{'='*80}")


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("测试结果报告器")
    print("="*80)
    
    # 创建mock结果
    mock_results = []
    
    conditions = ['baseline', 'plus_200', 'plus_500', 'plus_1000']
    models = ['LR', 'RF', 'XGB']
    
    np.random.seed(42)
    
    for cond in conditions:
        # 模拟性能提升
        if cond == 'baseline':
            boost = 0
        elif cond == 'plus_200':
            boost = 0.01
        elif cond == 'plus_500':
            boost = 0.02
        else:
            boost = 0.025
        
        for model in models:
            base_prauc = np.random.uniform(0.65, 0.75)
            
            mock_results.append({
                'condition': cond,
                'model': model,
                'seed': 42,
                'pr_auc': base_prauc + boost + np.random.uniform(-0.005, 0.005),
                'f1': np.random.uniform(0.6, 0.7),
                'precision': np.random.uniform(0.65, 0.75),
                'recall': np.random.uniform(0.55, 0.65),
                'train_size': 500 if cond == 'baseline' else 500 + int(cond.split('_')[1]) if 'plus' in cond else 500,
                'test_size': 100,
                'roc_auc': np.random.uniform(0.7, 0.8)
            })
    
    # 创建报告器
    reporter = ResultReporter()
    
    # 生成报告
    output_dir = Path(__file__).parent.parent / 'results' / 'test'
    reporter.generate_full_report(mock_results, output_dir)
    
    print("\n测试完成！")

