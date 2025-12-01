"""
Step 4: 数据集合并 - 创建增强训练集

严格遵循instruction.txt:
- 创建 train_plus_200, train_plus_500, train_plus_1000
- 仅增强TRAIN集，不修改TEST集
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class DatasetMerger:
    """数据集合并器 - 将合成数据与真实训练集合并"""
    
    def __init__(self, random_seed=42):
        """
        初始化合并器
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def create_all_augmented_datasets(self, real_train_df, synthetic_df, 
                                     output_dir, dataset_name="llm"):
        """
        创建所有增强数据集 (train_plus_200, train_plus_500, train_plus_1000)
        
        Args:
            real_train_df: 真实训练集DataFrame（包含submitted列）
            synthetic_df: 合成数据DataFrame（包含submitted列）
            output_dir: 输出目录
            dataset_name: 数据集名称前缀
            
        Returns:
            Dict: 包含所有增强数据集的字典
        """
        print(f"\n{'='*60}")
        print("创建增强数据集")
        print(f"{'='*60}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保合成数据有submitted列
        if 'submitted' not in synthetic_df.columns:
            raise ValueError("合成数据必须包含'submitted'列")
        
        # 确保真实训练集有submitted列
        if 'submitted' not in real_train_df.columns:
            raise ValueError("真实训练集必须包含'submitted'列")
        
        # 准备合成数据（移除不需要的列）
        synthetic_clean = synthetic_df.copy()
        for col in ['id_student', 'student_type', 'submitted_proba']:
            if col in synthetic_clean.columns:
                synthetic_clean = synthetic_clean.drop(columns=[col])
        
        # 对齐列（确保特征列一致）
        real_cols = set(real_train_df.columns)
        synth_cols = set(synthetic_clean.columns)
        
        # 添加缺失的列
        for col in real_cols - synth_cols:
            if col != 'submitted':
                synthetic_clean[col] = 0
        
        for col in synth_cols - real_cols:
            if col != 'submitted':
                real_train_df[col] = 0
        
        # 确保列顺序一致
        common_cols = sorted(list(real_cols | synth_cols))
        if 'submitted' in common_cols:
            common_cols.remove('submitted')
            common_cols.append('submitted')
        
        real_train_df = real_train_df[common_cols]
        synthetic_clean = synthetic_clean[common_cols]
        
        datasets = {}
        
        # 创建 baseline (仅真实数据)
        baseline_df = real_train_df.copy()
        baseline_path = output_dir / f"{dataset_name}_train_baseline.csv"
        baseline_df.to_csv(baseline_path, index=False)
        datasets['baseline'] = {
            'df': baseline_df,
            'path': baseline_path,
            'size': len(baseline_df)
        }
        print(f"✓ Baseline: {len(baseline_df)} 样本 -> {baseline_path}")
        
        # 创建 train_plus_200
        if len(synthetic_clean) >= 200:
            plus_200 = self._merge_datasets(real_train_df, synthetic_clean, n_synthetic=200)
            plus_200_path = output_dir / f"{dataset_name}_train_plus_200.csv"
            plus_200.to_csv(plus_200_path, index=False)
            datasets['plus_200'] = {
                'df': plus_200,
                'path': plus_200_path,
                'size': len(plus_200)
            }
            print(f"✓ Train+200: {len(plus_200)} 样本 -> {plus_200_path}")
        
        # 创建 train_plus_500
        if len(synthetic_clean) >= 500:
            plus_500 = self._merge_datasets(real_train_df, synthetic_clean, n_synthetic=500)
            plus_500_path = output_dir / f"{dataset_name}_train_plus_500.csv"
            plus_500.to_csv(plus_500_path, index=False)
            datasets['plus_500'] = {
                'df': plus_500,
                'path': plus_500_path,
                'size': len(plus_500)
            }
            print(f"✓ Train+500: {len(plus_500)} 样本 -> {plus_500_path}")
        
        # 创建 train_plus_1000
        if len(synthetic_clean) >= 1000:
            plus_1000 = self._merge_datasets(real_train_df, synthetic_clean, n_synthetic=1000)
            plus_1000_path = output_dir / f"{dataset_name}_train_plus_1000.csv"
            plus_1000.to_csv(plus_1000_path, index=False)
            datasets['plus_1000'] = {
                'df': plus_1000,
                'path': plus_1000_path,
                'size': len(plus_1000)
            }
            print(f"✓ Train+1000: {len(plus_1000)} 样本 -> {plus_1000_path}")
        
        print(f"{'='*60}\n")
        
        return datasets
    
    def _merge_datasets(self, real_df, synthetic_df, n_synthetic):
        """
        合并真实数据和合成数据
        
        Args:
            real_df: 真实训练集
            synthetic_df: 合成数据
            n_synthetic: 使用的合成样本数
            
        Returns:
            DataFrame: 合并后的数据集
        """
        # 随机采样合成数据
        if n_synthetic > len(synthetic_df):
            n_synthetic = len(synthetic_df)
        
        synthetic_sampled = synthetic_df.sample(n=n_synthetic, random_state=self.random_seed)
        
        # 合并
        merged = pd.concat([real_df, synthetic_sampled], axis=0, ignore_index=True)
        
        # 打乱顺序（保持随机种子）
        merged = merged.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return merged
    
    def validate_augmentation(self, datasets):
        """
        验证增强数据集
        
        Args:
            datasets: 数据集字典
        """
        print(f"\n{'='*60}")
        print("验证增强数据集")
        print(f"{'='*60}")
        
        for name, data in datasets.items():
            df = data['df']
            print(f"\n{name}:")
            print(f"  总样本数: {len(df)}")
            print(f"  特征数: {len(df.columns) - 1}")  # 减去submitted列
            
            if 'submitted' in df.columns:
                pos_rate = df['submitted'].mean()
                print(f"  Submitted=1: {df['submitted'].sum()} ({pos_rate:.1%})")
                print(f"  Submitted=0: {len(df) - df['submitted'].sum()} ({1-pos_rate:.1%})")
            
            # 检查缺失值
            missing = df.isnull().sum().sum()
            if missing > 0:
                print(f"  警告: 发现 {missing} 个缺失值")
            else:
                print(f"  ✓ 无缺失值")
        
        print(f"{'='*60}\n")


