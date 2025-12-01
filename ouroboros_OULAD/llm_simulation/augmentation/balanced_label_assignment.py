"""
平衡的标签分配策略

改进版：
1. 使用LR作为打标工具（baseline模型）
2. 考虑标签均衡性：基于预测概率进行概率采样，而非全部标为1
3. 保持合成数据标签分布接近真实数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent path
BASE_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_PATH))

# Add current path for imports
CURRENT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(CURRENT_PATH))

from augmentation.label_assignment import LabelAssigner


class BalancedLabelAssigner(LabelAssigner):
    """
    平衡的标签分配器
    
    改进点：
    1. 使用LR模型预测概率
    2. 基于概率进行采样，而非硬分类
    3. 控制正负样本比例，使其接近真实数据
    """
    
    def assign_labels_balanced(self, synthetic_df, target_positive_rate=None, 
                              strategy='probabilistic', save_path=None):
        """
        平衡的标签分配
        
        Args:
            synthetic_df: 合成数据DataFrame
            target_positive_rate: 目标正样本比例（None则使用真实数据的比例）
            strategy: 标签分配策略
                - 'probabilistic': 基于预测概率采样（推荐）
                - 'threshold': 使用阈值分类
                - 'proportional': 按比例采样
            save_path: 保存路径
            
        Returns:
            DataFrame: 带平衡标签的合成数据
        """
        print(f"\n{'='*60}")
        print(f"平衡标签分配 (策略: {strategy})")
        print(f"{'='*60}")
        
        if self.baseline_model is None:
            raise ValueError("Baseline模型未训练！请先调用train_baseline_model()")
        
        # 获取预测概率
        X_synthetic = self._prepare_features_for_prediction(synthetic_df)
        y_pred_proba = self.baseline_model.predict_proba(X_synthetic)[:, 1]
        
        # 确定目标正样本率
        if target_positive_rate is None:
            # 使用真实训练数据的比例
            from selflearner.problem_definition import ProblemDefinition, TrainingType
            from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad
            
            problem_def = ProblemDefinition(
                module=self.module,
                presentation=self.presentation,
                assessment_name=self.assessment_name,
                days_to_cutoff=self.days_to_cutoff,
                training_type=TrainingType.SELFLEARNER,
                presentation_train=self.presentation,
                y_column='submitted',
                grouping_column='submit_in',
                id_column='id_student'
            )
            
            extractor = FeatureExtractionOulad(problem_def, hdf5_path=str(BASE_PATH / 'selflearner' / 'data_load' / 'data' / 'oulad.h5'))
            data = extractor.extract_features(["demog"])
            
            # 合并训练集和测试集得到全部数据的分布
            y_all = pd.concat([data['y_train'], data['y_test']], axis=0, ignore_index=True)
            target_positive_rate = y_all['submitted'].mean()
            
            print(f"  使用真实数据的正样本率: {target_positive_rate:.1%}")
        else:
            print(f"  使用指定的正样本率: {target_positive_rate:.1%}")
        
        # 根据策略分配标签
        if strategy == 'probabilistic':
            y_pred = self._assign_probabilistic(y_pred_proba, target_positive_rate)
        elif strategy == 'threshold':
            y_pred = self._assign_threshold(y_pred_proba, target_positive_rate)
        elif strategy == 'proportional':
            y_pred = self._assign_proportional(y_pred_proba, target_positive_rate)
        else:
            raise ValueError(f"未知策略: {strategy}")
        
        # 添加到原始DataFrame
        synthetic_df_labeled = synthetic_df.copy()
        synthetic_df_labeled['submitted'] = y_pred
        synthetic_df_labeled['submitted_proba'] = y_pred_proba
        
        # 统计
        actual_rate = y_pred.mean()
        print(f"\n✓ 标签分配完成")
        print(f"  目标正样本率: {target_positive_rate:.1%}")
        print(f"  实际正样本率: {actual_rate:.1%}")
        print(f"  Submitted=1: {y_pred.sum()} ({actual_rate:.1%})")
        print(f"  Submitted=0: {len(y_pred) - y_pred.sum()} ({1-actual_rate:.1%})")
        print(f"  平均预测概率: {y_pred_proba.mean():.3f}")
        
        # 质量检查
        if abs(actual_rate - target_positive_rate) > 0.05:
            print(f"  ⚠️ 警告: 实际比例与目标比例差异较大 (>{5}%)")
        
        # 保存
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_df_labeled.to_csv(save_path, index=False)
            print(f"\n✓ 已保存到: {save_path}")
        
        print(f"{'='*60}")
        
        return synthetic_df_labeled
    
    def _prepare_features_for_prediction(self, synthetic_df):
        """准备用于预测的特征"""
        X_synthetic = synthetic_df.copy()
        
        # 移除非特征列
        for col in ['id_student', 'student_type', 'submitted', 'submitted_proba']:
            if col in X_synthetic.columns:
                X_synthetic = X_synthetic.drop(columns=[col])
        
        # 编码特征
        X_synthetic_encoded = self._encode_features(X_synthetic)
        
        # 确保特征列一致
        for col in self.baseline_model.feature_names_in_:
            if col not in X_synthetic_encoded.columns:
                X_synthetic_encoded[col] = 0
        
        X_synthetic_encoded = X_synthetic_encoded[self.baseline_model.feature_names_in_]
        X_synthetic_encoded = X_synthetic_encoded.fillna(0)
        
        return X_synthetic_encoded
    
    def _assign_probabilistic(self, y_pred_proba, target_rate):
        """
        策略1: 概率采样（推荐）
        
        基于预测概率进行伯努利采样：
        - 预测概率高的样本更可能被标为1
        - 预测概率低的样本更可能被标为0
        - 整体正样本率接近目标值
        """
        print(f"\n  策略: 概率采样")
        print(f"    - 基于LR预测概率进行伯努利采样")
        print(f"    - 高概率样本更可能为1，低概率样本更可能为0")
        
        # 伯努利采样
        y_pred = np.random.binomial(1, y_pred_proba)
        
        # 调整到目标比例（如果偏差太大）
        current_rate = y_pred.mean()
        if abs(current_rate - target_rate) > 0.05:
            print(f"    - 初始采样比例: {current_rate:.1%}，需调整")
            y_pred = self._adjust_to_target_rate(y_pred_proba, target_rate)
        
        return y_pred
    
    def _assign_threshold(self, y_pred_proba, target_rate):
        """
        策略2: 阈值分类
        
        选择合适的阈值，使得正样本率接近目标值
        """
        print(f"\n  策略: 阈值分类")
        print(f"    - 选择阈值使正样本率={target_rate:.1%}")
        
        # 找到合适的阈值
        sorted_proba = np.sort(y_pred_proba)[::-1]  # 降序
        n_positive = int(len(y_pred_proba) * target_rate)
        threshold = sorted_proba[n_positive] if n_positive < len(sorted_proba) else 0.5
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        print(f"    - 选定阈值: {threshold:.3f}")
        
        return y_pred
    
    def _assign_proportional(self, y_pred_proba, target_rate):
        """
        策略3: 比例采样
        
        直接按照目标比例随机采样，不考虑预测概率
        """
        print(f"\n  策略: 比例采样")
        print(f"    - 随机选择{target_rate:.1%}的样本标为1（忽略预测概率）")
        
        n = len(y_pred_proba)
        n_positive = int(n * target_rate)
        
        y_pred = np.zeros(n, dtype=int)
        positive_indices = np.random.choice(n, size=n_positive, replace=False)
        y_pred[positive_indices] = 1
        
        return y_pred
    
    def _adjust_to_target_rate(self, y_pred_proba, target_rate):
        """调整标签到目标比例（基于概率排序）"""
        n = len(y_pred_proba)
        n_positive = int(n * target_rate)
        
        # 按概率排序，选择top-k作为正样本
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_pred = np.zeros(n, dtype=int)
        y_pred[sorted_indices[:n_positive]] = 1
        
        return y_pred
    
    def compare_strategies(self, synthetic_df, target_positive_rate=None):
        """
        对比不同策略的效果
        """
        print(f"\n{'='*80}")
        print("对比不同标签分配策略")
        print(f"{'='*80}")
        
        strategies = ['probabilistic', 'threshold', 'proportional']
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"策略: {strategy}")
            print(f"{'='*60}")
            
            labeled_df = self.assign_labels_balanced(
                synthetic_df, 
                target_positive_rate=target_positive_rate,
                strategy=strategy,
                save_path=None
            )
            
            results[strategy] = {
                'positive_rate': labeled_df['submitted'].mean(),
                'mean_proba': labeled_df['submitted_proba'].mean(),
                'pos_proba': labeled_df[labeled_df['submitted']==1]['submitted_proba'].mean(),
                'neg_proba': labeled_df[labeled_df['submitted']==0]['submitted_proba'].mean()
            }
        
        # 打印对比
        print(f"\n{'='*80}")
        print("策略对比结果")
        print(f"{'='*80}")
        
        for strategy, metrics in results.items():
            print(f"\n{strategy}:")
            print(f"  正样本率: {metrics['positive_rate']:.1%}")
            print(f"  平均概率: {metrics['mean_proba']:.3f}")
            print(f"  正样本平均概率: {metrics['pos_proba']:.3f}")
            print(f"  负样本平均概率: {metrics['neg_proba']:.3f}")
        
        print(f"\n{'='*80}")
        print("推荐: probabilistic策略")
        print("  - 保留了LR预测的不确定性")
        print("  - 正样本倾向高概率，负样本倾向低概率")
        print("  - 整体分布更接近真实数据")
        print(f"{'='*80}")
        
        return results


if __name__ == "__main__":
    import os
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    
    print("="*80)
    print("测试平衡标签分配器")
    print("="*80)
    
    # 创建标签分配器
    assigner = BalancedLabelAssigner(
        module="BBB",
        presentation="2014J",
        assessment_name="TMA 1",
        days_to_cutoff=0
    )
    
    # 加载baseline数据
    print("\n加载baseline数据...")
    baseline_data = assigner.load_baseline_data()
    
    if baseline_data:
        # 训练baseline模型
        assigner.train_baseline_model(
            baseline_data['X_train'],
            baseline_data['y_train']
        )
        
        # 加载测试合成数据
        test_synth_path = Path(__file__).parent.parent / 'results' / 'full_20251109_221523' / 'synthetic_features_1000.csv'
        
        if test_synth_path.exists():
            df_synth = pd.read_csv(test_synth_path)
            print(f"\n加载合成数据: {len(df_synth)} 个学生")
            
            # 对比不同策略
            results = assigner.compare_strategies(df_synth, target_positive_rate=0.34)
            
            # 使用推荐策略生成最终标签
            print(f"\n\n{'='*80}")
            print("生成最终标签（使用probabilistic策略）")
            print(f"{'='*80}")
            
            output_path = test_synth_path.parent / 'synthetic_features_1000_labeled_balanced.csv'
            df_synth_labeled = assigner.assign_labels_balanced(
                df_synth,
                target_positive_rate=0.34,
                strategy='probabilistic',
                save_path=output_path
            )
            
            # 质量检查
            print(f"\n{'='*80}")
            print("质量检查")
            print(f"{'='*80}")
            
            print(f"\n原始（不平衡）标签:")
            print(f"  Submitted=1: 100.0%")
            
            print(f"\n平衡后的标签:")
            print(f"  Submitted=1: {df_synth_labeled['submitted'].mean():.1%}")
            print(f"  Submitted=0: {(1-df_synth_labeled['submitted'].mean()):.1%}")
            
            print(f"\n真实数据分布:")
            y_all = pd.concat([baseline_data['y_train'], baseline_data['y_test']], axis=0, ignore_index=True)
            print(f"  Submitted=1: {y_all['submitted'].mean():.1%}")
            print(f"  Submitted=0: {(1-y_all['submitted'].mean()):.1%}")
            
            diff = abs(df_synth_labeled['submitted'].mean() - y_all['submitted'].mean())
            print(f"\n差异: {diff:.1%}")
            if diff < 0.05:
                print("✓ 标签分布合理（差异<5%）")
            else:
                print("⚠️ 标签分布偏差较大（差异>5%）")
        else:
            print(f"错误: 找不到测试合成数据: {test_synth_path}")
    else:
        print("警告: 无法加载baseline数据，跳过测试")

