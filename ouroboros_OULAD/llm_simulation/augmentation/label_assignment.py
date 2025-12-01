"""
Step 3: 标签分配 - 使用baseline模型为合成数据预测标签

严格遵循instruction.txt:
- 使用训练好的baseline模型
- 预测合成数据的submitted标签
- 避免测试集泄漏
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent path
BASE_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_PATH))

from sklearn.linear_model import LogisticRegression
from selflearner.problem_definition import ProblemDefinition, TrainingType
from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad


class LabelAssigner:
    """使用baseline模型为合成数据分配标签"""
    
    def __init__(self, module="BBB", presentation="2014J", 
                 assessment_name="TMA 1", days_to_cutoff=0,
                 random_seed=42):
        """
        初始化标签分配器
        
        Args:
            module: 课程模块
            presentation: 课程呈现
            assessment_name: 评估名称
            days_to_cutoff: 距离截止日天数
            random_seed: 随机种子
        """
        self.module = module
        self.presentation = presentation
        self.assessment_name = assessment_name
        self.days_to_cutoff = days_to_cutoff
        self.random_seed = random_seed
        self.baseline_model = None
        self.feature_columns = None
        
    def load_baseline_data(self, hdf5_path=None):
        """
        加载baseline训练数据
        
        Returns:
            dict: {'X_train', 'y_train', 'X_test', 'y_test'}
        """
        if hdf5_path is None:
            hdf5_path = str(BASE_PATH / 'selflearner' / 'data_load' / 'data' / 'oulad.h5')
        
        print(f"加载baseline数据: {self.module} {self.presentation} {self.assessment_name}")
        
        # 定义问题（与baseline完全一致）
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
        
        # 使用简化特征集（与我们的合成数据一致）
        features = ["demog", "vle_statistics", "reg_statistics"]
        
        try:
            extractor = FeatureExtractionOulad(problem_def, hdf5_path=hdf5_path)
            data = extractor.extract_features(features)
            
            X_train = data['x_train']
            y_train = data['y_train']
            X_test = data['x_test']
            y_test = data['y_test']
            
            print(f"✓ 训练集: {len(X_train)} 学生")
            print(f"✓ 测试集: {len(X_test)} 学生")
            print(f"✓ 特征数: {len(X_train.columns)}")
            print(f"✓ 训练集标签分布: {y_train['submitted'].value_counts().to_dict()}")
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
        except Exception as e:
            print(f"错误: 无法加载baseline数据: {e}")
            # 返回None，使用备用方案
            return None
    
    def train_baseline_model(self, X_train, y_train):
        """
        训练baseline模型（LogisticRegression）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        print(f"\n训练baseline模型...")
        
        # 保存特征列名（用于对齐）
        self.feature_columns = X_train.columns.tolist()
        
        # 移除id_student和student_type（如果存在）
        X_train_model = X_train.copy()
        for col in ['id_student', 'student_type']:
            if col in X_train_model.columns:
                X_train_model = X_train_model.drop(columns=[col])
        
        # 更新特征列
        self.feature_columns = X_train_model.columns.tolist()
        
        # 处理类别特征（one-hot encoding）
        X_train_encoded = self._encode_features(X_train_model)
        
        # 填充NaN（关键步骤！）
        X_train_encoded = X_train_encoded.fillna(0)
        
        # 训练Logistic Regression（与baseline一致）
        self.baseline_model = LogisticRegression(
            C=0.01,
            penalty='l2',
            max_iter=1000,
            random_state=self.random_seed
        )
        
        y_train_values = y_train['submitted'].values
        self.baseline_model.fit(X_train_encoded, y_train_values)
        
        # 评估训练集性能
        train_score = self.baseline_model.score(X_train_encoded, y_train_values)
        print(f"✓ Baseline模型训练完成")
        print(f"✓ 训练集准确率: {train_score:.4f}")
        
        return self.baseline_model
    
    def _encode_features(self, X):
        """编码类别特征"""
        X_encoded = X.copy()
        
        # 类别特征列
        categorical_cols = ['gender', 'age_band', 'highest_education', 
                           'region', 'disability']
        
        for col in categorical_cols:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
        # 处理列表特征（weekly_vle_clicks等）
        list_cols = ['weekly_vle_clicks', 'active_days_per_week', 'recency_gaps']
        expected_list_length = 8  # 8 weeks
        
        for col in list_cols:
            if col in X_encoded.columns:
                # 如果是字符串，解析为列表
                if X_encoded[col].dtype == 'object':
                    import ast
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
                
                # 确保所有列表长度一致（8周）
                def normalize_list(x):
                    if not isinstance(x, list):
                        return [0] * expected_list_length
                    if len(x) < expected_list_length:
                        # 如果长度不足，用0填充
                        return x + [0] * (expected_list_length - len(x))
                    elif len(x) > expected_list_length:
                        # 如果长度超过，截断
                        return x[:expected_list_length]
                    return x
                
                X_encoded[col] = X_encoded[col].apply(normalize_list)
                
                # 转换为多列（week_0, week_1, ...）
                expanded = pd.DataFrame(
                    X_encoded[col].tolist(),
                    index=X_encoded.index
                )
                expanded.columns = [f'{col}_{i}' for i in range(expected_list_length)]
                X_encoded = pd.concat([X_encoded, expanded], axis=1)
                X_encoded = X_encoded.drop(columns=[col])
        
        return X_encoded
    
    def assign_labels(self, synthetic_df, save_path=None):
        """
        为合成数据分配标签
        
        Args:
            synthetic_df: 合成数据DataFrame
            save_path: 保存路径
            
        Returns:
            DataFrame: 带标签的合成数据
        """
        print(f"\n为 {len(synthetic_df)} 个合成学生分配标签...")
        
        if self.baseline_model is None:
            raise ValueError("Baseline模型未训练！请先调用train_baseline_model()")
        
        # 对齐特征列
        X_synthetic = synthetic_df.copy()
        
        # 移除id_student和student_type
        for col in ['id_student', 'student_type']:
            if col in X_synthetic.columns:
                X_synthetic = X_synthetic.drop(columns=[col])
        
        # 编码特征
        X_synthetic_encoded = self._encode_features(X_synthetic)
        
        # 填充NaN值（与训练时一致）
        X_synthetic_encoded = X_synthetic_encoded.fillna(0)
        
        # 确保特征列一致
        # 添加缺失的列（填0）
        for col in self.baseline_model.feature_names_in_:
            if col not in X_synthetic_encoded.columns:
                X_synthetic_encoded[col] = 0
        
        # 选择相同的列并排序
        X_synthetic_encoded = X_synthetic_encoded[self.baseline_model.feature_names_in_]
        
        # 再次填充NaN（防止添加新列时产生NaN）
        X_synthetic_encoded = X_synthetic_encoded.fillna(0)
        
        # 预测标签和概率
        y_pred = self.baseline_model.predict(X_synthetic_encoded)
        y_pred_proba = self.baseline_model.predict_proba(X_synthetic_encoded)[:, 1]
        
        # 添加到原始DataFrame
        synthetic_df_labeled = synthetic_df.copy()
        synthetic_df_labeled['submitted'] = y_pred
        synthetic_df_labeled['submitted_proba'] = y_pred_proba
        
        # 统计
        print(f"✓ 标签分配完成")
        print(f"✓ Submitted=1: {y_pred.sum()} ({y_pred.mean():.1%})")
        print(f"✓ Submitted=0: {len(y_pred) - y_pred.sum()} ({1-y_pred.mean():.1%})")
        print(f"✓ 平均预测概率: {y_pred_proba.mean():.3f}")
        
        # 保存
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_df_labeled.to_csv(save_path, index=False)
            print(f"✓ 已保存到: {save_path}")
        
        return synthetic_df_labeled
    
    def quality_check(self, synthetic_df_labeled, real_y_train):
        """
        质量检查：对比合成数据和真实数据的标签分布
        
        Args:
            synthetic_df_labeled: 带标签的合成数据
            real_y_train: 真实训练集标签
        """
        print(f"\n{'='*60}")
        print("标签质量检查")
        print(f"{'='*60}")
        
        real_pos_rate = real_y_train['submitted'].mean()
        synth_pos_rate = synthetic_df_labeled['submitted'].mean()
        diff = abs(synth_pos_rate - real_pos_rate)
        
        print(f"真实数据标签分布: {real_pos_rate:.1%}")
        print(f"合成数据标签分布: {synth_pos_rate:.1%}")
        print(f"差异: {diff:.1%}")
        
        if diff < 0.10:
            print("✓ 标签分布合理（差异<10%）")
        elif diff < 0.20:
            print("⚠ 标签分布可接受（差异<20%）")
        else:
            print("✗ 标签分布偏差较大（差异>20%）")
        
        print(f"{'='*60}")


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("测试标签分配器")
    print("="*80)
    
    # 创建标签分配器
    assigner = LabelAssigner(
        module="BBB",
        presentation="2014J",
        assessment_name="TMA 1",
        days_to_cutoff=0
    )
    
    # 加载baseline数据
    baseline_data = assigner.load_baseline_data()
    
    if baseline_data:
        # 训练baseline模型
        assigner.train_baseline_model(
            baseline_data['X_train'],
            baseline_data['y_train']
        )
        
        # 加载测试合成数据
        test_synth_path = Path(__file__).parent.parent / 'data' / 'simulated_students' / 'test_cohort_20_features.csv'
        
        if test_synth_path.exists():
            df_synth = pd.read_csv(test_synth_path)
            print(f"\n加载合成数据: {len(df_synth)} 个学生")
            
            # 分配标签
            output_path = test_synth_path.parent / 'test_cohort_20_labeled.csv'
            df_synth_labeled = assigner.assign_labels(df_synth, save_path=output_path)
            
            # 质量检查
            assigner.quality_check(df_synth_labeled, baseline_data['y_train'])
        else:
            print(f"错误: 找不到测试合成数据: {test_synth_path}")
    else:
        print("警告: 无法加载baseline数据，跳过测试")

