"""
多模块标签分配 - 使用4个模块的联合数据训练baseline模型

严格遵循论文配置:
- 使用 BBB, DDD, EEE, FFF 四个模块的联合数据训练baseline
- 预测合成数据的submitted标签
- 避免测试集泄漏
"""

import os
# 必须在导入任何可能使用HDF5的模块之前设置环境变量
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent path
BASE_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_PATH))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from selflearner.problem_definition import ProblemDefinition, TrainingType
from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad


class MultiModuleLabelAssigner:
    """使用多模块联合数据训练baseline模型为合成数据分配标签"""
    
    def __init__(self, modules=["BBB", "DDD", "EEE", "FFF"], 
                 presentation="2014J", 
                 assessment_name="TMA 1", 
                 days_to_cutoff=0,
                 random_seed=42):
        """
        初始化多模块标签分配器
        
        Args:
            modules: 模块列表 (默认: ["BBB", "DDD", "EEE", "FFF"])
            presentation: 课程呈现 (默认: "2014J")
            assessment_name: 评估名称 (默认: "TMA 1")
            days_to_cutoff: 距离截止日天数 (默认: 0)
            random_seed: 随机种子 (默认: 42)
        """
        self.modules = modules
        self.presentation = presentation
        self.assessment_name = assessment_name
        self.days_to_cutoff = days_to_cutoff
        self.random_seed = random_seed
        self.baseline_model = None
        self.feature_columns = None
        self.model_feature_names = None
        self.scaler = None
        self.feature_groups = [
            "vle_statistics",
            "vle_statistics_beforestart",
            "vle_day_activity_type_flags",
            "vle_day_activity_type",
            "vle_day",
            "vle_day_flags",
        ]
        
    def load_baseline_data(self, hdf5_path=None):
        """
        加载多模块的baseline训练数据（联合4个模块）
        
        Returns:
            dict: {'X_train', 'y_train', 'X_test', 'y_test'}
        """
        if hdf5_path is None:
            hdf5_path = str(BASE_PATH / 'selflearner' / 'data_load' / 'data' / 'oulad.h5')
        
        print(f"加载多模块baseline数据: {', '.join(self.modules)} {self.presentation} {self.assessment_name}")
        
        # 使用与dailyExpSampling配置一致的特征集合
        features = self.feature_groups.copy()
        
        all_X_train = []
        all_y_train = []
        all_X_test = []
        all_y_test = []
        
        # 为每个模块加载数据并合并
        for module in self.modules:
            print(f"\n  加载模块 {module}...")
            
            # 定义问题（与baseline完全一致）
            problem_def = ProblemDefinition(
                module=module,
                presentation=self.presentation,
                assessment_name=self.assessment_name,
                days_to_cutoff=self.days_to_cutoff,
                training_type=TrainingType.SELFLEARNER,
                presentation_train=self.presentation,
                y_column='submitted',
                grouping_column='submit_in',
                id_column='id_student'
            )
            
            try:
                extractor = FeatureExtractionOulad(problem_def, hdf5_path=hdf5_path)
                data = extractor.extract_features(features)
                
                X_train = data['x_train']
                y_train = data['y_train']
                X_test = data['x_test']
                y_test = data['y_test']
                
                # 添加code_module列（如果不存在）
                if 'code_module' not in X_train.columns:
                    X_train['code_module'] = module
                if 'code_module' not in X_test.columns:
                    X_test['code_module'] = module
                
                all_X_train.append(X_train)
                all_y_train.append(y_train)
                all_X_test.append(X_test)
                all_y_test.append(y_test)
                
                print(f"    ✓ {module}: 训练集 {len(X_train)}, 测试集 {len(X_test)}")
                
            except Exception as e:
                print(f"    ✗ {module}: 加载失败 - {e}")
                continue
        
        if not all_X_train:
            print("错误: 无法加载任何模块的数据")
            return None
        
        # 合并所有模块的数据
        X_train_combined = pd.concat(all_X_train, axis=0, ignore_index=True)
        y_train_combined = pd.concat(all_y_train, axis=0, ignore_index=True)
        X_test_combined = pd.concat(all_X_test, axis=0, ignore_index=True)
        y_test_combined = pd.concat(all_y_test, axis=0, ignore_index=True)
        
        print(f"\n✓ 联合数据:")
        print(f"  训练集: {len(X_train_combined)} 学生")
        print(f"  测试集: {len(X_test_combined)} 学生")
        print(f"  特征数: {len(X_train_combined.columns)}")
        print(f"  训练集标签分布: {y_train_combined['submitted'].value_counts().to_dict()}")
        
        # 按模块统计
        if 'code_module' in X_train_combined.columns:
            print(f"\n  各模块训练集大小:")
            for module in self.modules:
                count = len(X_train_combined[X_train_combined['code_module'] == module])
                print(f"    {module}: {count} 学生")
        
        return {
            'X_train': X_train_combined,
            'y_train': y_train_combined,
            'X_test': X_test_combined,
            'y_test': y_test_combined
        }
    
    def train_baseline_model(self, X_train, y_train):
        """
        训练baseline模型（SVM-W-R）
        
        Args:
            X_train: 训练特征（多模块联合）
            y_train: 训练标签
        """
        print(f"\n训练多模块联合baseline模型 (SVM-W-R)...")

        X_train_model = X_train.copy()
        for col in ['id_student', 'student_type']:
            if col in X_train_model.columns:
                X_train_model = X_train_model.drop(columns=[col])

        X_train_encoded = self._encode_features(X_train_model)
        X_train_encoded = X_train_encoded.fillna(0)

        self.model_feature_names = X_train_encoded.columns.tolist()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train_encoded.values.astype(float))

        y_train_values = y_train['submitted'].values
        self.baseline_model = SVC(
            kernel='rbf',
            gamma='auto',
            C=1,
            probability=True,
            class_weight='balanced',
            random_state=self.random_seed
        )

        self.baseline_model.fit(X_scaled, y_train_values)

        train_score = self.baseline_model.score(X_scaled, y_train_values)
        print(f"✓ Baseline模型训练完成")
        print(f"✓ 训练集准确率: {train_score:.4f}")
        print(f"✓ 特征数: {len(self.model_feature_names)}")

        return self.baseline_model
    
    def _encode_features(self, X):
        """编码类别特征"""
        X_encoded = X.copy()
        
        # 类别特征列
        categorical_cols = [
            'gender',
            'age_band',
            'highest_education',
            'region',
            'disability',
            'code_module',
            'code_presentation',
        ]
        
        for col in categorical_cols:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
        # 处理列表特征（weekly_vle_clicks等）
        list_cols = ['weekly_vle_clicks', 'active_days_per_week', 'recency_gaps']
        for col in list_cols:
            if col in X_encoded.columns:
                # 如果是字符串，解析为列表
                if X_encoded[col].dtype == 'object':
                    import ast
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
                # 转换为多列（week_0, week_1, ...）
                expanded = pd.DataFrame(
                    X_encoded[col].tolist(),
                    index=X_encoded.index
                )
                expanded.columns = [f'{col}_{i}' for i in range(len(expanded.columns))]
                X_encoded = pd.concat([X_encoded, expanded], axis=1)
                X_encoded = X_encoded.drop(columns=[col])
        
        return X_encoded
    
    def assign_labels(self, synthetic_df, save_path=None):
        """
        为合成数据分配标签
        
        Args:
            synthetic_df: 合成数据DataFrame（包含code_module列）
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
        
        if self.model_feature_names is None or self.scaler is None:
            raise ValueError("Baseline模型缺少特征定义，请重新训练模型")

        # 确保特征列一致，缺失列填0
        for col in self.model_feature_names:
            if col not in X_synthetic_encoded.columns:
                X_synthetic_encoded[col] = 0
        
        # 选择相同的列并排序
        X_synthetic_encoded = X_synthetic_encoded[self.model_feature_names]
        X_synthetic_encoded = X_synthetic_encoded.fillna(0)
        X_scaled = self.scaler.transform(X_synthetic_encoded.values.astype(float))
        
        # 预测标签和概率
        y_pred = self.baseline_model.predict(X_scaled)
        y_pred_proba = self.baseline_model.predict_proba(X_scaled)[:, 1]
        
        # 添加到原始DataFrame
        synthetic_df_labeled = synthetic_df.copy()
        synthetic_df_labeled['submitted'] = y_pred
        synthetic_df_labeled['submitted_proba'] = y_pred_proba
        
        # 统计
        print(f"✓ 标签分配完成")
        print(f"✓ Submitted=1: {y_pred.sum()} ({y_pred.mean():.1%})")
        print(f"✓ Submitted=0: {len(y_pred) - y_pred.sum()} ({1-y_pred.mean():.1%})")
        print(f"✓ 平均预测概率: {y_pred_proba.mean():.3f}")
        
        # 按模块统计
        if 'code_module' in synthetic_df_labeled.columns:
            print(f"\n  各模块标签分布:")
            for module in self.modules:
                module_data = synthetic_df_labeled[synthetic_df_labeled['code_module'] == module]
                if len(module_data) > 0:
                    pos_rate = module_data['submitted'].mean()
                    print(f"    {module}: {len(module_data)} 学生, {pos_rate:.1%} submitted")
        
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
    import os
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    
    print("="*80)
    print("测试多模块标签分配器")
    print("="*80)
    
    # 创建多模块标签分配器
    assigner = MultiModuleLabelAssigner(
        modules=["BBB", "DDD", "EEE", "FFF"],
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
        
        print("\n测试完成！")

