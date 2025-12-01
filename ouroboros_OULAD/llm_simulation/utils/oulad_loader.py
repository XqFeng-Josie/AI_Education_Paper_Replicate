"""
加载真实OULAD数据的工具函数
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
BASE_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_PATH))

from selflearner.problem_definition import ProblemDefinition, TrainingType
from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad


def load_oulad_train_data(module="BBB", presentation="2014J", 
                           assessment_name="TMA 1", days_to_cutoff=0,
                           features=None, hdf5_path=None):
    """
    加载OULAD训练数据
    
    Args:
        module: 课程模块 (e.g., "BBB")
        presentation: 课程呈现 (e.g., "2014J")
        assessment_name: 评估名称 (e.g., "TMA 1")
        days_to_cutoff: 距离截止日的天数
        features: 特征列表
        hdf5_path: HDF5文件路径
        
    Returns:
        dict: {'x_train': DataFrame, 'y_train': DataFrame}
    """
    if features is None:
        features = ["demog", "vle_statistics", "reg_statistics"]
    
    if hdf5_path is None:
        hdf5_path = str(BASE_PATH / 'selflearner' / 'data_load' / 'data' / 'oulad.h5')
    
    # 定义问题
    problem_def = ProblemDefinition(
        module=module,
        presentation=presentation,
        assessment_name=assessment_name,
        days_to_cutoff=days_to_cutoff,
        training_type=TrainingType.SELFLEARNER,
        presentation_train=presentation,
        y_column='submitted',
        grouping_column='submit_in',
        id_column='id_student'
    )
    
    # 提取特征
    extractor = FeatureExtractionOulad(problem_def, hdf5_path=hdf5_path)
    data = extractor.extract_features(features)
    
    return {
        'x_train': data['x_train'],
        'y_train': data['y_train'],
        'problem_def': problem_def
    }


def load_oulad_test_data(module="BBB", presentation="2014J", 
                          assessment_name="TMA 1", days_to_cutoff=0,
                          features=None, hdf5_path=None):
    """
    加载OULAD测试数据
    
    Args:
        module: 课程模块
        presentation: 课程呈现
        assessment_name: 评估名称
        days_to_cutoff: 距离截止日的天数
        features: 特征列表
        hdf5_path: HDF5文件路径
        
    Returns:
        dict: {'x_test': DataFrame, 'y_test': DataFrame}
    """
    if features is None:
        features = ["demog", "vle_statistics", "reg_statistics"]
    
    if hdf5_path is None:
        hdf5_path = str(BASE_PATH / 'selflearner' / 'data_load' / 'data' / 'oulad.h5')
    
    # 定义问题
    problem_def = ProblemDefinition(
        module=module,
        presentation=presentation,
        assessment_name=assessment_name,
        days_to_cutoff=days_to_cutoff,
        training_type=TrainingType.SELFLEARNER,
        presentation_train=presentation,
        y_column='submitted',
        grouping_column='submit_in',
        id_column='id_student'
    )
    
    # 提取特征
    extractor = FeatureExtractionOulad(problem_def, hdf5_path=hdf5_path)
    data = extractor.extract_features(features)
    
    return {
        'x_test': data['x_test'],
        'y_test': data['y_test'],
        'problem_def': problem_def
    }


def get_oulad_feature_distribution(module="BBB", presentation="2014J",
                                    assessment_name="TMA 1"):
    """
    获取OULAD真实数据的特征分布统计
    
    Returns:
        dict: 包含各特征的分布统计
    """
    data = load_oulad_train_data(module, presentation, assessment_name)
    X_train = data['x_train']
    
    stats = {}
    
    # VLE统计特征分布
    vle_features = ['sum_click_fromvleopen', 'count_days_fromvleopen', 
                    'last_login_rel', 'sum_material_fromvleopen']
    
    for feat in vle_features:
        if feat in X_train.columns:
            stats[feat] = {
                'mean': X_train[feat].mean(),
                'std': X_train[feat].std(),
                'p25': X_train[feat].quantile(0.25),
                'p50': X_train[feat].quantile(0.50),
                'p75': X_train[feat].quantile(0.75),
                'p95': X_train[feat].quantile(0.95),
                'min': X_train[feat].min(),
                'max': X_train[feat].max()
            }
    
    # 标签分布
    y_train = data['y_train']
    stats['label_distribution'] = {
        'submitted': y_train['submitted'].mean(),
        'not_submitted': 1 - y_train['submitted'].mean()
    }
    
    return stats


def sample_demographics_from_oulad():
    """
    从真实OULAD数据中采样demographics特征
    
    Returns:
        dict: demographics特征字典
    """
    # 加载studentInfo.csv
    data_path = BASE_PATH / 'selflearner' / 'data_load' / 'data'
    df_info = pd.read_csv(data_path / 'studentInfo.csv')
    
    # 随机采样一个学生的demographics
    sample = df_info.sample(n=1).iloc[0]
    
    demographics = {
        'gender': sample['gender'],
        'age_band': sample['age_band'],
        'highest_education': sample['highest_education'],
        'imd_band': sample.get('imd_band', 'Unknown'),
        'region': sample['region'],
        'num_of_prev_attempts': int(sample['num_of_prev_attempts']),
        'disability': sample['disability']
    }
    
    return demographics


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("测试OULAD数据加载")
    print("="*80)
    
    # 加载训练数据
    train_data = load_oulad_train_data()
    print(f"\n训练集大小: {len(train_data['x_train'])}")
    print(f"特征数量: {len(train_data['x_train'].columns)}")
    print(f"标签分布: {train_data['y_train']['submitted'].value_counts().to_dict()}")
    
    # 获取特征分布
    stats = get_oulad_feature_distribution()
    print(f"\n特征分布统计:")
    for feat, stat in stats.items():
        if feat != 'label_distribution':
            print(f"  {feat}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")
    
    # 采样demographics
    demo = sample_demographics_from_oulad()
    print(f"\n采样的demographics: {demo}")

