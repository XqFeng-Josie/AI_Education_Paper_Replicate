"""
特征验证器 - 验证合成特征的有效性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FeatureValidator:
    """验证合成特征是否符合OULAD数据的约束和分布"""
    
    def __init__(self, config=None):
        """
        初始化验证器
        
        Args:
            config: 验证配置字典
        """
        if config is None:
            config = {
                'weekly_features_length': 8,
                'vle_clicks_max': 300,
                'active_days_max': 7,
                'recency_gap_max': 56,
                'check_zero_activity': True,
                'check_distribution_match': True,
                'distribution_tolerance': 0.15
            }
        self.config = config
    
    def validate_features(self, df_features: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证特征DataFrame
        
        Args:
            df_features: 特征DataFrame
            
        Returns:
            Tuple[bool, List[str]]: (是否通过验证, 错误信息列表)
        """
        errors = []
        
        # 1. 长度检查
        length_errors = self._check_lengths(df_features)
        errors.extend(length_errors)
        
        # 2. 值域检查
        range_errors = self._check_ranges(df_features)
        errors.extend(range_errors)
        
        # 3. 逻辑一致性检查
        if self.config.get('check_zero_activity', True):
            logic_errors = self._check_zero_activity_consistency(df_features)
            errors.extend(logic_errors)
        
        # 4. 缺失值检查
        missing_errors = self._check_missing_values(df_features)
        errors.extend(missing_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _check_lengths(self, df_features: pd.DataFrame) -> List[str]:
        """检查周度特征的长度"""
        errors = []
        expected_length = self.config['weekly_features_length']
        
        list_features = ['weekly_vle_clicks', 'active_days_per_week', 'recency_gaps']
        
        for feat in list_features:
            if feat in df_features.columns:
                for idx, val in enumerate(df_features[feat]):
                    if isinstance(val, list) and len(val) != expected_length:
                        errors.append(f"Row {idx}, {feat}: 长度={len(val)}, 期望={expected_length}")
        
        return errors
    
    def _check_ranges(self, df_features: pd.DataFrame) -> List[str]:
        """检查特征值域"""
        errors = []
        
        # VLE点击数
        if 'sum_click_fromvleopen' in df_features.columns:
            invalid = df_features[df_features['sum_click_fromvleopen'] < 0]
            if len(invalid) > 0:
                errors.append(f"sum_click_fromvleopen: {len(invalid)}个负值")
        
        # 活跃天数
        if 'count_days_fromvleopen' in df_features.columns:
            invalid = df_features[
                (df_features['count_days_fromvleopen'] < 0) | 
                (df_features['count_days_fromvleopen'] > 56)  # 8周最多56天
            ]
            if len(invalid) > 0:
                errors.append(f"count_days_fromvleopen: {len(invalid)}个超出范围[0, 56]")
        
        # 周度点击数
        if 'weekly_vle_clicks' in df_features.columns:
            vle_clicks_max = self.config['vle_clicks_max']
            for idx, weekly in enumerate(df_features['weekly_vle_clicks']):
                if isinstance(weekly, list):
                    if any(c < 0 for c in weekly):
                        errors.append(f"Row {idx}, weekly_vle_clicks: 包含负值")
                    if any(c > vle_clicks_max for c in weekly):
                        errors.append(f"Row {idx}, weekly_vle_clicks: 超过最大值{vle_clicks_max}")
        
        # 周度活跃天数
        if 'active_days_per_week' in df_features.columns:
            active_days_max = self.config['active_days_max']
            for idx, weekly in enumerate(df_features['active_days_per_week']):
                if isinstance(weekly, list):
                    if any(d < 0 or d > active_days_max for d in weekly):
                        errors.append(f"Row {idx}, active_days_per_week: 超出范围[0, {active_days_max}]")
        
        return errors
    
    def _check_zero_activity_consistency(self, df_features: pd.DataFrame) -> List[str]:
        """检查零活跃度的逻辑一致性"""
        errors = []
        
        if 'weekly_vle_clicks' in df_features.columns and 'active_days_per_week' in df_features.columns:
            for idx, (clicks, days) in enumerate(zip(
                df_features['weekly_vle_clicks'],
                df_features['active_days_per_week']
            )):
                if isinstance(clicks, list) and isinstance(days, list):
                    for week, (c, d) in enumerate(zip(clicks, days)):
                        # 如果活跃天数=0，点击数应该=0
                        if d == 0 and c != 0:
                            errors.append(f"Row {idx}, Week {week+1}: active_days=0 但 clicks={c}≠0")
        
        return errors
    
    def _check_missing_values(self, df_features: pd.DataFrame) -> List[str]:
        """检查关键特征的缺失值"""
        errors = []
        
        required_features = [
            'id_student', 'gender', 'age_band', 
            'sum_click_fromvleopen', 'count_days_fromvleopen',
            'a1_submitted'
        ]
        
        for feat in required_features:
            if feat not in df_features.columns:
                errors.append(f"缺失必需特征: {feat}")
            elif df_features[feat].isnull().any():
                n_missing = df_features[feat].isnull().sum()
                errors.append(f"{feat}: {n_missing}个缺失值")
        
        return errors
    
    def print_validation_report(self, df_features: pd.DataFrame):
        """打印验证报告"""
        print("="*80)
        print("特征验证报告")
        print("="*80)
        
        is_valid, errors = self.validate_features(df_features)
        
        if is_valid:
            print("✓ 所有验证通过！")
        else:
            print(f"✗ 发现 {len(errors)} 个错误:")
            for i, error in enumerate(errors[:10], 1):  # 最多显示10个
                print(f"  {i}. {error}")
            if len(errors) > 10:
                print(f"  ... 以及其他 {len(errors)-10} 个错误")
        
        print("="*80)


if __name__ == "__main__":
    # 测试代码
    from pathlib import Path
    
    print("="*80)
    print("测试特征验证器")
    print("="*80)
    
    # 加载测试特征
    test_features_file = Path(__file__).parent.parent / 'data' / 'simulated_students' / 'test_cohort_20_features.csv'
    
    if test_features_file.exists():
        df_features = pd.read_csv(test_features_file)
        
        # 解析列表字符串
        import ast
        for col in ['weekly_vle_clicks', 'active_days_per_week', 'recency_gaps']:
            if col in df_features.columns:
                df_features[col] = df_features[col].apply(ast.literal_eval)
        
        # 创建验证器
        validator = FeatureValidator()
        
        # 验证
        validator.print_validation_report(df_features)
    else:
        print(f"错误: 测试文件不存在: {test_features_file}")
        print("请先运行 features/mapper.py")

