"""
特征映射器 - 将模拟日志转换为OULAD特征格式
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any


class FeatureMapper:
    """将模拟的学生行为日志转换为OULAD风格的特征"""
    
    def __init__(self):
        pass
    
    def map_cohort_to_features(self, simulation_log_path):
        """
        将整个队列的模拟日志转换为特征DataFrame
        
        Args:
            simulation_log_path: 模拟日志文件路径 (JSONL格式)
            
        Returns:
            pd.DataFrame: OULAD风格的特征DataFrame
        """
        # 加载模拟日志
        cohort_data = []
        with open(simulation_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                cohort_data.append(json.loads(line))
        
        print(f"加载了 {len(cohort_data)} 个学生的模拟日志")
        
        # 为每个学生提取特征
        features_list = []
        for student_data in cohort_data:
            features = self._extract_student_features(student_data)
            features_list.append(features)
        
        # 转换为DataFrame
        df_features = pd.DataFrame(features_list)
        
        print(f"✓ 提取了 {len(df_features)} 个学生的特征 ({len(df_features.columns)} 列)")
        return df_features
    
    def _extract_student_features(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从单个学生的模拟数据中提取OULAD特征
        
        Args:
            student_data: 学生模拟数据字典
            
        Returns:
            dict: OULAD特征字典
        """
        features = {}
        
        # 1. 基本信息
        features['id_student'] = student_data['student_id']
        features['student_type'] = student_data['student_type']  # 仅用于分析，不用于训练
        
        # 2. Demographics特征
        demo = student_data['demographics']
        features['gender'] = demo['gender']
        features['age_band'] = demo['age_band']
        features['highest_education'] = demo['highest_education']
        features['region'] = demo['region']
        features['num_of_prev_attempts'] = demo['num_of_prev_attempts']
        features['disability'] = demo['disability']
        
        # 3. 从8周数据中提取VLE统计特征
        weeks_data = student_data['weeks']
        
        # 3.1 周度特征 (len=8)
        weekly_vle_clicks = [week['total_clicks'] for week in weeks_data]
        active_days_per_week = [week['active_days'] for week in weeks_data]
        
        features['weekly_vle_clicks'] = weekly_vle_clicks
        features['active_days_per_week'] = active_days_per_week
        
        # 3.2 计算recency gaps (距上次登录天数)
        recency_gaps = self._calculate_recency_gaps(weeks_data)
        features['recency_gaps'] = recency_gaps
        
        # 3.3 累计VLE统计
        total_clicks = sum(weekly_vle_clicks)
        total_active_days = sum(active_days_per_week)
        
        features['sum_click_fromvleopen'] = total_clicks
        features['count_days_fromvleopen'] = total_active_days
        
        # 3.4 首次和最后登录
        first_login_day = self._find_first_login(weeks_data)
        last_login_day = self._find_last_login(weeks_data)
        
        features['first_login'] = first_login_day
        features['last_login'] = last_login_day
        features['last_login_rel'] = student_data['total_days'] - last_login_day if last_login_day >= 0 else 999
        
        # 3.5 材料访问量（估算）
        total_materials = sum(
            sum(interaction['resources_accessed'] for interaction in week['daily_interactions'])
            for week in weeks_data
        )
        features['sum_material_fromvleopen'] = total_materials
        
        # 3.6 平均点击 / 天
        if total_active_days > 0:
            features['avg_clicks_per_active_day'] = total_clicks / total_active_days
        else:
            features['avg_clicks_per_active_day'] = 0
        
        # 3.7 最长连续活跃天数
        features['consecutive_days'] = self._calculate_consecutive_days(weeks_data)
        
        # 4. A1评估特征 (Week 4)
        week_4_data = weeks_data[3]  # Week 4 (index 3)
        features['a1_submitted'] = 1 if week_4_data['a1_submitted'] else 0
        features['a1_score'] = week_4_data['a1_score'] if week_4_data['a1_score'] is not None else 0
        features['a1_submission_day'] = 28 if week_4_data['a1_submitted'] else -1  # Week 4结束是第28天
        
        # 5. Registration统计 (模拟：大部分学生提前注册)
        features['date_registration'] = np.random.randint(-30, 0)  # 课程开始前30天内注册
        
        return features
    
    def _calculate_recency_gaps(self, weeks_data: List[Dict]) -> List[int]:
        """
        计算每周的recency gap (距上次活跃的天数)
        
        Args:
            weeks_data: 8周数据列表
            
        Returns:
            List[int]: 8个recency gap值
        """
        recency_gaps = []
        last_active_day = -1
        
        for week_idx, week in enumerate(weeks_data):
            week_start_day = week_idx * 7
            
            if week['active_days'] > 0 and len(week['daily_interactions']) > 0:
                # 找到这周的第一个活跃天
                first_active_in_week = min(
                    interaction['absolute_day'] 
                    for interaction in week['daily_interactions']
                )
                
                if last_active_day < 0:
                    # 第一次活跃
                    gap = 0
                else:
                    gap = first_active_in_week - last_active_day
                
                recency_gaps.append(gap)
                
                # 更新最后活跃天
                last_active_in_week = max(
                    interaction['absolute_day'] 
                    for interaction in week['daily_interactions']
                )
                last_active_day = last_active_in_week
            else:
                # 这周没活跃
                if last_active_day < 0:
                    gap = week_start_day + 7  # 从课程开始算
                else:
                    gap = (week_start_day + 7) - last_active_day
                recency_gaps.append(gap)
        
        return recency_gaps
    
    def _find_first_login(self, weeks_data: List[Dict]) -> int:
        """找到首次登录日"""
        for week in weeks_data:
            if week['daily_interactions']:
                return min(i['absolute_day'] for i in week['daily_interactions'])
        return -1  # 从未登录
    
    def _find_last_login(self, weeks_data: List[Dict]) -> int:
        """找到最后登录日"""
        last_day = -1
        for week in weeks_data:
            if week['daily_interactions']:
                week_last = max(i['absolute_day'] for i in week['daily_interactions'])
                last_day = max(last_day, week_last)
        return last_day
    
    def _calculate_consecutive_days(self, weeks_data: List[Dict]) -> int:
        """计算最长连续活跃天数"""
        # 构建所有活跃天的列表
        active_days_set = set()
        for week in weeks_data:
            for interaction in week['daily_interactions']:
                active_days_set.add(interaction['absolute_day'])
        
        if not active_days_set:
            return 0
        
        active_days = sorted(active_days_set)
        
        # 找最长连续序列
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(active_days)):
            if active_days[i] == active_days[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        return max_consecutive
    
    def save_features(self, df_features, output_path):
        """保存特征到CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)
        print(f"✓ 特征已保存到: {output_path}")


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("测试特征映射器")
    print("="*80)
    
    # 查找测试日志文件
    test_log = Path(__file__).parent.parent / 'data' / 'simulation_logs' / 'test_cohort_20.jsonl'
    
    if not test_log.exists():
        print(f"错误: 测试日志不存在: {test_log}")
        print("请先运行 run_llm_end_to_end.py 生成LLM模拟数据")
    else:
        # 创建映射器
        mapper = FeatureMapper()
        
        # 转换特征
        df_features = mapper.map_cohort_to_features(test_log)
        
        print(f"\n特征列:")
        print(list(df_features.columns))
        
        print(f"\n前3个学生的特征样本:")
        print(df_features.head(3))
        
        print(f"\n关键特征统计:")
        print(f"  总点击数: mean={df_features['sum_click_fromvleopen'].mean():.1f}, "
              f"std={df_features['sum_click_fromvleopen'].std():.1f}")
        print(f"  总活跃天数: mean={df_features['count_days_fromvleopen'].mean():.1f}, "
              f"std={df_features['count_days_fromvleopen'].std():.1f}")
        print(f"  A1提交率: {df_features['a1_submitted'].mean():.1%}")
        
        # 保存特征
        output_path = Path(__file__).parent.parent / 'data' / 'simulated_students' / 'test_cohort_20_features.csv'
        mapper.save_features(df_features, output_path)

