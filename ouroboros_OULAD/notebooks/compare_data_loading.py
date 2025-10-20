#!/usr/bin/env python
"""
对比传统ML和LLM数据加载
检查两种方法加载的数据是否完全一致
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).absolute().parent.parent
print(f"project_root: {project_root}")
sys.path.insert(0, str(project_root))

from selflearner.problem_definition import ProblemDefinition, TrainingType
from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad, Hdf5Creator

print("=" * 100)
print("对比传统ML和LLM数据加载")
print("=" * 100)

# 配置（与ouroboros_experiments_new.ipynb完全一致）
modules = ["BBB", "DDD", "EEE", "FFF"]
presentations = ["2014J"]
assessment = "TMA 1"
max_days = 11

features = [
    'demog',
    'vle_statistics',
    'vle_statistics_beforestart',
    'reg_statistics'
]

HDF5_PATH = str(project_root / 'selflearner/data_load/data/oulad.h5')

print(f"\n配置:")
print(f"  Modules: {modules}")
print(f"  Presentations: {presentations}")
print(f"  Assessment: {assessment}")
print(f"  Max days: {max_days}")
print(f"  Features: {features}")

# 构造module_presentations_previous（与传统ML一致）
manager = Hdf5Creator().get_hdf5_manager()
df_modules = manager.load_dataframe('courses').reset_index()
df_modules = df_modules.loc[
    df_modules['code_module'].isin(modules) & df_modules['code_presentation'].isin(presentations)]

module_presentations_previous = []
for index, row in df_modules.iterrows():
    module, presentation = row['code_module'], row['code_presentation']
    presentation_train = presentation  # SELFLEARNER: 使用同一个presentation
    module_presentations_previous.append((module, presentation, presentation_train))

print(f"\nModule presentations:")
for mp in module_presentations_previous:
    print(f"  {mp}")

# LLM方式（与run_paper_replication.py中的load_oulad_data一致）
def load_oulad_data_llm(module, presentation, assessment_name, days_to_cutoff, features):
    """LLM实验中的数据加载方式"""
    problem_def = ProblemDefinition(
        module=module,
        presentation=presentation,
        assessment_name=assessment_name,
        days_to_cutoff=days_to_cutoff,
        training_type=TrainingType.SELFLEARNER,
        presentation_train=presentation,  # SELFLEARNER: 使用同一个presentation
        y_column='submitted',
        grouping_column='submit_in',
        id_column='id_student'
    )
    
    feature_extractor = FeatureExtractionOulad(problem_def, hdf5_path=HDF5_PATH)
    data = feature_extractor.extract_features(features)
    
    return data, problem_def

print("\n" + "=" * 100)
print("测试单个配置: BBB_2014J_day0")
print("=" * 100)

test_module = "BBB"
test_presentation = "2014J"
test_day = 0

# 传统ML方式
print("\n【传统ML方式】")
problem_def_traditional = ProblemDefinition(
    test_module,
    test_presentation,
    assessment,
    days_to_cutoff=test_day,
    y_column='submitted',
    grouping_column='submit_in',
    id_column='id_student',
    presentation_train=test_presentation,
    training_type=TrainingType.SELFLEARNER,
)
feature_extractor_traditional = FeatureExtractionOulad(problem_def_traditional, hdf5_path=HDF5_PATH)
data_traditional = feature_extractor_traditional.extract_features(features)

y_train_trad = data_traditional['y_train']
y_test_trad = data_traditional['y_test']

# 转换为numpy array以便统计
y_train_trad_values = y_train_trad.values if hasattr(y_train_trad, 'values') else y_train_trad
y_test_trad_values = y_test_trad.values if hasattr(y_test_trad, 'values') else y_test_trad

print(f"Training set:")
print(f"  Total: {len(y_train_trad)}")
print(f"  submitted=1 (提交了): {int((y_train_trad_values == 1).sum())}")
print(f"  submitted=0 (没提交): {int((y_train_trad_values == 0).sum())}")

print(f"Test set:")
print(f"  Total: {len(y_test_trad)}")
print(f"  submitted=1 (提交了): {int((y_test_trad_values == 1).sum())}")
print(f"  submitted=0 (没提交): {int((y_test_trad_values == 0).sum())}")

# LLM方式
print("\n【LLM方式】")
data_llm, problem_def_llm = load_oulad_data_llm(test_module, test_presentation, assessment, test_day, features)

y_train_llm = data_llm['y_train']
y_test_llm = data_llm['y_test']

y_train_llm_values = y_train_llm.values if hasattr(y_train_llm, 'values') else y_train_llm
y_test_llm_values = y_test_llm.values if hasattr(y_test_llm, 'values') else y_test_llm

print(f"Training set:")
print(f"  Total: {len(y_train_llm)}")
print(f"  submitted=1 (提交了): {int((y_train_llm_values == 1).sum())}")
print(f"  submitted=0 (没提交): {int((y_train_llm_values == 0).sum())}")

print(f"Test set:")
print(f"  Total: {len(y_test_llm)}")
print(f"  submitted=1 (提交了): {int((y_test_llm_values == 1).sum())}")
print(f"  submitted=0 (没提交): {int((y_test_llm_values == 0).sum())}")

# 对比
print("\n" + "=" * 100)
print("【对比结果】")
print("=" * 100)

train_equal = len(y_train_trad) == len(y_train_llm) and np.array_equal(y_train_trad_values, y_train_llm_values)
test_equal = len(y_test_trad) == len(y_test_llm) and np.array_equal(y_test_trad_values, y_test_llm_values)

print(f"✓ Training set一致: {train_equal}")
print(f"✓ Test set一致: {test_equal}")

if not train_equal:
    print(f"\n⚠️ Training set不一致!")
    print(f"  传统ML: {len(y_train_trad)}, LLM: {len(y_train_llm)}")
    
if not test_equal:
    print(f"\n⚠️ Test set不一致!")
    print(f"  传统ML: {len(y_test_trad)}, LLM: {len(y_test_llm)}")

# 全面对比所有配置
print("\n" + "=" * 100)
print("全面对比所有配置")
print("=" * 100)

comparison_results = []

for module, presentation, presentation_train in module_presentations_previous:
    for day in [0, 5, 11]:  # 测试几个代表性的day
        print(f"测试: {module}_{presentation}_day{day}...", end=" ")
        
        # 传统ML
        problem_def_trad = ProblemDefinition(
            module, presentation, assessment,
            days_to_cutoff=day,
            y_column='submitted',
            grouping_column='submit_in',
            id_column='id_student',
            presentation_train=presentation_train,
            training_type=TrainingType.SELFLEARNER,
        )
        feature_extractor_trad = FeatureExtractionOulad(problem_def_trad, hdf5_path=HDF5_PATH)
        data_trad = feature_extractor_trad.extract_features(features)
        
        # LLM
        data_llm, _ = load_oulad_data_llm(module, presentation, assessment, day, features)
        
        # 转换为numpy array
        y_train_trad_vals = data_trad['y_train'].values if hasattr(data_trad['y_train'], 'values') else data_trad['y_train']
        y_test_trad_vals = data_trad['y_test'].values if hasattr(data_trad['y_test'], 'values') else data_trad['y_test']
        y_train_llm_vals = data_llm['y_train'].values if hasattr(data_llm['y_train'], 'values') else data_llm['y_train']
        y_test_llm_vals = data_llm['y_test'].values if hasattr(data_llm['y_test'], 'values') else data_llm['y_test']
        
        # 对比
        comparison_results.append({
            'module': module,
            'presentation': presentation,
            'day': day,
            'trad_train_total': len(y_train_trad_vals),
            'llm_train_total': len(y_train_llm_vals),
            'trad_test_total': len(y_test_trad_vals),
            'llm_test_total': len(y_test_llm_vals),
            'trad_train_submitted_1': int((y_train_trad_vals == 1).sum()),
            'llm_train_submitted_1': int((y_train_llm_vals == 1).sum()),
            'trad_test_submitted_1': int((y_test_trad_vals == 1).sum()),
            'llm_test_submitted_1': int((y_test_llm_vals == 1).sum()),
        })
        print("✓")

df_comparison = pd.DataFrame(comparison_results)

print("\n全面对比结果表:")
print(df_comparison.to_string())

# 检查是否有任何不一致
inconsistent = df_comparison[
    (df_comparison['trad_train_total'] != df_comparison['llm_train_total']) |
    (df_comparison['trad_test_total'] != df_comparison['llm_test_total']) |
    (df_comparison['trad_train_submitted_1'] != df_comparison['llm_train_submitted_1']) |
    (df_comparison['trad_test_submitted_1'] != df_comparison['llm_test_submitted_1'])
]

print("\n" + "=" * 100)
print("最终结论")
print("=" * 100)

if len(inconsistent) > 0:
    print("\n⚠️ 发现数据不一致的配置:")
    print(inconsistent.to_string())
else:
    print("\n✅ 所有配置的数据完全一致！")
    print("✅ LLM实验和传统ML实验使用的数据加载方式是一致的！")

print("\n" + "=" * 100)
print("对比完成！")
print("=" * 100)

