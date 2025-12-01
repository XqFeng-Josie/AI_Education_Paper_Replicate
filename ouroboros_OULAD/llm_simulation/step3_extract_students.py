"""
步骤3：数据提取脚本

从已生成的VLE数据中按需提取指定数量的学生数据（200/500/1000），
确保格式与真实OULAD studentVle.csv一致。

Usage:
    python step3_extract_students.py --input studentVle_1000.csv --n_students 200 --output studentVle_200.csv
    python step3_extract_students.py --input studentVle_1000.csv --n_students 500 --output studentVle_500.csv
    python step3_extract_students.py --input studentVle_1000.csv --n_students 1000 --output studentVle_1000.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('step3_extract_students.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_student_completeness(vle_df, student_id, expected_days=set(range(1, 57))):
    """
    验证学生数据完整性（必须有完整的56天数据）
    
    Args:
        vle_df: VLE数据DataFrame
        student_id: 学生ID
        expected_days: 期望的日期集合（默认：1-56）
        
    Returns:
        bool: 是否完整
    """
    student_data = vle_df[vle_df['id_student'] == student_id]
    actual_days = set(student_data['date'].unique())
    return actual_days == expected_days


def extract_students(
    input_path,
    n_students,
    output_path,
    modules=None,
    strategy='balanced',
    random_seed=42,
    validate_completeness=True
):
    """
    从VLE数据中提取指定数量的学生
    
    Args:
        input_path: 输入VLE数据CSV路径
        n_students: 要提取的学生数（200/500/1000）
        output_path: 输出CSV路径
        modules: 模块列表（如果为None，自动检测）
        strategy: 提取策略 ('balanced': 按模块均匀分配, 'random': 随机采样)
        random_seed: 随机种子
        validate_completeness: 是否验证学生数据完整性（56天）
        
    Returns:
        DataFrame: 提取的数据
    """
    logger.info("=" * 80)
    logger.info("步骤3：提取学生数据")
    logger.info("=" * 80)
    logger.info(f"输入文件: {input_path}")
    logger.info(f"目标学生数: {n_students}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"策略: {strategy}")
    logger.info("=" * 80)
    
    # 加载数据
    logger.info(f"\n加载VLE数据...")
    vle_df = pd.read_csv(input_path)
    logger.info(f"✓ 总记录数: {len(vle_df)}")
    logger.info(f"✓ 总学生数: {vle_df['id_student'].nunique()}")
    
    # 验证必需列
    required_columns = ['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click']
    missing_columns = [col for col in required_columns if col not in vle_df.columns]
    if missing_columns:
        raise ValueError(f"输入数据缺少必需列: {missing_columns}")
    
    # 检测模块（如果未指定）
    if modules is None:
        modules = sorted(vle_df['code_module'].unique().tolist())
        logger.info(f"检测到模块: {', '.join(modules)}")
    else:
        # 只保留指定模块的数据
        vle_df = vle_df[vle_df['code_module'].isin(modules)]
        logger.info(f"使用指定模块: {', '.join(modules)}")
    
    # 获取所有学生ID
    all_student_ids = vle_df['id_student'].unique().tolist()
    logger.info(f"可用学生数: {len(all_student_ids)}")
    
    # 验证学生数据完整性（可选）
    if validate_completeness:
        logger.info(f"\n验证学生数据完整性（56天）...")
        expected_days = set(range(1, 57))
        complete_students = []
        incomplete_students = []
        
        for student_id in all_student_ids:
            if validate_student_completeness(vle_df, student_id, expected_days):
                complete_students.append(student_id)
            else:
                incomplete_students.append(student_id)
        
        logger.info(f"✓ 完整学生: {len(complete_students)}")
        if incomplete_students:
            logger.warning(f"⚠ 不完整学生: {len(incomplete_students)} (将被排除)")
            all_student_ids = complete_students
    
    if len(all_student_ids) < n_students:
        logger.warning(f"⚠ 可用学生数 ({len(all_student_ids)}) 少于目标数 ({n_students})")
        logger.warning(f"  将提取所有可用学生: {len(all_student_ids)}")
        n_students = len(all_student_ids)
    
    # 提取学生
    np.random.seed(random_seed)
    
    if strategy == 'balanced':
        # 按模块均匀分配
        logger.info(f"\n按模块均匀分配学生...")
        n_modules = len(modules)
        n_students_per_module = n_students // n_modules
        remainder = n_students % n_modules
        
        selected_students = []
        
        for module_idx, module in enumerate(modules):
            module_students = vle_df[vle_df['code_module'] == module]['id_student'].unique().tolist()
            
            # 计算该模块应提取的学生数
            module_n = n_students_per_module
            if module_idx < remainder:  # 余数分配给前几个模块
                module_n += 1
            
            if len(module_students) < module_n:
                logger.warning(f"  {module}: 可用 {len(module_students)}, 需要 {module_n}, 将提取所有")
                module_n = len(module_students)
            
            # 随机采样
            module_selected = np.random.choice(module_students, size=module_n, replace=False).tolist()
            selected_students.extend(module_selected)
            
            logger.info(f"  {module}: {module_n} 学生")
        
        # 如果总数不足，从所有模块中补充
        if len(selected_students) < n_students:
            remaining_needed = n_students - len(selected_students)
            remaining_students = [s for s in all_student_ids if s not in selected_students]
            if len(remaining_students) >= remaining_needed:
                additional = np.random.choice(remaining_students, size=remaining_needed, replace=False).tolist()
                selected_students.extend(additional)
                logger.info(f"  补充: {remaining_needed} 学生")
        
    elif strategy == 'random':
        # 随机采样
        logger.info(f"\n随机采样学生...")
        selected_students = np.random.choice(all_student_ids, size=n_students, replace=False).tolist()
    else:
        raise ValueError(f"未知策略: {strategy}")
    
    logger.info(f"✓ 已选择 {len(selected_students)} 个学生")
    
    # 提取这些学生的所有VLE记录
    extracted_df = vle_df[vle_df['id_student'].isin(selected_students)].copy()
    
    # 确保列顺序与OULAD一致
    oulad_columns = ['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click']
    extracted_df = extracted_df[oulad_columns]
    
    # 按学生和日期排序（保持一致性）
    extracted_df = extracted_df.sort_values(['id_student', 'date', 'id_site']).reset_index(drop=True)
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ 数据提取完成！")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"总记录数: {len(extracted_df)}")
    logger.info(f"学生数: {extracted_df['id_student'].nunique()}")
    logger.info(f"日期范围: {extracted_df['date'].min()} - {extracted_df['date'].max()}")
    
    # 按模块统计
    logger.info(f"\n各模块统计:")
    for module in modules:
        module_data = extracted_df[extracted_df['code_module'] == module]
        if len(module_data) > 0:
            n_students_module = module_data['id_student'].nunique()
            n_events_module = len(module_data)
            logger.info(f"  {module}: {n_students_module} 学生, {n_events_module} 事件")
    
    # 保存元数据
    metadata = {
        'input_file': str(input_path),
        'n_students_requested': n_students,
        'n_students_extracted': int(extracted_df['id_student'].nunique()),
        'n_events': len(extracted_df),
        'modules': modules,
        'strategy': strategy,
        'random_seed': random_seed,
        'date_range': {
            'min': int(extracted_df['date'].min()),
            'max': int(extracted_df['date'].max())
        },
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"元数据: {metadata_path}")
    
    return extracted_df


def main():
    parser = argparse.ArgumentParser(description="步骤3：从VLE数据中提取指定数量的学生")
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入VLE数据CSV文件路径（studentVle.csv格式）'
    )
    parser.add_argument(
        '--n_students',
        type=int,
        required=True,
        choices=[200, 500, 1000],
        help='要提取的学生数（200/500/1000）'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出CSV文件路径'
    )
    parser.add_argument(
        '--modules',
        type=str,
        nargs='+',
        default=None,
        help='模块列表（默认：自动检测）'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['balanced', 'random'],
        default='balanced',
        help='提取策略：balanced=按模块均匀分配, random=随机采样'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='跳过学生数据完整性验证（不推荐）'
    )
    
    args = parser.parse_args()
    
    try:
        extract_students(
            input_path=args.input,
            n_students=args.n_students,
            output_path=args.output,
            modules=args.modules,
            strategy=args.strategy,
            random_seed=args.seed,
            validate_completeness=not args.skip_validation
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 提取失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

