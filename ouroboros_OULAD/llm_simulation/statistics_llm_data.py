"""
统计LLM Agent模拟生成的数据量

Usage:
    python statistics_llm_data.py --results_dir results/llm_agent
    python statistics_llm_data.py --results_dir results/llm_agent_stream
    python statistics_llm_data.py --results_dir results --output stats_table.csv
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_ndjson_lines(file_path):
    """统计NDJSON文件的行数（事件数）"""
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
        return count
    except Exception as e:
        logger.warning(f"无法读取文件 {file_path}: {e}")
        return 0


def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def extract_student_ids_from_ndjson(file_path):
    """从NDJSON文件中提取唯一学生ID"""
    student_ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get('id_student') or obj.get('student_id')
                    if sid:
                        student_ids.add(sid)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"无法解析文件 {file_path}: {e}")
    return student_ids


def analyze_stream_dir(stream_dir):
    """分析流式数据目录（llm_agent_stream格式）"""
    stats = []
    stream_dir = Path(stream_dir)
    
    if not stream_dir.exists():
        logger.warning(f"目录不存在: {stream_dir}")
        return stats
    
    # 查找所有模块目录
    module_dirs = [d for d in stream_dir.iterdir() if d.is_dir() and d.name in ['BBB', 'DDD', 'EEE', 'FFF']]
    
    if not module_dirs:
        logger.info("未找到模块目录，尝试查找实验目录...")
        return stats
    
    logger.info(f"找到 {len(module_dirs)} 个模块目录")
    
    for module_dir in sorted(module_dirs):
        module = module_dir.name
        events_file = module_dir / f"{module}_events_raw.ndjson"
        interactions_file = module_dir / f"{module}_interactions.ndjson"
        checkpoint_file = module_dir / f"{module}_checkpoint.json"
        
        if events_file.exists():
            # 统计事件数
            event_count = count_ndjson_lines(events_file)
            file_size = get_file_size_mb(events_file)
            
            # 提取学生ID和事件详情
            student_ids = extract_student_ids_from_ndjson(events_file)
            n_students = len(student_ids)
            
            # 统计每个学生的事件数
            student_event_counts = defaultdict(int)
            try:
                with open(events_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            sid = obj.get('id_student') or obj.get('student_id')
                            if sid:
                                student_event_counts[sid] += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
            
            avg_events_per_student = event_count / n_students if n_students > 0 else 0
            
            # 读取checkpoint信息（如果有）
            checkpoint_info = {}
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_info = json.load(f)
                except Exception:
                    pass
            
            # 统计交互数
            interaction_count = 0
            if interactions_file.exists():
                interaction_count = count_ndjson_lines(interactions_file)
            
            stats.append({
                'source': 'stream_dir',
                'module': module,
                'n_students': n_students,
                'n_events': event_count,
                'n_interactions': interaction_count,
                'avg_events_per_student': round(avg_events_per_student, 2),
                'file_size_mb': file_size,
                'events_file': str(events_file),
                'checkpoint': checkpoint_info.get('current_student_idx', 0) if checkpoint_info else None,
            })
            
            logger.info(f"  {module}: {n_students} 学生, {event_count} 事件, {interaction_count} 交互, {file_size:.2f} MB, 平均每学生 {avg_events_per_student:.1f} 事件")
    
    return stats


def analyze_experiment_dir(exp_dir):
    """分析实验输出目录（end_to_end_*格式）"""
    stats = []
    exp_dir = Path(exp_dir)
    
    if not exp_dir.exists():
        return stats
    
    # 解析实验目录名
    exp_name = exp_dir.name
    parts = exp_name.split('_')
    
    # 查找数据文件
    vle_logs_files = list(exp_dir.glob('vle_logs_*.csv'))
    features_files = list(exp_dir.glob('synthetic_features_*.csv'))
    labeled_files = list(exp_dir.glob('synthetic_features_*_labeled.csv'))
    metrics_files = list(exp_dir.glob('metrics_*.csv'))
    
    # 统计VLE logs
    for vle_file in vle_logs_files:
        try:
            df = pd.read_csv(vle_file)
            n_events = len(df)
            n_students = df['id_student'].nunique() if 'id_student' in df.columns else 0
            n_modules = df['code_module'].nunique() if 'code_module' in df.columns else 0
            file_size = get_file_size_mb(vle_file)
            
            # 提取学生数（从文件名）
            n_students_from_name = None
            for part in parts:
                if part.isdigit():
                    n_students_from_name = int(part)
                    break
            
            stats.append({
                'source': 'experiment',
                'experiment_name': exp_name,
                'file_type': 'vle_logs',
                'n_students': n_students if n_students > 0 else n_students_from_name,
                'n_events': n_events,
                'n_modules': n_modules,
                'file_size_mb': file_size,
                'file_path': str(vle_file),
            })
        except Exception as e:
            logger.warning(f"无法读取VLE logs文件 {vle_file}: {e}")
    
    # 统计特征文件
    for feat_file in features_files:
        try:
            df = pd.read_csv(feat_file)
            n_students = len(df)
            file_size = get_file_size_mb(feat_file)
            
            stats.append({
                'source': 'experiment',
                'experiment_name': exp_name,
                'file_type': 'features',
                'n_students': n_students,
                'n_events': None,
                'n_modules': None,
                'file_size_mb': file_size,
                'file_path': str(feat_file),
            })
        except Exception as e:
            logger.warning(f"无法读取特征文件 {feat_file}: {e}")
    
    # 统计标记文件
    for labeled_file in labeled_files:
        try:
            df = pd.read_csv(labeled_file)
            n_students = len(df)
            file_size = get_file_size_mb(labeled_file)
            
            # 统计提交率
            submitted_count = None
            if 'submitted' in df.columns:
                submitted_count = df['submitted'].sum()
            
            stats.append({
                'source': 'experiment',
                'experiment_name': exp_name,
                'file_type': 'labeled_features',
                'n_students': n_students,
                'n_events': None,
                'n_modules': None,
                'n_submitted': submitted_count,
                'file_size_mb': file_size,
                'file_path': str(labeled_file),
            })
        except Exception as e:
            logger.warning(f"无法读取标记文件 {labeled_file}: {e}")
    
    return stats


def analyze_results_dir(results_dir):
    """分析结果目录，支持多种格式"""
    results_dir = Path(results_dir)
    all_stats = []
    
    if not results_dir.exists():
        logger.error(f"目录不存在: {results_dir}")
        return all_stats
    
    logger.info(f"分析目录: {results_dir}")
    
    # 检查是否是流式数据目录
    module_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name in ['BBB', 'DDD', 'EEE', 'FFF']]
    if module_dirs:
        logger.info("检测到流式数据目录格式")
        all_stats.extend(analyze_stream_dir(results_dir))
    else:
        # 检查是否是实验目录
        exp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('end_to_end_')]
        if exp_dirs:
            logger.info(f"检测到 {len(exp_dirs)} 个实验目录")
            for exp_dir in sorted(exp_dirs):
                logger.info(f"  分析: {exp_dir.name}")
                all_stats.extend(analyze_experiment_dir(exp_dir))
        else:
            # 递归查找所有可能的目录
            logger.info("递归查找数据文件...")
            for subdir in results_dir.rglob('*'):
                if subdir.is_dir():
                    # 检查是否是模块目录
                    if subdir.name in ['BBB', 'DDD', 'EEE', 'FFF']:
                        parent_stats = analyze_stream_dir(subdir.parent)
                        if parent_stats:
                            all_stats.extend(parent_stats)
                            break
                    # 检查是否是实验目录
                    elif subdir.name.startswith('end_to_end_'):
                        all_stats.extend(analyze_experiment_dir(subdir))
    
    return all_stats


def create_summary_table(stats):
    """创建汇总表格"""
    if not stats:
        logger.warning("没有统计数据")
        return pd.DataFrame()
    
    df = pd.DataFrame(stats)
    
    # 如果是流式数据，按模块汇总
    if 'module' in df.columns:
        summary_rows = []
        
        # 总体统计
        total_students = df['n_students'].sum()
        total_events = df['n_events'].sum()
        total_interactions = df['n_interactions'].sum()
        total_size = df['file_size_mb'].sum()
        
        summary_rows.append({
            '类型': '总计',
            '模块': '全部',
            '学生数': total_students,
            '事件数': total_events,
            '交互数': total_interactions,
            '平均每学生事件数': f"{total_events / total_students:.1f}" if total_students > 0 else "0",
            '文件大小(MB)': f"{total_size:.2f}",
            '数据源': 'stream_dir'
        })
        
        # 按模块统计
        for _, row in df.iterrows():
            summary_rows.append({
                '类型': '模块',
                '模块': row['module'],
                '学生数': row['n_students'],
                '事件数': row['n_events'],
                '交互数': row['n_interactions'],
                '平均每学生事件数': f"{row.get('avg_events_per_student', 0):.1f}",
                '文件大小(MB)': f"{row['file_size_mb']:.2f}",
                '数据源': 'stream_dir'
            })
        
        return pd.DataFrame(summary_rows)
    
    # 如果是实验数据，按实验汇总
    elif 'experiment_name' in df.columns:
        summary_rows = []
        
        # 按实验分组
        for exp_name in df['experiment_name'].unique():
            exp_df = df[df['experiment_name'] == exp_name]
            
            # 提取VLE logs统计
            vle_df = exp_df[exp_df['file_type'] == 'vle_logs']
            if not vle_df.empty:
                vle_row = vle_df.iloc[0]
                summary_rows.append({
                    '实验名称': exp_name,
                    '文件类型': 'VLE Logs',
                    '学生数': vle_row['n_students'],
                    '事件数': vle_row['n_events'],
                    '模块数': vle_row['n_modules'],
                    '文件大小(MB)': f"{vle_row['file_size_mb']:.2f}",
                })
            
            # 提取特征统计
            feat_df = exp_df[exp_df['file_type'] == 'features']
            if not feat_df.empty:
                feat_row = feat_df.iloc[0]
                summary_rows.append({
                    '实验名称': exp_name,
                    '文件类型': 'Features',
                    '学生数': feat_row['n_students'],
                    '事件数': '-',
                    '模块数': '-',
                    '文件大小(MB)': f"{feat_row['file_size_mb']:.2f}",
                })
            
            # 提取标记特征统计
            labeled_df = exp_df[exp_df['file_type'] == 'labeled_features']
            if not labeled_df.empty:
                labeled_row = labeled_df.iloc[0]
                summary_rows.append({
                    '实验名称': exp_name,
                    '文件类型': 'Labeled Features',
                    '学生数': labeled_row['n_students'],
                    '事件数': '-',
                    '模块数': '-',
                    '提交数': labeled_row.get('n_submitted', '-'),
                    '文件大小(MB)': f"{labeled_row['file_size_mb']:.2f}",
                })
        
        return pd.DataFrame(summary_rows)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="统计LLM Agent模拟生成的数据量")
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='结果文件夹路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出CSV文件路径（默认：在结果目录下生成stats_table.csv）'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='输出详细统计（包含所有原始数据）'
    )
    
    args = parser.parse_args()
    
    # 分析数据
    stats = analyze_results_dir(args.results_dir)
    
    if not stats:
        logger.error("未找到任何数据文件")
        return 1
    
    # 创建汇总表格
    summary_df = create_summary_table(stats)
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(args.results_dir)
        output_path = results_dir / 'stats_table.csv'
    
    # 保存汇总表格
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"✓ 汇总表格已保存到: {output_path}")
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("数据统计汇总")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    # 如果要求详细输出，也保存详细数据
    if args.detailed:
        detailed_path = output_path.parent / f"{output_path.stem}_detailed.csv"
        detailed_df = pd.DataFrame(stats)
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 详细统计已保存到: {detailed_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

