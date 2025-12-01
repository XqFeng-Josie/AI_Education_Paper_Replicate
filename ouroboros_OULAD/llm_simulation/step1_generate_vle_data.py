"""
步骤1：模拟生成VLE数据

生成LLM Agent模拟的VLE数据，输出格式与真实OULAD studentVle.csv一致。

Usage:
    python step1_generate_vle_data.py --n_students 200 --modules BBB DDD EEE FFF
    python step1_generate_vle_data.py --n_students 500 --output_dir results/vle_data
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup environment
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from generators.llm.llm_client import LlamaClient
from generators.llm.simulation.course_simulator import CourseSimulator
from generators.llm.simulation.action_to_vle_mapper import ActionToVLEMapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('step1_generate_vle_data.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="步骤1：生成LLM模拟的VLE数据")
    
    parser.add_argument(
        '--n_students',
        type=int,
        required=True,
        help='总学生数（将平均分配到各模块）'
    )
    parser.add_argument(
        '--modules',
        type=str,
        nargs='+',
        default=['BBB', 'DDD', 'EEE', 'FFF'],
        help='要生成数据的模块列表'
    )
    parser.add_argument(
        '--llama_url',
        type=str,
        default='http://localhost:8001',
        help='Llama服务器URL'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/vle_data',
        help='输出目录'
    )
    parser.add_argument(
        '--stream_dir',
        type=str,
        default=None,
        help='流式输出目录（支持增量写入和恢复）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从checkpoint恢复（需要--stream_dir）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"vle_data_{args.n_students}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("步骤1：生成LLM模拟的VLE数据")
    logger.info("=" * 80)
    logger.info(f"总学生数: {args.n_students}")
    logger.info(f"模块: {', '.join(args.modules)}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)
    
    try:
        # 检查Llama服务器
        logger.info(f"检查Llama服务器连接: {args.llama_url}")
        llm_client = LlamaClient(base_url=args.llama_url)
        if not llm_client.is_healthy():
            logger.error(f"❌ Llama服务器未就绪: {args.llama_url}")
            return 1
        logger.info("✓ Llama服务器连接正常")
        
        # 计算每个模块的学生数
        modules = args.modules
        n_modules = len(modules)
        n_students_per_module = args.n_students // n_modules
        
        logger.info(f"\n每个模块学生数: {n_students_per_module}")
        
        all_vle_events = []
        
        # 为每个模块生成数据
        for module_idx, module in enumerate(modules):
            logger.info(f"\n{'='*60}")
            logger.info(f"模块 {module_idx+1}/{n_modules}: {module}")
            logger.info(f"{'='*60}")
            
            # 检查流式目录中的已完成学生
            # 重要：基于 interactions.ndjson 检查完整性，因为它是原始数据源
            # events_raw.ndjson 是从 interactions 中的 actions 转换生成的展开数据
            complete_students = set()
            incomplete_students = set()
            student_errors = set()  # 记录有错误的学生
            
            if args.stream_dir:
                import json as _json
                from collections import defaultdict
                stream_module_dir = Path(args.stream_dir) / module
                events_file = stream_module_dir / f"{module}_events_raw.ndjson"
                interactions_file = stream_module_dir / f"{module}_interactions.ndjson"
                
                # 第一步：从 interactions.ndjson 读取 student_daily_actions 来检查完整性
                student_days_from_interactions = defaultdict(set)  # 基于interactions
                student_max_day = {}
                
                if interactions_file.exists():
                    logger.info(f"  检查现有数据: {interactions_file}")
                    try:
                        with open(interactions_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = _json.loads(line)
                                    interaction_type = obj.get('type')
                                    
                                    # 检查错误记录
                                    if interaction_type in ['student_action_error', 'student_week_plan_error']:
                                        sid = obj.get('student_id')
                                        if sid:
                                            student_errors.add(sid)
                                            logger.warning(f"    发现错误记录: {sid} (type: {interaction_type})")
                                    
                                    # 只关注student_daily_actions（这是数据完整性的依据）
                                    if interaction_type == 'student_daily_actions':
                                        sid = obj.get('student_id')
                                        day = obj.get('day')
                                        status = obj.get('status', 'success')
                                        
                                        if sid and day is not None and status == 'success':
                                            student_days_from_interactions[sid].add(day)
                                            student_max_day[sid] = max(student_max_day.get(sid, 0), day)
                                except Exception:
                                    continue
                    except Exception as e:
                        logger.warning(f"  读取interactions文件失败: {e}")
                
                # 第二步：基于 interactions 检查完整性
                expected_days = set(range(1, 57))
                for sid, days_set in student_days_from_interactions.items():
                    if sid in student_errors:
                        # 有错误的学生视为不完整
                        incomplete_students.add(sid)
                        logger.warning(f"    学生 {sid}: 有生成错误，视为不完整")
                    elif days_set == expected_days:
                        # 有完整的56天数据
                        complete_students.add(sid)
                    else:
                        # 缺失某些天
                        incomplete_students.add(sid)
                        missing_days = expected_days - days_set
                        max_day = student_max_day.get(sid, 0)
                        logger.warning(f"    学生 {sid}: {len(days_set)}/56 天, 最大日期: {max_day}, 缺失: {sorted(list(missing_days))[:10]}{'...' if len(missing_days) > 10 else ''}")
                
                logger.info(f"  基于interactions检查结果: {len(complete_students)} 个完整学生, {len(incomplete_students)} 个不完整学生")
                
                # 第三步：如果有不完整学生，清理 events_raw（根据 interactions 的完整性来过滤）
                if incomplete_students and events_file.exists():
                    logger.warning(f"  发现 {len(incomplete_students)} 个不完整学生，清理 events_raw...")
                    
                    # 读取 events_raw，只保留完整学生的数据
                    complete_events = []
                    try:
                        with open(events_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = _json.loads(line)
                                    sid = obj.get('id_student') or obj.get('student_id')
                                    if not sid:
                                        continue
                                    
                                    # 只保留完整学生的events
                                    if sid in complete_students:
                                        # 只统计成功生成的事件
                                        status = obj.get('generation_status', 'success')
                                        if status == 'success':
                                            complete_events.append(obj)
                                except Exception:
                                    continue
                    except Exception as e:
                        logger.warning(f"  读取events文件失败: {e}")
                        complete_events = []
                    
                    # 备份原文件
                    backup_file = events_file.with_suffix('.ndjson.backup')
                    if not backup_file.exists():
                        import shutil
                        try:
                            shutil.copy2(events_file, backup_file)
                            logger.info(f"  已备份events: {backup_file}")
                        except Exception as e:
                            logger.warning(f"  备份失败: {e}")
                    
                    # 只保留完整学生的数据
                    try:
                        with open(events_file, 'w') as f:
                            for event in complete_events:
                                f.write(_json.dumps(event, ensure_ascii=False) + '\n')
                        logger.info(f"  已移除 {len(incomplete_students)} 个不完整学生的events，保留 {len(complete_students)} 个完整学生，共 {len(complete_events)} 个events")
                    except Exception as e:
                        logger.error(f"  写入events失败: {e}")
                
                logger.info(f"  现有完整学生: {len(complete_students)}")
                if student_errors:
                    logger.warning(f"  有错误的学生数: {len(student_errors)}")
            
            # 计算需要生成的学生数
            # 策略：优先填补中间缺失的索引，然后再从最大索引+1开始生成
            # 这样可以更高效地利用ID空间，更快达到目标学生数量
            from check_and_fix_data import extract_student_index
            
            students_needed = max(0, n_students_per_module - len(complete_students))
            missing_indices_to_fill = []
            next_continuous_offset = 0
            
            if complete_students:
                # 从完整学生中提取所有索引
                student_indices = []
                for sid in complete_students:
                    idx = extract_student_index(sid)
                    if idx >= 0:
                        student_indices.append(idx)
                
                if student_indices:
                    min_index = min(student_indices)
                    max_index = max(student_indices)
                    
                    # 检查是否有缺失的索引（中间有gap）
                    expected_indices = set(range(min_index, max_index + 1))
                    actual_indices = set(student_indices)
                    missing_indices = sorted(list(expected_indices - actual_indices))
                    
                    if missing_indices:
                        logger.info(f"  完整学生索引范围: {min_index} - {max_index}, 发现 {len(missing_indices)} 个缺失索引")
                        logger.info(f"  缺失索引: {missing_indices[:10]}{'...' if len(missing_indices) > 10 else ''}")
                        
                        # 优先填补缺失的索引（最多填补students_needed个）
                        missing_indices_to_fill = missing_indices[:students_needed]
                        if missing_indices_to_fill:
                            logger.info(f"  将优先填补 {len(missing_indices_to_fill)} 个缺失索引: {missing_indices_to_fill[:10]}{'...' if len(missing_indices_to_fill) > 10 else ''}")
                    
                    # 下一个连续索引位置（用于生成剩余学生）
                    next_continuous_offset = max_index + 1
                else:
                    # 无法提取索引，回退到使用数量
                    next_continuous_offset = len(complete_students)
                    logger.warning(f"  无法从学生ID提取索引，使用学生数量作为起始偏移: {next_continuous_offset}")
            else:
                next_continuous_offset = 0
            
            logger.info(f"  目标总数: {n_students_per_module}, 现有完整学生: {len(complete_students)}, 需生成: {students_needed}")
            if missing_indices_to_fill:
                logger.info(f"  填补缺失索引: {len(missing_indices_to_fill)} 个, 剩余需生成: {students_needed - len(missing_indices_to_fill)} 个")
            
            # 生成数据
            total_generated = 0
            
            # 第一步：填补缺失的索引（如果存在）
            if missing_indices_to_fill:
                logger.info(f"  步骤1: 填补 {len(missing_indices_to_fill)} 个缺失索引...")
                # 由于CourseSimulator只能生成连续的索引，我们需要逐个填补
                # 但为了效率，我们可以将连续的缺失索引分组，一次性生成
                from itertools import groupby
                
                # 将连续的缺失索引分组
                def consecutive_groups(seq):
                    groups = []
                    for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                        group = [x[1] for x in g]
                        groups.append((group[0], len(group)))  # (起始索引, 数量)
                    return groups
                
                consecutive_missing = consecutive_groups(missing_indices_to_fill)
                logger.info(f"  缺失索引分为 {len(consecutive_missing)} 个连续组")
                
                for group_start, group_size in consecutive_missing:
                    logger.info(f"    生成索引 {group_start} 到 {group_start + group_size - 1} ({group_size} 个学生)")
                    simulator = CourseSimulator(
                        n_students=group_size,
                        llama_server_url=args.llama_url,
                        random_seed=args.seed + module_idx + group_start,  # 使用不同的seed确保随机性
                        module_code=module,
                        stream_dir=(Path(args.stream_dir) / module) if args.stream_dir else None,
                        resume=bool(args.stream_dir and args.resume),
                        id_offset=group_start
                    )
                    
                    results = simulator.simulate_8_weeks()
                    all_vle_events.extend(results['vle_events'])
                    total_generated += group_size
                    logger.info(f"    ✅ 填补完成: {group_size} 个学生, 共 {results['total_events']} 个事件")
            
            # 第二步：生成剩余的学生（从最大索引+1开始）
            remaining_needed = students_needed - len(missing_indices_to_fill)
            if remaining_needed > 0:
                logger.info(f"  步骤2: 从索引 {next_continuous_offset} 开始生成剩余 {remaining_needed} 个学生...")
                simulator = CourseSimulator(
                    n_students=remaining_needed,
                    llama_server_url=args.llama_url,
                    random_seed=args.seed + module_idx + next_continuous_offset,
                    module_code=module,
                    stream_dir=(Path(args.stream_dir) / module) if args.stream_dir else None,
                    resume=bool(args.stream_dir and args.resume),
                    id_offset=next_continuous_offset
                )
                
                results = simulator.simulate_8_weeks()
                all_vle_events.extend(results['vle_events'])
                total_generated += remaining_needed
                logger.info(f"    ✅ 生成完成: {remaining_needed} 个学生, 共 {results['total_events']} 个事件")
            
            if total_generated > 0:
                logger.info(f"  ✅ {module} 模拟完成: 共生成 {total_generated} 个学生")
            else:
                logger.info(f"  ✅ {module} 跳过，已有足够的完整学生")
        
        # 转换为OULAD格式
        logger.info(f"\n{'='*60}")
        logger.info("转换为OULAD格式")
        logger.info(f"{'='*60}")
        
        mapper = ActionToVLEMapper(random_seed=args.seed)
        
        # 如果使用流式输出，从流式文件读取
        if args.stream_dir:
            import glob
            raw_files = []
            for module in modules:
                raw_files.extend(glob.glob(str(Path(args.stream_dir) / module / f"{module}_events_raw.ndjson")))
            
            raw_events = []
            for rf in raw_files:
                with open(rf, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            # 只保留成功生成的事件（排除错误）
                            if event.get('generation_status') == 'success':
                                raw_events.append(event)
                        except Exception:
                            continue
            
            # 使用模块代码转换
            oulad_events = []
            for module in modules:
                module_events = [e for e in raw_events if e.get('module_code') == module]
                oulad_module_events = mapper.convert_to_oulad_format(module_events, module_code=module)
                oulad_events.extend(oulad_module_events)
        else:
            # 非流式模式：从内存中的事件转换
            # 需要按模块分组转换
            oulad_events = []
            for module in modules:
                module_events = [e for e in all_vle_events if e.get('module_code') == module]
                oulad_module_events = mapper.convert_to_oulad_format(module_events, module_code=module)
                oulad_events.extend(oulad_module_events)
        
        # 保存为CSV（studentVle.csv格式）
        oulad_columns = ['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click']
        
        if not oulad_events:
            logger.warning("未生成任何VLE事件，输出为空。请检查上游模拟日志。")
            vle_df = pd.DataFrame(columns=oulad_columns + ['activity_type'])
        else:
            vle_df = pd.DataFrame(oulad_events)
            missing_cols = [col for col in oulad_columns if col not in vle_df.columns]
            if missing_cols:
                logger.warning(f"VLE事件缺少列 {missing_cols}，将填充NaN后继续导出。")
                for col in missing_cols:
                    vle_df[col] = pd.NA
        
        # 确保列顺序与OULAD一致
        vle_df_final = vle_df[oulad_columns].copy()
        
        vle_output_path = output_dir / f"studentVle_{args.n_students}.csv"
        vle_df_final.to_csv(vle_output_path, index=False)
        
        logger.info(f"✅ 已保存VLE数据: {vle_output_path}")
        logger.info(f"   总记录数: {len(vle_df_final)}")
        logger.info(f"   学生数: {vle_df_final['id_student'].nunique()}")
        logger.info(f"   日期范围: {vle_df_final['date'].min()} - {vle_df_final['date'].max()}")
        
        # 按模块统计
        logger.info(f"\n  各模块统计:")
        for module in modules:
            module_data = vle_df_final[vle_df_final['code_module'] == module]
            if len(module_data) > 0:
                n_students = module_data['id_student'].nunique()
                n_events = len(module_data)
                logger.info(f"    {module}: {n_students} 学生, {n_events} 事件")
        
        # 保存元数据
        metadata = {
            'n_students': args.n_students,
            'n_students_per_module': n_students_per_module,
            'modules': modules,
            'seed': args.seed,
            'timestamp': timestamp,
            'total_events': len(vle_df_final),
            'unique_students': int(vle_df_final['id_student'].nunique()),
            'date_range': {
                'min': int(vle_df_final['date'].min()),
                'max': int(vle_df_final['date'].max())
            }
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n✅ 步骤1完成！")
        logger.info(f"输出文件: {vle_output_path}")
        logger.info(f"元数据: {metadata_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 生成失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

