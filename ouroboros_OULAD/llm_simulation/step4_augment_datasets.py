"""
Step 4: 数据增强 - 将真实训练集与合成数据融合，保持测试集不变

用途：
1. 重新加载真实OULAD baseline（4模块联合）训练/测试数据
2. 读取步骤2/3输出的带标签合成特征（支持批量处理多个day off）
3. 为每份合成数据创建 train_plus_{200,500,1000}（仅增强TRAIN）
4. 导出固定的baseline测试集，供后续步骤5复用

支持两种模式：
- 模式1：指定具体文件（--synthetic_csv）
- 模式2：自动发现目录中的所有带标签文件（--input_dir），支持批量处理0-11天的day off数据
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from augmentation.dataset_merger import DatasetMerger
from augmentation.multi_module_label_assignment import MultiModuleLabelAssigner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="步骤4：合并真实训练集与合成数据，创建 train_plus_* 数据集（支持批量处理多个day off）"
    )
    
    # 输入模式：二选一
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--synthetic_csv",
        type=str,
        nargs="+",
        help="步骤2/3输出的 synthetic_features_*_labeled.csv 路径，可一次传入多个",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="步骤2/3的输出目录，自动发现所有 *_labeled.csv 文件（支持批量处理0-11天的day off数据）",
    )
    
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=["BBB", "DDD", "EEE", "FFF"],
        help="Baseline使用的模块列表（默认：BBB DDD EEE FFF）",
    )
    parser.add_argument(
        "--presentation",
        type=str,
        default="2014J",
        help="课程呈现（默认：2014J）",
    )
    parser.add_argument(
        "--assessment_name",
        type=str,
        default="TMA 1",
        help="评估名称（默认：TMA 1）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/augmented_data",
        help="输出目录（默认：results/augmented_data）",
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="llm",
        help="输出文件名前缀，如 llm_1000_train_plus_500（默认：llm）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）",
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def attach_labels(
    features: pd.DataFrame, labels: pd.DataFrame, label_col: str = "submitted"
) -> pd.DataFrame:
    """将y中的标签列附加回特征表"""
    if label_col not in labels.columns:
        raise ValueError(f"标签数据缺少 {label_col} 列")

    features_aligned = features.reset_index(drop=True).copy()
    labels_aligned = labels[label_col].reset_index(drop=True)

    if label_col in features_aligned.columns:
        features_aligned = features_aligned.drop(columns=[label_col])

    if len(features_aligned) != len(labels_aligned):
        raise ValueError("特征与标签行数不一致，无法对齐")

    features_aligned[label_col] = labels_aligned
    return features_aligned


def extract_day_off_from_path(path: Path) -> Optional[int]:
    """
    从文件路径中提取 day off (days_to_cutoff) 信息
    
    支持多种格式：
    - 目录名：labeled_200_day_sweep_f01_d04_r001 -> 4
    - 文件名：synthetic_features_200_day4_labeled.csv -> 4
    - 目录名：labeled_200_20240101_d00 -> 0
    
    Returns:
        days_to_cutoff 值，如果无法提取则返回 None
    """
    # 先尝试从目录名提取（step2_batch_labeling.py 的格式：d04）
    parent_dir = path.parent.name
    dir_match = re.search(r"_d(\d{2})_", parent_dir)
    if dir_match:
        return int(dir_match.group(1))
    
    # 尝试从文件名提取（如 day4, day04）
    stem = path.stem
    day_match = re.search(r"day[_\-]?(\d+)", stem, re.IGNORECASE)
    if day_match:
        return int(day_match.group(1))
    
    # 尝试从完整路径提取
    full_path = str(path)
    full_match = re.search(r"_d(\d{2})_", full_path)
    if full_match:
        return int(full_match.group(1))
    
    return None


def extract_dataset_size_from_path(path: Path) -> Optional[int]:
    """
    从文件路径中提取数据集规模（如 200, 500, 1000）
    
    支持格式：
    - synthetic_features_200_labeled.csv -> 200
    - synthetic_features_500_day4_labeled.csv -> 500
    - llm_1000_day0 -> 1000
    
    Returns:
        数据集规模，如果无法提取则返回 None
    """
    stem = path.stem  # e.g. synthetic_features_200_labeled
    # 查找所有数字，通常第一个较大的数字是规模
    numbers = re.findall(r'\d+', stem)
    if numbers:
        # 通常规模是较大的数字（200, 500, 1000），而不是 day 的数字（0-11）
        for num_str in numbers:
            num = int(num_str)
            # 如果数字在合理范围内（100-10000），可能是规模
            if 100 <= num <= 10000:
                return num
        # 如果没有找到合理范围的数字，返回最大的数字
        return max(int(n) for n in numbers)
    return None


def infer_dataset_name(path: Path, prefix: str, days_to_cutoff: Optional[int] = None) -> str:
    """
    根据文件名推断数据集名称
    
    Args:
        path: 文件路径
        prefix: 数据集前缀
        days_to_cutoff: 可选的 day off 值（如果为 None，会尝试从路径提取）
    """
    stem = path.stem  # e.g. synthetic_features_1000_labeled
    # 提取第一个整数作为规模
    match = re.search(r"(\d+)", stem)
    suffix = match.group(1) if match else stem
    
    # 提取 day off 信息
    if days_to_cutoff is None:
        days_to_cutoff = extract_day_off_from_path(path)
    
    if days_to_cutoff is not None:
        suffix = f"{suffix}_day{days_to_cutoff}"
    
    return f"{prefix}_{suffix}"


def discover_labeled_files(input_dir: Path) -> List[Tuple[Path, Optional[int]]]:
    """
    从目录中自动发现所有带标签的 CSV 文件
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        List of (file_path, days_to_cutoff) tuples
    """
    labeled_files = []
    
    # 递归搜索所有 *_labeled.csv 文件
    for csv_file in input_dir.rglob("*_labeled.csv"):
        if csv_file.is_file():
            days_to_cutoff = extract_day_off_from_path(csv_file)
            labeled_files.append((csv_file, days_to_cutoff))
    
    # 按 day off 排序（None 排在最后）
    labeled_files.sort(key=lambda x: (x[1] is None, x[1] or 999))
    
    return labeled_files


def summarize_dataframe(df: pd.DataFrame, name: str) -> Dict:
    info = {
        "name": name,
        "num_rows": len(df),
        "num_features": len(df.columns) - (1 if "submitted" in df.columns else 0),
        "submitted_rate": float(df["submitted"].mean()) if "submitted" in df.columns else None,
    }
    return info


def main() -> int:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()
    configure_logging()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"step4_augmented_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    # 收集要处理的文件列表
    files_to_process: List[Tuple[Path, Optional[int]]] = []
    
    if args.input_dir:
        # 模式2：自动发现目录中的所有文件
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logging.error("❌ 输入目录不存在: %s", input_dir)
            return 1
        
        logging.info("=" * 80)
        logging.info("步骤4：创建增强训练集（Train-Only Augmentation）")
        logging.info("模式：自动发现文件（批量处理多个day off）")
        logging.info("=" * 80)
        logging.info("输入目录: %s", input_dir.resolve())
        
        files_to_process = discover_labeled_files(input_dir)
        logging.info("发现 %d 个带标签的 CSV 文件:", len(files_to_process))
        for file_path, day_off in files_to_process:
            day_str = f" (day_off={day_off})" if day_off is not None else " (day_off=未知)"
            logging.info("  - %s%s", file_path, day_str)
    else:
        # 模式1：使用指定的文件列表
        logging.info("=" * 80)
        logging.info("步骤4：创建增强训练集（Train-Only Augmentation）")
        logging.info("模式：指定文件列表")
        logging.info("=" * 80)
        
        for csv_path_str in args.synthetic_csv:
            csv_path = Path(csv_path_str)
            if not csv_path.exists():
                logging.warning("跳过不存在的文件: %s", csv_path)
                continue
            day_off = extract_day_off_from_path(csv_path)
            files_to_process.append((csv_path, day_off))
            day_str = f" (day_off={day_off})" if day_off is not None else ""
            logging.info("  - %s%s", csv_path, day_str)
    
    if not files_to_process:
        logging.error("❌ 没有找到要处理的文件")
        return 1
    
    logging.info("输出目录: %s", output_root.resolve())
    logging.info("=" * 80)

    # 按 day_off 分组处理，为每个 day_off 创建对应的 assigner
    day_off_groups: Dict[Optional[int], List[Path]] = {}
    for file_path, day_off in files_to_process:
        if day_off not in day_off_groups:
            day_off_groups[day_off] = []
        day_off_groups[day_off].append(file_path)

    merger = DatasetMerger(random_seed=args.seed)
    summary_records: List[Dict] = []

    # 为每个 day_off 组处理
    for day_off, file_list in sorted(day_off_groups.items(), key=lambda x: (x[0] is None, x[0] or 999)):
        # 使用该 day_off 创建 assigner（用于加载对应的 baseline 数据）
        # 如果无法从路径提取 day_off，使用默认值 0
        current_days_to_cutoff = day_off if day_off is not None else 0
        
        logging.info("\n%s", "=" * 80)
        if day_off is not None:
            logging.info("处理 day_off=%d 的数据 (%d 个文件)", day_off, len(file_list))
        else:
            logging.info("处理 day_off=未知的数据 (%d 个文件)", len(file_list))
        logging.info("%s", "=" * 80)
        
        assigner = MultiModuleLabelAssigner(
            modules=args.modules,
            presentation=args.presentation,
            assessment_name=args.assessment_name,
            days_to_cutoff=current_days_to_cutoff,
            random_seed=args.seed,
        )

        baseline_data = assigner.load_baseline_data()
        if baseline_data is None:
            logging.error("❌ 无法加载baseline数据 (day_off=%s)，跳过", day_off)
            continue

        real_train_df = attach_labels(baseline_data["X_train"], baseline_data["y_train"])
        real_test_df = attach_labels(baseline_data["X_test"], baseline_data["y_test"])

        # 为每个 day_off 创建输出目录（直接在 day_* 下，不创建 augmented_datasets 子目录）
        if day_off is not None:
            day_dir = output_root / f"day_{day_off}"
        else:
            day_dir = output_root / "day_unknown"
        
        day_dir.mkdir(parents=True, exist_ok=True)

        # 为每个 day_off 保存自己的测试集（与对应的 baseline 对齐）
        test_path = day_dir / "test_set.csv"
        logging.info(
            "真实训练集: %d 样本, 测试集: %d 样本",
            len(real_train_df),
            len(real_test_df),
        )
        real_test_df.to_csv(test_path, index=False)
        logging.info("✓ 测试集已保存 (day_off=%s): %s", day_off, test_path.resolve())

        # 按数据集规模分组合成数据文件（200, 500, 1000）
        synthetic_by_size: Dict[Optional[int], Path] = {}
        for synth_path in file_list:
            size = extract_dataset_size_from_path(synth_path)
            if size is not None:
                synthetic_by_size[size] = synth_path
                logging.info("  规模 %d: %s", size, synth_path.name)
            else:
                logging.warning("⚠ 无法从路径提取规模: %s", synth_path)
        
        if not synthetic_by_size:
            logging.warning("⚠ day_off=%s 没有有效的合成数据文件，跳过", day_off)
            continue
        
        logging.info("\n%s", "-" * 80)
        logging.info("创建增强数据集目录: %s", day_dir.resolve())
        logging.info("合成数据规模: %s", sorted(synthetic_by_size.keys()))
        
        # 创建统一的增强数据集
        # 使用对应规模的数据创建 plus_200, plus_500, plus_1000
        # baseline 使用真实训练集
        datasets = {}
        
        # 1. 创建 baseline（仅真实数据）
        baseline_path = day_dir / f"train_baseline.csv"
        real_train_df.to_csv(baseline_path, index=False)
        datasets['baseline'] = {
            'df': real_train_df.copy(),
            'path': baseline_path,
            'size': len(real_train_df)
        }
        logging.info("✓ Baseline: %d 样本 -> %s", len(real_train_df), baseline_path.name)
        
        # 辅助函数：对齐列
        def align_columns(real_df, synth_df):
            """对齐真实数据和合成数据的列"""
            real_clean = real_df.copy()
            synth_clean = synth_df.copy()
            
            # 移除不需要的列
            for col in ['id_student', 'student_type', 'submitted_proba']:
                if col in synth_clean.columns:
                    synth_clean = synth_clean.drop(columns=[col])
            
            # 对齐列
            real_cols = set(real_clean.columns)
            synth_cols = set(synth_clean.columns)
            
            # 添加缺失的列
            for col in real_cols - synth_cols:
                if col != 'submitted':
                    synth_clean[col] = 0
            
            for col in synth_cols - real_cols:
                if col != 'submitted':
                    real_clean[col] = 0
            
            # 确保列顺序一致
            common_cols = sorted(list(real_cols | synth_cols))
            if 'submitted' in common_cols:
                common_cols.remove('submitted')
                common_cols.append('submitted')
            
            real_aligned = real_clean[common_cols]
            synth_aligned = synth_clean[common_cols]
            
            return real_aligned, synth_aligned
        
        # 2. 创建 plus_200（使用 200 规模的数据）
        if 200 in synthetic_by_size:
            synth_200_df = pd.read_csv(synthetic_by_size[200])
            if "submitted" not in synth_200_df.columns:
                logging.warning("⚠ %s 缺少 submitted 列，跳过 plus_200", synthetic_by_size[200])
            else:
                real_aligned, synth_aligned = align_columns(real_train_df, synth_200_df)
                plus_200 = merger._merge_datasets(real_aligned, synth_aligned, n_synthetic=200)
                plus_200_path = day_dir / f"train_plus_200.csv"
                plus_200.to_csv(plus_200_path, index=False)
                datasets['plus_200'] = {
                    'df': plus_200,
                    'path': plus_200_path,
                    'size': len(plus_200)
                }
                logging.info("✓ Train+200: %d 样本 -> %s", len(plus_200), plus_200_path.name)
        
        # 3. 创建 plus_500（使用 500 规模的数据）
        if 500 in synthetic_by_size:
            synth_500_df = pd.read_csv(synthetic_by_size[500])
            if "submitted" not in synth_500_df.columns:
                logging.warning("⚠ %s 缺少 submitted 列，跳过 plus_500", synthetic_by_size[500])
            else:
                real_aligned, synth_aligned = align_columns(real_train_df, synth_500_df)
                plus_500 = merger._merge_datasets(real_aligned, synth_aligned, n_synthetic=500)
                plus_500_path = day_dir / f"train_plus_500.csv"
                plus_500.to_csv(plus_500_path, index=False)
                datasets['plus_500'] = {
                    'df': plus_500,
                    'path': plus_500_path,
                    'size': len(plus_500)
                }
                logging.info("✓ Train+500: %d 样本 -> %s", len(plus_500), plus_500_path.name)
        
        # 4. 创建 plus_1000（使用 1000 规模的数据）
        if 1000 in synthetic_by_size:
            synth_1000_df = pd.read_csv(synthetic_by_size[1000])
            if "submitted" not in synth_1000_df.columns:
                logging.warning("⚠ %s 缺少 submitted 列，跳过 plus_1000", synthetic_by_size[1000])
            else:
                real_aligned, synth_aligned = align_columns(real_train_df, synth_1000_df)
                plus_1000 = merger._merge_datasets(real_aligned, synth_aligned, n_synthetic=1000)
                plus_1000_path = day_dir / f"train_plus_1000.csv"
                plus_1000.to_csv(plus_1000_path, index=False)
                datasets['plus_1000'] = {
                    'df': plus_1000,
                    'path': plus_1000_path,
                    'size': len(plus_1000)
                }
                logging.info("✓ Train+1000: %d 样本 -> %s", len(plus_1000), plus_1000_path.name)
        
        # 记录摘要信息
        summary_entry = {
            "days_to_cutoff": day_off,
            "real_train": summarize_dataframe(real_train_df, "real_train"),
            "synthetic_sources": {size: str(path.resolve()) for size, path in synthetic_by_size.items()},
            "outputs": {},
        }
        
        for condition, info in datasets.items():
            summary_entry["outputs"][condition] = {
                "path": str(Path(info["path"]).resolve()),
                "train_size": info["size"],
            }
        
        summary_records.append(summary_entry)
        logging.info("✓ 完成 day_off=%s", day_off)

    summary_payload = {
        "timestamp": timestamp,
        "modules": args.modules,
        "presentation": args.presentation,
        "assessment_name": args.assessment_name,
        "seed": args.seed,
        "output_root": str(output_root.resolve()),
        "synthetic_inputs": summary_records,
        "note": "测试集保持不变，仅增强训练集。支持批量处理多个day off数据。",
    }

    summary_path = output_root / "augmentation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    logging.info("\n%s", "=" * 80)
    logging.info("✓ 增强完成！")
    logging.info("  处理文件数: %d", len(files_to_process))
    logging.info("  成功处理: %d", len(summary_records))
    logging.info("  详情见: %s", summary_path.resolve())
    logging.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


