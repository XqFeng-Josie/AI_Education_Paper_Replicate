"""
步骤2批处理脚本
----------------
一次性对多个 studentVle.csv 与多个 days_to_cutoff 组合运行 baseline 打标流程。

示例：
    python step2_batch_labeling.py \
        --vle_files results/vle_data/studentVle_200_ex.csv results/vle_data/studentVle_500_ex.csv \
        --days_to_cutoff 0 4 7 \
        --output_dir results/labeled_data
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from step2_baseline_labeling import run_labeling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行步骤2（Baseline训练 + 打标）脚本"
    )
    parser.add_argument(
        "--vle_files",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 studentVle.csv 文件路径",
    )
    parser.add_argument(
        "--days_to_cutoff",
        type=int,
        nargs="+",
        default=[0],
        help="需要遍历的 days_to_cutoff 组合（可多值）",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=["BBB", "DDD", "EEE", "FFF"],
        help="模块列表（默认：BBB DDD EEE FFF）",
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
        default="results/labeled_data",
        help="输出根目录（所有运行结果将落在该目录下）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（传递给 baseline 模型）",
    )
    parser.add_argument(
        "--timestamp_prefix",
        type=str,
        default=None,
        help="可选：自定义时间戳前缀，方便批量结果对齐",
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    configure_logging()

    timestamp_prefix = args.timestamp_prefix or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 100)
    logging.info("步骤2批处理：共 %d 个VLE文件 × %d 个days_to_cutoff", len(args.vle_files), len(args.days_to_cutoff))
    logging.info("输出根目录: %s", output_root.resolve())
    logging.info("=" * 100)

    summary: List[Dict] = []
    run_counter = 0

    for file_idx, vle_file in enumerate(args.vle_files, start=1):
        for days in args.days_to_cutoff:
            run_counter += 1
            run_tag = f"{timestamp_prefix}_f{file_idx:02d}_d{days:02d}_r{run_counter:03d}"
            logging.info("\n%s", "-" * 80)
            logging.info("运行 #%d | VLE: %s | days_to_cutoff=%d | tag=%s", run_counter, vle_file, days, run_tag)

            try:
                result = run_labeling(
                    vle_data=vle_file,
                    modules=args.modules,
                    presentation=args.presentation,
                    assessment_name=args.assessment_name,
                    days_to_cutoff=days,
                    output_dir=str(output_root),
                    seed=args.seed,
                    timestamp=run_tag,
                )
                summary.append(result)
                logging.info("✓ 完成 #%d，输出目录: %s", run_counter, result["output_dir"])
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("✗ 运行失败 #%d (%s, days=%d): %s", run_counter, vle_file, days, exc, exc_info=True)

    summary_path = output_root / f"batch_labeling_summary_{timestamp_prefix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("\n%s", "=" * 100)
    logging.info("批处理完成：成功 %d / %d", len(summary), run_counter)
    logging.info("摘要文件: %s", summary_path.resolve())
    logging.info("=" * 100)

    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())


