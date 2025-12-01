"""
步骤6：结果报告
--------------
读取步骤5输出的 training_results.json/csv，生成 metrics_sim.csv、summary_table.csv 及可视化图。

支持两种模式：
- 模式1：指定具体的 results_json 文件（--results_json）
- 模式2：自动发现第五步输出目录中的所有结果（--step5_output_dir）
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from evaluation.result_reporter import ResultReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="步骤6：生成评估报告（支持批量处理所有结果）")
    
    # 输入模式：二选一
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--step5_output_dir",
        type=str,
        help="步骤5的输出根目录（step5_results_*），自动发现 training_results.json",
    )
    input_group.add_argument(
        "--results_json",
        type=str,
        help="步骤5生成的 training_results.json（向后兼容）",
    )
    
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="*",
        default=None,
        help="可选：仅使用指定 dataset_name 的结果",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="可选：仅分析指定模型",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/model_reports",
        help="报告输出目录",
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_step5_results(step5_root: Path) -> Path:
    """
    从步骤5的输出根目录自动发现 training_results.json
    
    Args:
        step5_root: 步骤5的输出根目录（step5_results_*）
        
    Returns:
        training_results.json 的路径
    """
    results_json = step5_root / "training_results.json"
    if not results_json.exists():
        raise FileNotFoundError(f"未找到训练结果文件: {results_json}")
    return results_json


def filter_results(results: List[dict], dataset_names: Optional[List[str]], models: Optional[List[str]]):
    filtered = []
    for record in results:
        if dataset_names and record.get("dataset_name") not in dataset_names:
            continue
        if models and record.get("model") not in models:
            continue
        filtered.append(record)
    return filtered


def main() -> int:
    args = parse_args()
    configure_logging()

    # 根据输入模式确定结果文件
    if args.step5_output_dir:
        # 模式2：自动发现第五步输出目录中的结果
        step5_root = Path(args.step5_output_dir).resolve()
        if not step5_root.exists():
            raise FileNotFoundError(f"步骤5输出目录不存在: {step5_root}")
        
        logging.info("=" * 100)
        logging.info("步骤6：生成结果报告")
        logging.info("模式：自动发现结果")
        logging.info("步骤5输出目录: %s", step5_root)
        
        results_path = discover_step5_results(step5_root)
        logging.info("找到结果文件: %s", results_path)
    else:
        # 模式1：使用指定的结果文件（向后兼容）
        results_path = Path(args.results_json).resolve()
        if not results_path.exists():
            raise FileNotFoundError(f"results_json 不存在: {results_path}")
        
        logging.info("=" * 100)
        logging.info("步骤6：生成结果报告")
        logging.info("模式：指定文件")
        logging.info("输入: %s", results_path)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    filtered = filter_results(results, args.dataset_names, args.models)
    if not filtered:
        logging.error("没有匹配到任何结果，请检查筛选条件")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"step6_report_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("输出目录: %s", output_dir)
    logging.info("总记录数: %d", len(results))
    logging.info("过滤后记录数: %d", len(filtered))
    if args.dataset_names:
        logging.info("数据集筛选: %s", ", ".join(args.dataset_names))
    if args.models:
        logging.info("模型筛选: %s", ", ".join(args.models))
    logging.info("=" * 100)

    reporter = ResultReporter()
    reporter.generate_full_report(filtered, output_dir)

    snapshot = {
        "results_json": str(results_path),
        "dataset_names": args.dataset_names,
        "models": args.models,
        "timestamp": timestamp,
        "num_records": len(filtered),
    }
    with open(output_dir / "report_config.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    logging.info("✓ 报告生成完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


