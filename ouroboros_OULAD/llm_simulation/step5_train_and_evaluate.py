"""
步骤5：训练与评估
----------------
读取步骤4生成的 `augmented_datasets/` CSV + 固定测试集，训练多个模型并输出评估结果。

实验设计：
- day_off 目录下是"增加的数据"（子目录如 llm_200_dayX, llm_500_dayX 等）
- 如果某个 day_off 目录下没有子目录（没有增加的数据），只训练一个 baseline
- 如果某个 day_off 目录下有子目录（有增加的数据），每个子目录下的 augmented_datasets/ 
  包含多个条件（baseline + plus_*），实验数 = 子目录数量 × day_off 数量

主要特性：
- 默认只训练 baseline, +200, +500（排除 +1000）
- 支持结果缓存：每个模型训练后立即保存，后续相同配置自动跳过
- 按 cut day off 分组训练

支持两种模式：
- 模式1：指定具体目录和测试集（--augmented_root + --test_csv）
- 模式2：自动发现第四步输出目录中的所有数据（--step4_output_dir），自动处理所有day_off

缓存机制：
- 默认在输出目录下创建 cache/ 子目录存储缓存
- 可通过 --cache-dir 指定自定义缓存目录
- 使用 --skip-cache 可强制重新训练所有模型
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.model_trainer import ModelTrainer

CONDITION_SUFFIXES = {
    "baseline": "_train_baseline",
    "plus_200": "_train_plus_200",
    "plus_500": "_train_plus_500",
    "plus_1000": "_train_plus_1000",
}

# 默认训练的条件（排除 plus_1000，因为实验设计是每组3个数据：baseline, +200, +500）
DEFAULT_CONDITIONS = ["baseline", "plus_200", "plus_500"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="步骤5：训练并评估增强数据集（支持批量处理所有day_off）")
    
    # 输入模式：二选一
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--step4_output_dir",
        type=str,
        help="步骤4的输出根目录（step4_augmented_*），自动发现所有day_off的数据",
    )
    input_group.add_argument(
        "--augmented_root",
        type=str,
        help="步骤4输出的根目录，或包含 augmented_datasets 的某个子目录（向后兼容）",
    )
    
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="固定baseline测试集CSV（使用--step4_output_dir时自动发现，否则必须指定）",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="*",
        default=None,
        help="仅评估指定数据集（对应步骤4的dataset_name），默认全部",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="模型列表（默认：LR RF NB 以及安装了XGBoost则包含XGB）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/model_training",
        help="训练结果输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（传递给ModelTrainer）",
    )
    parser.add_argument(
        "--exclude-conditions",
        type=str,
        nargs="*",
        default=None,
        help="排除的训练条件（默认排除 plus_1000，只训练 baseline, plus_200, plus_500）",
    )
    parser.add_argument(
        "--include-conditions",
        type=str,
        nargs="*",
        default=None,
        help="指定要训练的条件（覆盖默认设置，例如：--include-conditions baseline plus_200 plus_500 plus_1000）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="结果缓存目录（如果指定，会检查缓存并跳过已训练的组合，训练后保存结果）",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="跳过缓存检查，强制重新训练所有模型",
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_step4_datasets(step4_root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    从步骤4的输出根目录自动发现所有数据集和测试集（按day分组）
    
    新的目录结构（平展化后）：
    - day_0/train_baseline.csv
    - day_0/train_plus_200.csv
    - day_0/train_plus_500.csv
    - day_0/train_plus_1000.csv
    - day_0/test_set.csv  (每个 day 有自己的测试集)
    
    Args:
        step4_root: 步骤4的输出根目录（step4_augmented_*）
        
    Returns:
        (day_groups, test_csvs)
        - day_groups: {day_name: day_dir_path} 字典
          例如: {"day_0": path_to_day_0, "day_4": path_to_day_4, ...}
        - test_csvs: {day_name: test_csv_path} 字典
          例如: {"day_0": path_to_day_0/test_set.csv, ...}
    """
    day_groups: Dict[str, Path] = {}
    test_csvs: Dict[str, Path] = {}
    
    # 查找所有 day_X/ 目录（直接在 day_* 下查找数据文件，不再需要 augmented_datasets 子目录）
    for day_dir in sorted(step4_root.glob("day_*")):
        if not day_dir.is_dir():
            continue
        
        day_name = day_dir.name  # 例如 "day_0"
        
        # 检查是否有数据文件（至少有一个 train_*.csv）
        has_data = any(day_dir.glob("train_*.csv"))
        if has_data:
            day_groups[day_name] = day_dir
            
            # 查找该 day 对应的测试集
            test_csv = day_dir / "test_set.csv"
            if test_csv.exists():
                test_csvs[day_name] = test_csv
            else:
                raise FileNotFoundError(f"未找到 {day_name} 的测试集: {test_csv}")
    
    # 如果没有找到 day_X/ 目录，尝试旧格式（向后兼容）
    if not day_groups:
        # 尝试旧格式1：day_X/augmented_datasets/
        for day_dir in sorted(step4_root.glob("day_*")):
            if not day_dir.is_dir():
                continue
            augmented_dir = day_dir / "augmented_datasets"
            if augmented_dir.is_dir():
                day_groups[day_dir.name] = augmented_dir
                # 旧格式：使用根目录下的统一测试集
                old_test_csv = step4_root / "baseline_test_set.csv"
                if old_test_csv.exists():
                    test_csvs[day_dir.name] = old_test_csv
        
        # 尝试旧格式2：直接在根目录下的 augmented_datasets
        if not day_groups:
            root_augmented = step4_root / "augmented_datasets"
            if root_augmented.is_dir():
                day_groups["day_unknown"] = root_augmented
                old_test_csv = step4_root / "baseline_test_set.csv"
                if old_test_csv.exists():
                    test_csvs["day_unknown"] = old_test_csv
            else:
                # 尝试旧格式3：每个子目录下有 augmented_datasets
                for child in sorted(step4_root.iterdir()):
                    if not child.is_dir() or child.name.startswith("day_"):
                        continue
                    candidate = child / "augmented_datasets"
                    if candidate.is_dir():
                        day_groups[child.name] = candidate
                        old_test_csv = step4_root / "baseline_test_set.csv"
                        if old_test_csv.exists():
                            test_csvs[child.name] = old_test_csv
    
    if not day_groups:
        raise RuntimeError(f"未在 {step4_root} 找到任何数据集")
    
    if not test_csvs:
        # 向后兼容：尝试使用根目录下的统一测试集
        old_test_csv = step4_root / "baseline_test_set.csv"
        if old_test_csv.exists():
            for day_name in day_groups.keys():
                test_csvs[day_name] = old_test_csv
        else:
            raise FileNotFoundError(f"未找到任何测试集")
    
    return day_groups, test_csvs


def list_dataset_dirs(root: Path) -> Dict[str, Path]:
    """
    根据步骤4的输出结构定位所有 dataset_name 的 augmented_datasets 目录（向后兼容）
    """
    dataset_dirs: Dict[str, Path] = {}
    if (root / "augmented_datasets").is_dir():
        dataset_name = root.name
        dataset_dirs[dataset_name] = root / "augmented_datasets"
        return dataset_dirs

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / "augmented_datasets"
        if candidate.is_dir():
            dataset_dirs[child.name] = candidate
    return dataset_dirs


def load_augmented_csvs(augmented_dir: Path, conditions_filter: Optional[List[str]] = None):
    """
    加载增强数据集CSV文件
    
    支持两种文件名格式：
    1. 新格式（平展化后）：train_baseline.csv, train_plus_200.csv, ...
    2. 旧格式（向后兼容）：*_train_baseline.csv, *_train_plus_200.csv, ...
    """
    datasets = {}
    csv_files = list(augmented_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{augmented_dir} 中未找到CSV文件")

    for condition, suffix in CONDITION_SUFFIXES.items():
        if conditions_filter and condition not in conditions_filter:
            continue
        
        # 尝试新格式：train_baseline.csv, train_plus_200.csv, ...
        new_format_name = suffix.replace("_train_", "train_")  # _train_baseline -> train_baseline
        new_matches = [path for path in csv_files if path.stem == new_format_name]
        
        # 如果新格式没找到，尝试旧格式：*_train_baseline.csv
        if not new_matches:
            old_matches = [path for path in csv_files if path.stem.endswith(suffix)]
            matches = old_matches
        else:
            matches = new_matches
        
        if not matches:
            continue
        
        csv_path = matches[0]
        df = pd.read_csv(csv_path)
        datasets[condition] = {"df": df, "path": str(csv_path.resolve()), "size": len(df)}

    if not datasets:
        raise RuntimeError(f"{augmented_dir} 中未匹配到任何条件: {conditions_filter}")
    return datasets


def get_cache_key(day_name: str, dataset_name: str, condition: str, model: str, seed: int) -> str:
    """生成缓存键（文件系统安全）"""
    # 替换可能存在的特殊字符，确保文件名安全
    safe_day = day_name.replace("/", "_").replace("\\", "_")
    safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
    return f"{safe_day}_{safe_dataset}_{condition}_{model}_seed{seed}"


def load_cached_result(cache_dir: Path, cache_key: str) -> Optional[Dict]:
    """从缓存加载结果"""
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning("加载缓存失败 %s: %s", cache_file, e)
    return None


def save_cached_result(cache_dir: Path, cache_key: str, result: Dict):
    """保存结果到缓存"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.warning("保存缓存失败 %s: %s", cache_file, e)


def main() -> int:
    args = parse_args()
    configure_logging()

    # 根据输入模式确定数据集和测试集
    if args.step4_output_dir:
        # 模式2：自动发现第四步输出目录中的所有数据（按day分组）
        step4_root = Path(args.step4_output_dir).resolve()
        if not step4_root.exists():
            raise FileNotFoundError(f"步骤4输出目录不存在: {step4_root}")
        
        logging.info("=" * 100)
        logging.info("步骤5：训练并评估增强数据集")
        logging.info("模式：自动发现所有day_off数据（按day分组训练）")
        logging.info("步骤4输出目录: %s", step4_root)
        
        day_groups, test_csvs = discover_step4_datasets(step4_root)
        augmented_root = step4_root  # 用于日志记录
        
        # 统计信息
        logging.info("自动发现 %d 个day组", len(day_groups))
        for day_name, augmented_dir in sorted(day_groups.items()):
            test_csv = test_csvs.get(day_name)
            logging.info("  %s: %s (测试集: %s)", day_name, augmented_dir, test_csv)
    else:
        # 模式1：使用指定的目录和测试集（向后兼容）
        if not args.test_csv:
            raise ValueError("使用 --augmented_root 时必须指定 --test_csv")
        
        augmented_root = Path(args.augmented_root).resolve()
        test_csv = Path(args.test_csv).resolve()
        if not test_csv.exists():
            raise FileNotFoundError(f"测试集不存在: {test_csv}")

        dataset_dirs = list_dataset_dirs(augmented_root)
        if not dataset_dirs:
            raise RuntimeError(f"未在 {augmented_root} 找到任何 augmented_datasets 目录")
        
        # 将旧格式转换为day分组格式
        # 旧格式：每个子目录有自己的 augmented_datasets
        # 新格式：每个 day 只有一个 augmented_datasets
        # 为了兼容，我们为每个 dataset 创建一个 day 组
        day_groups = {}
        test_csvs = {}
        for dataset_name, augmented_dir in dataset_dirs.items():
            day_name = f"day_{dataset_name}"  # 使用 dataset_name 作为 day 名
            day_groups[day_name] = augmented_dir
            test_csvs[day_name] = test_csv  # 所有 day 使用同一个测试集（向后兼容）
        
        logging.info("=" * 100)
        logging.info("步骤5：训练并评估增强数据集")
        logging.info("模式：指定目录（向后兼容）")
        logging.info("Augmented root: %s", augmented_root)
        logging.info("Test CSV: %s", test_csv)

    # 过滤数据集（如果指定了dataset_names）- 在新格式中不再需要，因为每个 day 只有一个 augmented_datasets
    # 保留此代码以向后兼容旧格式
    if args.dataset_names:
        logging.warning("⚠ --dataset_names 在新格式中已不再使用（每个 day 只有一个 augmented_datasets）")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"step5_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置缓存目录
    cache_dir = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info("缓存目录: %s", cache_dir)
    elif not args.skip_cache:
        # 如果没有指定缓存目录，使用输出目录下的 cache 子目录
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info("使用默认缓存目录: %s", cache_dir)

    logging.info("输出目录: %s", output_dir)
    logging.info("=" * 100)

    # 确定要训练的条件
    if args.include_conditions:
        # 如果指定了 include-conditions，使用指定的条件
        conditions_filter = list(args.include_conditions)
        logging.info("使用指定的条件: %s", ", ".join(conditions_filter))
    elif args.exclude_conditions:
        # 如果指定了 exclude-conditions，从所有条件中排除
        all_conditions = set(CONDITION_SUFFIXES.keys())
        excluded = set(args.exclude_conditions)
        conditions_filter = list(all_conditions - excluded)
        logging.info("排除条件: %s", ", ".join(sorted(excluded)))
        logging.info("将训练条件: %s", ", ".join(sorted(conditions_filter)) if conditions_filter else "无")
    else:
        # 默认：只训练 baseline, plus_200, plus_500（排除 plus_1000）
        conditions_filter = DEFAULT_CONDITIONS.copy()
        logging.info("使用默认条件（排除 plus_1000）: %s", ", ".join(conditions_filter))

    trainer = ModelTrainer(random_seed=args.seed)

    all_results: List[Dict] = []
    cached_count = 0
    trained_count = 0

    # 按day分组训练
    for day_name, day_dir in sorted(day_groups.items()):
        # 获取该 day 对应的测试集
        test_csv = test_csvs.get(day_name)
        if not test_csv or not test_csv.exists():
            logging.error("❌ %s 的测试集不存在: %s", day_name, test_csv)
            continue
        
        test_df = pd.read_csv(test_csv)
        
        logging.info("\n%s", "=" * 100)
        logging.info("处理 %s", day_name)
        logging.info("数据目录: %s", day_dir)
        logging.info("测试集: %s (%d 样本)", test_csv, len(test_df))
        logging.info("%s", "=" * 100)
        
        # 加载该 day_off 的所有条件
        datasets = load_augmented_csvs(day_dir, conditions_filter=conditions_filter)
        logging.info("发现条件: %s", ", ".join(datasets.keys()))

        # 确定要训练的模型
        if args.models is None:
            models = ['LR', 'RF', 'NB']
            try:
                import xgboost as xgb
                models.append('XGB')
            except ImportError:
                pass
        else:
            models = args.models

        # 逐个条件、逐个模型训练（支持缓存）
        for condition, data_info in datasets.items():
            train_df = data_info['df']
            train_size = len(train_df)
            
            logging.info("\n  条件: %s (训练集大小: %d)", condition, train_size)
            
            for model_name in models:
                cache_key = get_cache_key(day_name, "", condition, model_name, args.seed)
                
                # 检查缓存
                cached_result = None
                if cache_dir and not args.skip_cache:
                    cached_result = load_cached_result(cache_dir, cache_key)
                
                if cached_result:
                    # 确保缓存结果包含所有必要字段（向后兼容）
                    if 'day_name' not in cached_result:
                        cached_result["day_name"] = day_name
                    if 'augmented_dir' not in cached_result:
                        cached_result["augmented_dir"] = str(day_dir.resolve())
                    logging.info("    ✓ %s: 使用缓存结果 (PR-AUC=%.4f)", model_name, cached_result.get('pr_auc', 0))
                    all_results.append(cached_result)
                    cached_count += 1
                else:
                    # 训练模型
                    try:
                        logging.info("    → 训练 %s...", model_name)
                        
                        # 重置特征信息（每个数据集独立）
                        trainer._feature_columns = None
                        trainer._categorical_encoders = {}
                        trainer._list_col_expansions = {}
                        
                        # 准备数据
                        X_train, y_train = trainer._prepare_features(train_df, fit_mode=True)
                        X_test, y_test = trainer._prepare_features(test_df, fit_mode=False)
                        
                        # 创建并训练模型
                        model_list = trainer._create_models()
                        model_dict = {name: model for model, name in model_list}
                        
                        if model_name not in model_dict:
                            logging.warning("    跳过 %s (不可用)", model_name)
                            continue
                        
                        model = model_dict[model_name]
                        model.fit(X_train, y_train)
                        
                        # 评估
                        metrics = trainer._evaluate_model(model, X_test, y_test)
                        logging.info("    ✓ %s: PR-AUC=%.4f, F1=%.4f", model_name, metrics['pr_auc'], metrics['f1'])
                        
                        # 构建结果记录（包含所有必要字段，以便缓存）
                        result = {
                            'condition': condition,
                            'model': model_name,
                            'seed': args.seed,
                            'pr_auc': float(metrics['pr_auc']),
                            'f1': float(metrics['f1']),
                            'precision': float(metrics['precision']),
                            'recall': float(metrics['recall']),
                            'train_size': int(train_size),
                            'test_size': int(len(test_df)),
                            'day_name': day_name,
                            'augmented_dir': str(day_dir.resolve()),
                        }
                        
                        if not pd.isna(metrics.get('roc_auc', np.nan)):
                            result['roc_auc'] = float(metrics['roc_auc'])
                        
                        # 保存到缓存（在添加到 all_results 之前）
                        if cache_dir:
                            save_cached_result(cache_dir, cache_key, result)
                        
                        all_results.append(result)
                        trained_count += 1
                        
                    except Exception as e:
                        logging.error("    ✗ %s 训练失败: %s", model_name, e, exc_info=True)
                        continue

    if not all_results:
        logging.warning("未生成任何结果")
        return 1

    results_json = output_dir / "training_results.json"
    results_csv = output_dir / "training_results.csv"

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    pd.DataFrame(all_results).to_csv(results_csv, index=False)

    config_snapshot = {
        "augmented_root": str(augmented_root),
        "test_csv": str(test_csv),
        "dataset_names": args.dataset_names,
        "models": args.models,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    logging.info("\n%s", "=" * 100)
    logging.info("训练完成！")
    logging.info("  总结果数: %d", len(all_results))
    logging.info("  从缓存加载: %d", cached_count)
    logging.info("  新训练: %d", trained_count)
    logging.info("  结果文件: %s", results_json)
    if cache_dir:
        logging.info("  缓存目录: %s", cache_dir)
    logging.info("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


