"""
步骤2：Baseline训练并打标

1. 加载真实OULAD数据，训练baseline模型
2. 从VLE数据提取特征
3. 使用baseline模型为合成数据分配标签

Usage:
    python step2_baseline_labeling.py --vle_data results/vle_data/vle_data_200_*/studentVle_200.csv
    python step2_baseline_labeling.py --vle_data studentVle_500.csv --output_dir results/labeled_data
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup environment
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from augmentation.multi_module_label_assignment import MultiModuleLabelAssigner
from selflearner.data_load.features_utils import FeaturesMapping
from selflearner.data_load.hdf5.pytables_hdf5_manager import PytablesHdf5Manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('step2_baseline_labeling.log')
    ]
)
logger = logging.getLogger(__name__)

HDF5_PATH_DEFAULT = PROJECT_ROOT / 'selflearner' / 'data_load' / 'data' / 'oulad.h5'
MAX_HISTORY_DAYS = 50
VLE_STATS_COLUMNS = [
    'count_days_fromvleopen', 'sum_material_fromvleopen', 'sum_click_fromvleopen',
    'count_days_fromvleopen_rel', 'first_login', 'last_login', 'last_login_rel',
    'never_logged', 'count_days_fromstart', 'sum_material_fromstart',
    'max_material_fromstart', 'min_material_count_fromstart_peractive',
    'median_material_count_fromstart_peractive', 'avg_material_count_fromstart_peractive',
    'sum_click_fromstart', 'max_clicks_fromstart', 'min_clicks_fromstart_peractive',
    'median_clicks_fromstart_peractive', 'avg_clicks_fromstart_peractive',
    'count_days_fromstart_rel', 'min_material_count_fromstart', 'median_material_count_fromstart',
    'avg_material_count_fromstart', 'min_clicks_fromstart', 'median_clicks_fromstart',
    'avg_clicks_fromstart', 'consecutive_days'
]
VLE_STATS_BEFORE_COLUMNS = [
    'count_days_beforestart', 'sum_material_beforestart', 'max_material_beforestart',
    'min_material_count_beforestart_peractive', 'median_material_count_beforestart_peractive',
    'avg_material_count_beforestart_peractive', 'sum_click_beforestart', 'max_clicks_beforestart',
    'min_clicks_beforestart_peractive', 'median_clicks_beforestart_peractive',
    'avg_clicks_beforestart_peractive', 'count_days_beforestart_rel', 'min_material_count_beforestart',
    'median_material_count_beforestart', 'avg_material_count_beforestart', 'min_clicks_beforestart',
    'median_clicks_beforestart', 'avg_clicks_beforestart'
]


class SyntheticFeatureExtractor:
    def __init__(
        self,
        modules: List[str],
        presentation: str,
        assessment_name: str,
        days_to_cutoff: int = 0,
        hdf5_path: Path = HDF5_PATH_DEFAULT,
        max_history_days: int = MAX_HISTORY_DAYS,
    ) -> None:
        self.modules = modules
        self.presentation = presentation
        self.assessment_name = assessment_name
        self.days_to_cutoff = days_to_cutoff
        self.max_history_days = max_history_days
        self.hdf5_path = Path(hdf5_path)
        self.manager = PytablesHdf5Manager(str(self.hdf5_path))
        self._assessments = self.manager.load_dataframe('assessments')
        self._vle_meta_cache: Dict[str, pd.DataFrame] = {}
        self._features_mapping = FeaturesMapping()

    def build_features(self, vle_df: pd.DataFrame) -> pd.DataFrame:
        module_features = []
        for module in self.modules:
            module_df = vle_df[vle_df['code_module'] == module].copy()
            if module_df.empty:
                logger.warning(f"模块 {module} 在提供的VLE数据中无记录，跳过")
                continue
            try:
                module_features.append(self._build_module_features(module_df, module))
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"模块 {module} 特征构建失败: {exc}", exc_info=True)
                continue

        if not module_features:
            raise ValueError("无法为任何模块生成特征，请检查输入数据")

        return pd.concat(module_features, axis=0, ignore_index=True)

    def _build_module_features(self, module_df: pd.DataFrame, module: str) -> pd.DataFrame:
        module_df = module_df.copy()
        if 'code_presentation' in module_df.columns:
            module_df = module_df[module_df['code_presentation'] == self.presentation]

        module_df['date'] = module_df['date'].astype(int)
        module_df['sum_click'] = module_df['sum_click'].fillna(0).astype(float)
        if 'id_site' not in module_df.columns:
            module_df['id_site'] = -1
        else:
            module_df['id_site'] = module_df['id_site'].fillna(-1).astype(int)

        cutoff_date = self._get_cutoff_date(module)
        features_to_date = cutoff_date - self.days_to_cutoff - 1
        features_from_date = -50

        df_filtered = module_df[
            (module_df['date'] >= features_from_date)
            & (module_df['date'] <= features_to_date)
        ].copy()
        df_filtered['date_back'] = features_to_date - df_filtered['date'] + 1
        df_filtered = df_filtered[df_filtered['date_back'] <= self.max_history_days]

        students = module_df[['id_student']].drop_duplicates().reset_index(drop=True)

        statistics = self._compute_vle_statistics(df_filtered, students)
        statistics_before = self._compute_vle_statistics_before_start(df_filtered, students)
        day_sums = self._compute_day_sums(df_filtered, students)
        day_flags = self._compute_day_flags(day_sums)
        activity_sums = self._compute_day_activity_type(df_filtered, students, module)
        activity_flags = self._compute_day_activity_type_flags(activity_sums)

        features = students.copy()
        for part in [statistics, statistics_before, day_sums, day_flags, activity_sums, activity_flags]:
            features = features.merge(part, on='id_student', how='left')

        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        features[numeric_cols] = features[numeric_cols].fillna(0)

        features['code_module'] = module
        features['code_presentation'] = self.presentation

        col_order = ['code_module', 'code_presentation', 'id_student'] + [
            col for col in features.columns if col not in {'code_module', 'code_presentation', 'id_student'}
        ]
        return features[col_order]

    def _get_cutoff_date(self, module: str) -> int:
        module_assessments = self._assessments.loc[module, self.presentation]
        if not isinstance(module_assessments, pd.DataFrame):
            module_assessments = module_assessments.to_frame().T
        target_row = module_assessments.loc[module_assessments['assessment_name'] == self.assessment_name]
        if target_row.empty:
            raise ValueError(f"未在模块 {module} 中找到评估 {self.assessment_name}")
        return int(target_row.iloc[0]['date'])

    def _compute_day_sums(self, df_filtered: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
        if df_filtered.empty:
            return students.copy()
        day_sums = (
            df_filtered.groupby(['id_student', 'date_back'])['sum_click']
            .sum()
            .unstack(fill_value=0)
        )
        day_sums = day_sums.rename(columns=lambda col: f"sum_click_{int(col)}")
        day_sums.reset_index(inplace=True)
        return pd.merge(students, day_sums, on='id_student', how='left')

    def _compute_day_flags(self, day_sums: pd.DataFrame) -> pd.DataFrame:
        flags = day_sums.copy()
        flag_cols = [col for col in flags.columns if col.startswith('sum_click_')]
        rename_map = {col: col.replace('sum_click', 'is_click') for col in flag_cols}
        for col in flag_cols:
            flags.rename(columns={col: rename_map[col]}, inplace=True)
            flags[rename_map[col]] = (flags[rename_map[col]] > 0).astype(int)
        return flags

    def _compute_day_activity_type(
        self,
        df_filtered: pd.DataFrame,
        students: pd.DataFrame,
        module: str
    ) -> pd.DataFrame:
        if df_filtered.empty:
            return students.copy()

        vle_meta = self._get_vle_meta(module)
        df_activity = df_filtered.merge(vle_meta, on='id_site', how='left')
        df_activity['activity_type'] = df_activity['activity_type'].fillna('unknown').str.lower()

        activity_pivot = (
            df_activity.pivot_table(
                index='id_student',
                columns=['date_back', 'activity_type'],
                values='sum_click',
                aggfunc='sum',
                fill_value=0,
            )
        )
        activity_pivot.columns = [
            f"sum_click_{int(date_back)}_{activity.replace(' ', '_')}"
            for date_back, activity in activity_pivot.columns
        ]
        activity_pivot.reset_index(inplace=True)
        return pd.merge(students, activity_pivot, on='id_student', how='left')

    def _compute_day_activity_type_flags(self, activity_sums: pd.DataFrame) -> pd.DataFrame:
        flags = activity_sums.copy()
        sum_cols = [
            col for col in flags.columns
            if col.startswith('sum_click_') and '_' in col[len('sum_click_'):]
        ]
        rename_map = {col: col.replace('sum_click', 'is_click') for col in sum_cols}
        for col in sum_cols:
            new_col = rename_map[col]
            flags.rename(columns={col: new_col}, inplace=True)
            flags[new_col] = (flags[new_col] > 0).astype(int)
        return flags

    def _compute_vle_statistics(self, df_filtered: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
        if df_filtered.empty:
            zeros = pd.DataFrame(0, index=np.arange(len(students)), columns=VLE_STATS_COLUMNS)
            zeros.insert(0, 'id_student', students['id_student'].values)
            return zeros

        max_click_day = df_filtered['date'].max()
        min_click_day = df_filtered['date'].min()
        num_days_from_vleopen = max_click_day - min_click_day + 1 if max_click_day >= min_click_day else 1
        num_days_from_start = max_click_day - 0 + 1 if max_click_day >= 0 else 1

        df_last_login = (
            df_filtered.groupby('id_student')
            .agg(first_login=('date', 'min'), last_login=('date', 'max'), last_login_rel=('date_back', 'min'))
            .reset_index()
        )
        df_last_login = pd.merge(students, df_last_login, on='id_student', how='left')
        df_last_login['never_logged'] = df_last_login['first_login'].isnull().astype(float)

        df_date_sums = (
            df_filtered.groupby(['id_student', 'date'])['sum_click']
            .agg(count_materials='count', sum_click='sum')
            .reset_index()
        )

        df_from_vleopen = df_date_sums.groupby('id_student').agg(
            count_days_fromvleopen=('count_materials', 'count'),
            sum_material_fromvleopen=('count_materials', 'sum'),
            sum_click_fromvleopen=('sum_click', 'sum'),
        ).reset_index()
        df_from_vleopen['count_days_fromvleopen_rel'] = df_from_vleopen['count_days_fromvleopen'] / max(
            num_days_from_vleopen, 1
        )
        df_from_vleopen = pd.merge(students, df_from_vleopen, on='id_student', how='left')

        ret_val = pd.merge(df_from_vleopen, df_last_login, on='id_student', how='left')

        df_after_start = df_filtered[df_filtered['date'] >= 0]
        if not df_after_start.empty:
            df_after_sum = (
                df_after_start.groupby(['id_student', 'date'])['sum_click']
                .agg(count_materials='count', sum_click='sum')
                .reset_index()
            )
            df_active = df_after_sum.groupby('id_student').agg(
                count_days_fromstart=('count_materials', 'count'),
                sum_material_fromstart=('count_materials', 'sum'),
                max_material_fromstart=('count_materials', 'max'),
                min_material_count_fromstart_peractive=('count_materials', 'min'),
                median_material_count_fromstart_peractive=('count_materials', 'median'),
                avg_material_count_fromstart_peractive=('count_materials', 'mean'),
                sum_click_fromstart=('sum_click', 'sum'),
                max_clicks_fromstart=('sum_click', 'max'),
                min_clicks_fromstart_peractive=('sum_click', 'min'),
                median_clicks_fromstart_peractive=('sum_click', 'median'),
                avg_clicks_fromstart_peractive=('sum_click', 'mean'),
            ).reset_index()
            df_active['count_days_fromstart_rel'] = df_active['count_days_fromstart'] / max(num_days_from_start, 1)

            df_all_days = df_after_sum.set_index(['id_student', 'date'])
            df_all_days = df_all_days.unstack().fillna(0).stack().reset_index()
            df_all_stats = df_all_days.groupby('id_student').agg(
                min_material_count_fromstart=('count_materials', 'min'),
                median_material_count_fromstart=('count_materials', 'median'),
                avg_material_count_fromstart=('count_materials', 'mean'),
                min_clicks_fromstart=('sum_click', 'min'),
                median_clicks_fromstart=('sum_click', 'median'),
                avg_clicks_fromstart=('sum_click', 'mean'),
            ).reset_index()

            ret_val = ret_val.merge(df_active, on='id_student', how='left')
            ret_val = ret_val.merge(df_all_stats, on='id_student', how='left')

        df_consecutive = (
            df_filtered.groupby(['id_student', 'date_back'])['sum_click']
            .count()
            .reset_index()[['id_student', 'date_back']]
        )
        df_consecutive = df_consecutive.groupby('id_student').agg(
            consecutive_days=('date_back', self._features_mapping.consecut)
        ).reset_index()

        ret_val = ret_val.merge(df_consecutive, on='id_student', how='left')
        ret_val = ret_val.fillna(0)
        return ret_val

    def _compute_vle_statistics_before_start(self, df_filtered: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
        df_before_start = df_filtered[df_filtered['date'] < 0].copy()
        if df_before_start.empty:
            zeros = pd.DataFrame(0, index=np.arange(len(students)), columns=VLE_STATS_BEFORE_COLUMNS)
            zeros.insert(0, 'id_student', students['id_student'].values)
            return zeros

        min_click_day = df_before_start['date'].min()
        num_days_before_start = abs(min_click_day - 0)
        num_days_before_start = max(num_days_before_start, 1)

        df_day = (
            df_before_start.groupby(['id_student', 'date'])['sum_click']
            .agg(count_materials='count', sum_click='sum')
            .reset_index()
        )

        df_active = df_day.groupby('id_student').agg(
            count_days_beforestart=('count_materials', 'count'),
            sum_material_beforestart=('count_materials', 'sum'),
            max_material_beforestart=('count_materials', 'max'),
            min_material_count_beforestart_peractive=('count_materials', 'min'),
            median_material_count_beforestart_peractive=('count_materials', 'median'),
            avg_material_count_beforestart_peractive=('count_materials', 'mean'),
            sum_click_beforestart=('sum_click', 'sum'),
            max_clicks_beforestart=('sum_click', 'max'),
            min_clicks_beforestart_peractive=('sum_click', 'min'),
            median_clicks_beforestart_peractive=('sum_click', 'median'),
            avg_clicks_beforestart_peractive=('sum_click', 'mean'),
        ).reset_index()
        df_active['count_days_beforestart_rel'] = df_active['count_days_beforestart'] / num_days_before_start

        df_all_days = df_day.set_index(['id_student', 'date'])
        df_all_days = df_all_days.unstack().fillna(0).stack().reset_index()
        df_all_stats = df_all_days.groupby('id_student').agg(
            min_material_count_beforestart=('count_materials', 'min'),
            median_material_count_beforestart=('count_materials', 'median'),
            avg_material_count_beforestart=('count_materials', 'mean'),
            min_clicks_beforestart=('sum_click', 'min'),
            median_clicks_beforestart=('sum_click', 'median'),
            avg_clicks_beforestart=('sum_click', 'mean'),
        ).reset_index()

        ret_val = pd.merge(students, df_active, on='id_student', how='left')
        ret_val = pd.merge(ret_val, df_all_stats, on='id_student', how='left')
        ret_val = ret_val.fillna(0)
        return ret_val

    def _get_vle_meta(self, module: str) -> pd.DataFrame:
        if module in self._vle_meta_cache:
            return self._vle_meta_cache[module]
        df_vle = self.manager.load_dataframe('vle').loc[module, self.presentation]
        if not isinstance(df_vle, pd.DataFrame):
            df_vle = df_vle.to_frame().T
        df_vle = df_vle.reset_index()[['id_site', 'activity_type']]
        self._vle_meta_cache[module] = df_vle
        return df_vle


def vle_logs_to_features(
    vle_logs_path: str,
    modules: List[str],
    presentation: str,
    assessment_name: str,
    days_to_cutoff: int
) -> pd.DataFrame:
    """
    从VLE logs转换为完整OULAD特征
    
    Args:
        vle_logs_path: VLE logs CSV文件路径（studentVle.csv格式）
        modules: 模块列表
        presentation: 呈现批次
        assessment_name: 评估名称
        days_to_cutoff: 距离截止日天数
        
    Returns:
        DataFrame: 特征DataFrame
    """
    logger.info(f"\n{'='*80}")
    logger.info("从VLE Logs转换为特征")
    logger.info(f"{'='*80}")

    vle_df = pd.read_csv(vle_logs_path)
    logger.info(f"加载VLE logs: {len(vle_df)} 条记录")

    required_columns = ['code_module', 'id_student', 'date', 'sum_click']
    missing_columns = [col for col in required_columns if col not in vle_df.columns]
    if missing_columns:
        raise ValueError(f"VLE数据缺少必需列: {missing_columns}")

    extractor = SyntheticFeatureExtractor(
        modules=modules,
        presentation=presentation,
        assessment_name=assessment_name,
        days_to_cutoff=days_to_cutoff,
    )
    features_df = extractor.build_features(vle_df)

    logger.info(f"✓ 提取了 {len(features_df)} 个学生的特征")
    logger.info(f"✓ 特征维度: {len(features_df.columns)}")

    logger.info(f"\n  各模块学生数:")
    for module in modules:
        count = len(features_df[features_df['code_module'] == module])
        logger.info(f"    {module}: {count} 学生")

    return features_df


def infer_student_count(vle_path: Path) -> str:
    """根据文件名推断学生数量（若失败则返回unknown）"""
    match = re.search(r"(\d+)", vle_path.stem)
    if match:
        return match.group(1)
    return vle_path.stem.split('_')[-1] if '_' in vle_path.stem else 'unknown'


def run_labeling(
    vle_data: str,
    modules: List[str],
    presentation: str,
    assessment_name: str,
    days_to_cutoff: int,
    output_dir: str,
    seed: int = 42,
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    """
    执行步骤2：从VLE数据生成特征并打标签

    Returns:
        包含输出路径和关键统计信息的字典
    """
    vle_path = Path(vle_data)
    if not vle_path.exists():
        raise FileNotFoundError(f"VLE数据不存在: {vle_data}")

    n_students = infer_student_count(vle_path)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir)
    output_dir_path = output_root / f"labeled_{n_students}_{timestamp}"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("步骤2：Baseline训练并打标")
    logger.info("=" * 80)
    logger.info("VLE数据: %s", vle_path)
    logger.info("模块: %s", ", ".join(modules))
    logger.info("天数距离截止日: %d", days_to_cutoff)
    logger.info("输出目录: %s", output_dir_path)
    logger.info("=" * 80)

    assigner = MultiModuleLabelAssigner(
        modules=modules,
        presentation=presentation,
        assessment_name=assessment_name,
        days_to_cutoff=days_to_cutoff,
        random_seed=seed,
    )

    logger.info("\n%s", "=" * 80)
    logger.info("加载Baseline数据并训练模型 (SVM-W-R)")
    logger.info("%s", "=" * 80)
    baseline_data = assigner.load_baseline_data()
    if baseline_data is None:
        raise RuntimeError("无法加载baseline数据")

    assigner.train_baseline_model(baseline_data["X_train"], baseline_data["y_train"])

    logger.info("\n%s", "=" * 80)
    logger.info("从合成VLE数据生成实验特征")
    logger.info("%s", "=" * 80)
    features_df = vle_logs_to_features(
        vle_logs_path=str(vle_path),
        modules=modules,
        presentation=presentation,
        assessment_name=assessment_name,
        days_to_cutoff=days_to_cutoff,
    )
    feature_filename = f"synthetic_features_{n_students}.csv"
    features_path = output_dir_path / feature_filename
    features_df.to_csv(features_path, index=False)
    logger.info("✓ 保存特征到: %s", features_path)

    logger.info("\n%s", "=" * 80)
    logger.info("为合成数据分配标签")
    logger.info("%s", "=" * 80)
    labeled_filename = f"synthetic_features_{n_students}_labeled.csv"
    labeled_path = output_dir_path / labeled_filename
    features_labeled = assigner.assign_labels(features_df, save_path=labeled_path)

    assigner.quality_check(features_labeled, baseline_data["y_train"])

    baseline_info = {
        "modules": modules,
        "presentation": presentation,
        "assessment_name": assessment_name,
        "days_to_cutoff": days_to_cutoff,
        "n_synthetic_students": len(features_labeled),
        "n_real_train_students": len(baseline_data["X_train"]),
        "n_real_test_students": len(baseline_data["X_test"]),
        "real_train_submitted_rate": float(baseline_data["y_train"]["submitted"].mean()),
        "synthetic_submitted_rate": float(features_labeled["submitted"].mean()),
        "timestamp": timestamp,
        "vle_data": str(vle_path.resolve()),
    }
    info_path = output_dir_path / "baseline_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(baseline_info, f, indent=2, ensure_ascii=False)

    logger.info("\n✅ 步骤2完成！")
    logger.info("输出文件:")
    logger.info("  - 特征: %s", features_path)
    logger.info("  - 带标签特征: %s", labeled_path)
    logger.info("  - Baseline信息: %s", info_path)

    return {
        "vle_data": str(vle_path.resolve()),
        "output_dir": str(output_dir_path.resolve()),
        "features_path": str(features_path.resolve()),
        "labeled_path": str(labeled_path.resolve()),
        "baseline_info_path": str(info_path.resolve()),
        "n_students": n_students,
        "days_to_cutoff": days_to_cutoff,
        "timestamp": timestamp,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="步骤2：Baseline训练并打标")
    parser.add_argument(
        "--vle_data",
        type=str,
        required=True,
        help="VLE数据CSV文件路径（studentVle.csv格式）",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=["BBB", "DDD", "EEE", "FFF"],
        help="模块列表",
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
        "--days_to_cutoff",
        type=int,
        default=0,
        help="距离截止日天数（默认：0）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/labeled_data",
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run_labeling(
            vle_data=args.vle_data,
            modules=args.modules,
            presentation=args.presentation,
            assessment_name=args.assessment_name,
            days_to_cutoff=args.days_to_cutoff,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("❌ 处理失败: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

