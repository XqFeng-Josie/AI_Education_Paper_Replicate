"""
Calculate metrics from saved experiment results

This script reads saved experiment results and calculates evaluation metrics.
Supports both incremental JSONL files and full JSON result files.

Usage:
  # Calculate metrics for all incremental files in a directory
  python -m llm.calculate_metrics --results_dir ./llm_experiments/results/paper_replication
  
  # Calculate metrics for specific configuration
  python -m llm.calculate_metrics --incremental_file ./llm_experiments/results/paper_replication/incremental_BBB_2014J_day0.jsonl
  
  # Calculate metrics from full results JSON
  python -m llm.calculate_metrics --full_results ./llm_experiments/results/paper_replication/paper_replication_full.json
  
  # Filter by module/presentation/day
  python -m llm.calculate_metrics --results_dir ./results --module BBB --day 0
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from llm.metrics import evaluate_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_incremental_results(filepath: Path) -> List[Dict[str, Any]]:
    """Load results from incremental JSONL file"""
    results = []
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return results
    
    logger.info(f"Loading results from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line.strip())
                results.append(result)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(results)} results")
    return results


def load_full_results(filepath: Path) -> List[Dict[str, Any]]:
    """Load results from full JSON file"""
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return []
    
    logger.info(f"Loading results from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all predictions from all configurations
    all_results = []
    if isinstance(data, list):
        for config_result in data:
            if 'predictions' in config_result:
                all_results.extend(config_result['predictions'])
    
    logger.info(f"Loaded {len(all_results)} total predictions")
    return all_results


def extract_config_from_filename(filename: str) -> Dict[str, Any]:
    """Extract configuration info from incremental filename
    
    Format: incremental_{module}_{presentation}_day{day}.jsonl
    Example: incremental_BBB_2014J_day0.jsonl
    """
    if not filename.startswith('incremental_'):
        return None
    
    # Remove 'incremental_' prefix and '.jsonl' suffix
    parts = filename[12:-6].split('_')
    
    if len(parts) < 3:
        return None
    
    # Last part should be 'dayX'
    day_str = parts[-1]
    if not day_str.startswith('day'):
        return None
    
    try:
        day = int(day_str[3:])
    except ValueError:
        return None
    
    # Everything before the day is module and presentation
    module = parts[0]
    presentation = '_'.join(parts[1:-1])
    
    return {
        'module': module,
        'presentation': presentation,
        'day': day
    }


def calculate_metrics_for_results(results: List[Dict[str, Any]], 
                                  config_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Calculate metrics for a set of results"""
    
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r or 'final_decision' in r]
    
    if len(valid_results) == 0:
        logger.warning("No valid results to evaluate")
        return None
    
    logger.info(f"Calculating metrics for {len(valid_results)} predictions")
    
    # Calculate metrics
    metrics = evaluate_predictions(valid_results)
    
    if not metrics:
        return None
    
    # Add configuration info
    result = {
        'n_students': len(valid_results),
        'n_errors': len(results) - len(valid_results),
        'metrics': metrics
    }
    
    if config_info:
        result.update(config_info)
    
    return result


def find_incremental_files(results_dir: Path, 
                           module: str = None,
                           presentation: str = None,
                           day: int = None) -> List[Path]:
    """Find all incremental result files matching filters"""
    
    pattern = "incremental_*.jsonl"
    files = list(results_dir.glob(pattern))
    
    if not files:
        logger.warning(f"No incremental files found in {results_dir}")
        return []
    
    # Apply filters
    filtered_files = []
    for file in files:
        config = extract_config_from_filename(file.name)
        if not config:
            continue
        
        if module and config['module'] != module:
            continue
        if presentation and config['presentation'] != presentation:
            continue
        if day is not None and config['day'] != day:
            continue
        
        filtered_files.append((file, config))
    
    logger.info(f"Found {len(filtered_files)} matching files")
    return filtered_files


def create_summary_table(all_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary table from metrics"""
    
    summary_data = []
    for result in all_metrics:
        if 'metrics' not in result:
            continue
        
        metrics = result['metrics']
        summary_data.append({
            'Module': result.get('module', 'N/A'),
            'Presentation': result.get('presentation', 'N/A'),
            'Day': result.get('day', 'N/A'),
            'N_Students': result.get('n_students', 0),
            'N_Errors': result.get('n_errors', 0),
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1': metrics.get('f1', 0),
            'PR-AUC': metrics.get('pr_auc', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            'Specificity': metrics.get('specificity', 0),
        })
    
    if not summary_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(summary_data)
    
    # Sort by Module, Presentation, Day
    if 'Module' in df.columns and 'Day' in df.columns:
        df = df.sort_values(['Module', 'Presentation', 'Day'])
    
    return df


def create_table4_format(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create Table 4 format: PR-AUC by day (aggregated across all modules)
    
    This replicates the format from the paper's Table 4, showing how
    PR-AUC changes from day 0 to day 11.
    """
    if 'Day' not in summary_df.columns or 'PR-AUC' not in summary_df.columns:
        logger.warning("Cannot create Table 4 format: missing Day or PR-AUC columns")
        return pd.DataFrame()
    
    # Aggregate by day (mean across all modules/presentations)
    day_metrics = summary_df.groupby('Day').agg({
        'PR-AUC': ['mean', 'std', 'count'],
        'ROC-AUC': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'N_Students': 'sum'
    }).round(4)
    
    # Flatten column names
    day_metrics.columns = ['_'.join(col).strip() for col in day_metrics.columns.values]
    day_metrics = day_metrics.reset_index()
    
    # Rename for clarity
    day_metrics = day_metrics.rename(columns={
        'PR-AUC_mean': 'LLM (mean)',
        'PR-AUC_std': 'LLM (std)',
        'PR-AUC_count': 'N_configs'
    })
    
    return day_metrics


def create_per_module_table(summary_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create per-module tables showing metrics by day"""
    if 'Module' not in summary_df.columns or 'Day' not in summary_df.columns:
        return {}
    
    module_tables = {}
    numeric_cols = ['PR-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall']
    
    for module in sorted(summary_df['Module'].unique()):
        module_data = summary_df[summary_df['Module'] == module]
        module_by_day = module_data.groupby('Day')[numeric_cols].mean().round(4)
        module_tables[module] = module_by_day
    
    return module_tables


def print_summary_statistics(summary_df: pd.DataFrame):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'PR-AUC', 'ROC-AUC']
    
    print("\nMean metrics across all configurations:")
    for col in numeric_cols:
        if col in summary_df.columns:
            mean_val = summary_df[col].mean()
            std_val = summary_df[col].std()
            print(f"  {col:12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\nTotal configurations: {len(summary_df)}")
    print(f"Total students: {summary_df['N_Students'].sum()}")
    print(f"Total errors: {summary_df['N_Errors'].sum()}")
    
    # Group by module
    if 'Module' in summary_df.columns:
        print("\nBy Module:")
        module_stats = summary_df.groupby('Module')[numeric_cols].mean()
        print(module_stats.to_string())
    
    # Group by day (key metric for paper replication)
    if 'Day' in summary_df.columns:
        print("\nBy Day (Paper Replication Format):")
        day_stats = summary_df.groupby('Day')[numeric_cols].mean()
        print(day_stats.to_string())
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate metrics from saved experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate for all incremental files
  python -m llm.calculate_metrics --results_dir ./llm_experiments/results/paper_replication
  
  # Calculate for specific file
  python -m llm.calculate_metrics --incremental_file ./results/incremental_BBB_2014J_day0.jsonl
  
  # Calculate from full results JSON
  python -m llm.calculate_metrics --full_results ./results/paper_replication_full.json
  
  # Filter by module
  python -m llm.calculate_metrics --results_dir ./results --module BBB
"""
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--results_dir', type=str,
                            help='Directory containing incremental result files')
    input_group.add_argument('--incremental_file', type=str,
                            help='Single incremental JSONL file')
    input_group.add_argument('--full_results', type=str,
                            help='Full results JSON file')
    
    # Filters (only work with --results_dir)
    parser.add_argument('--module', type=str, default=None,
                       help='Filter by module (e.g., BBB)')
    parser.add_argument('--presentation', type=str, default=None,
                       help='Filter by presentation (e.g., 2014J)')
    parser.add_argument('--day', type=int, default=None,
                       help='Filter by day (e.g., 0)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for summary files (default: same as input)')
    parser.add_argument('--output_prefix', type=str, default='metrics_summary',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Collect all results
    all_metrics = []
    
    if args.results_dir:
        # Process all incremental files in directory
        results_dir = Path(args.results_dir)
        
        if not results_dir.exists():
            logger.error(f"Directory not found: {results_dir}")
            return
        
        file_configs = find_incremental_files(
            results_dir,
            module=args.module,
            presentation=args.presentation,
            day=args.day
        )
        
        for file_path, config in file_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {file_path.name}")
            logger.info(f"Config: {config}")
            logger.info(f"{'='*60}")
            
            results = load_incremental_results(file_path)
            
            if results:
                metrics_result = calculate_metrics_for_results(results, config)
                if metrics_result:
                    all_metrics.append(metrics_result)
                    
                    # Print individual result
                    print(f"\n{file_path.name}:")
                    print(f"  Students: {metrics_result['n_students']}")
                    print(f"  Errors: {metrics_result['n_errors']}")
                    m = metrics_result['metrics']
                    print(f"  PR-AUC: {m.get('pr_auc', 0):.4f}")
                    print(f"  ROC-AUC: {m.get('roc_auc', 0):.4f}")
                    print(f"  F1: {m.get('f1', 0):.4f}")
        
        output_dir = Path(args.output_dir) if args.output_dir else results_dir
        
    elif args.incremental_file:
        # Process single incremental file
        file_path = Path(args.incremental_file)
        config = extract_config_from_filename(file_path.name)
        
        results = load_incremental_results(file_path)
        
        if results:
            metrics_result = calculate_metrics_for_results(results, config)
            if metrics_result:
                all_metrics.append(metrics_result)
                # Print key metrics
                m = metrics_result['metrics']
                print(f"\nMetrics for {file_path.name}:")
                print(f"  Students: {metrics_result['n_students']}")
                print(f"  PR-AUC: {m.get('pr_auc', 0):.4f}")
                print(f"  ROC-AUC: {m.get('roc_auc', 0):.4f}")
                print(f"  F1: {m.get('f1', 0):.4f}")
        
        output_dir = Path(args.output_dir) if args.output_dir else file_path.parent
        
    elif args.full_results:
        # Process full results JSON
        file_path = Path(args.full_results)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # data is a list of configuration results
        for config_result in data:
            if 'predictions' not in config_result:
                continue
            
            config = {
                'module': config_result.get('module'),
                'presentation': config_result.get('presentation'),
                'day': config_result.get('days_to_cutoff')
            }
            
            predictions = config_result['predictions']
            
            if predictions:
                metrics_result = calculate_metrics_for_results(predictions, config)
                if metrics_result:
                    all_metrics.append(metrics_result)
        
        output_dir = Path(args.output_dir) if args.output_dir else file_path.parent
    
    # Create summary
    if not all_metrics:
        logger.warning("No metrics calculated")
        return
    
    logger.info(f"\nTotal configurations processed: {len(all_metrics)}")
    
    # Create summary table
    summary_df = create_summary_table(all_metrics)
    
    if summary_df.empty:
        logger.warning("No data for summary table")
        return
    
    # Print summary table
    print("\n" + "="*80)
    print("METRICS SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Print statistics
    print_summary_statistics(summary_df)
    
    # Create Table 4 format (PR-AUC by day)
    table4_df = create_table4_format(summary_df)
    if not table4_df.empty:
        print("\n" + "="*80)
        print("TABLE 4 FORMAT: PR-AUC BY DAY (LLM Results)")
        print("="*80)
        print("\nThis table shows the average PR-AUC across all modules for each day.")
        print("Compare with paper Table 4 to see LLM vs traditional ML performance.\n")
        print(table4_df.to_string(index=False))
        print("="*80 + "\n")
    
    # Create per-module tables
    module_tables = create_per_module_table(summary_df)
    if module_tables:
        print("\n" + "="*80)
        print("PER-MODULE METRICS BY DAY")
        print("="*80)
        for module, table in module_tables.items():
            print(f"\nModule: {module}")
            print("-" * 60)
            print(table.to_string())
        print("="*80 + "\n")
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV (all configurations)
    csv_file = output_dir / f"{args.output_prefix}.csv"
    summary_df.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"\n✓ Summary table saved to: {csv_file}")
    
    # Save Table 4 format (by day)
    if not table4_df.empty:
        table4_file = output_dir / f"{args.output_prefix}_by_day.csv"
        table4_df.to_csv(table4_file, index=False, float_format='%.4f')
        logger.info(f"✓ Table 4 format (by day) saved to: {table4_file}")
    
    # Save per-module tables
    if module_tables:
        for module, table in module_tables.items():
            module_file = output_dir / f"{args.output_prefix}_module_{module}.csv"
            table.to_csv(module_file, float_format='%.4f')
        logger.info(f"✓ Per-module tables saved (one file per module)")
    
    # Save full metrics JSON
    json_file = output_dir / f"{args.output_prefix}_full.json"
    with open(json_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"✓ Full metrics saved to: {json_file}")
    
    # Save statistics
    stats_file = output_dir / f"{args.output_prefix}_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("METRICS SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'PR-AUC', 'ROC-AUC']
        
        f.write("Overall Statistics:\n")
        f.write("-"*40 + "\n")
        for col in numeric_cols:
            if col in summary_df.columns:
                mean_val = summary_df[col].mean()
                std_val = summary_df[col].std()
                min_val = summary_df[col].min()
                max_val = summary_df[col].max()
                f.write(f"{col:12s}: {mean_val:.4f} ± {std_val:.4f} (min={min_val:.4f}, max={max_val:.4f})\n")
        
        f.write(f"\nTotal configurations: {len(summary_df)}\n")
        f.write(f"Total students: {summary_df['N_Students'].sum()}\n")
        f.write(f"Total errors: {summary_df['N_Errors'].sum()}\n")
        
        if 'Module' in summary_df.columns:
            f.write("\n\nBy Module:\n")
            f.write("-"*40 + "\n")
            module_stats = summary_df.groupby('Module')[numeric_cols].mean()
            f.write(module_stats.to_string())
        
        if 'Day' in summary_df.columns:
            f.write("\n\nBy Day (Paper Replication Format):\n")
            f.write("-"*40 + "\n")
            day_stats = summary_df.groupby('Day')[numeric_cols].mean()
            f.write(day_stats.to_string())
            
            # Add Table 4 format
            if not table4_df.empty:
                f.write("\n\nTable 4 Format (PR-AUC by Day):\n")
                f.write("-"*40 + "\n")
                f.write("Compare with paper Table 4 to evaluate LLM vs traditional ML.\n\n")
                f.write(table4_df.to_string(index=False))
    
    logger.info(f"✓ Statistics saved to: {stats_file}")
    
    print(f"\n{'='*80}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

