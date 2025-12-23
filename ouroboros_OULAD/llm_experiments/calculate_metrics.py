#!/usr/bin/env python
"""
Calculate metrics from experiment result files
"""
import json
import argparse
import os
from pathlib import Path
import numpy as np
from glob import glob
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, 
    accuracy_score, f1_score, precision_score, recall_score
)


def find_result_files(paths):
    """Find all JSON result files from given paths (files or directories)"""
    result_files = []
    
    for path in paths:
        path_obj = Path(path)
        
        if path_obj.is_file():
            if path_obj.suffix == '.json':
                result_files.append(str(path_obj))
        elif path_obj.is_dir():
            # Find all JSON files in directory
            json_files = list(path_obj.glob('*.json'))
            result_files.extend([str(f) for f in json_files])
            
            # Also check results subdirectory if it exists
            results_dir = path_obj / 'results'
            if results_dir.exists() and results_dir.is_dir():
                json_files = list(results_dir.glob('*.json'))
                result_files.extend([str(f) for f in json_files])
        else:
            print(f"Warning: {path} is not a valid file or directory")
    
    return sorted(set(result_files))  # Remove duplicates and sort


def load_results(result_file):
    """Load predictions and labels from result file"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    predictions = []
    true_labels = []
    
    # Try both 'student_results' and 'students' keys for compatibility
    student_data_key = 'student_results' if 'student_results' in data else 'students'
    
    if student_data_key in data:
        for student_data in data[student_data_key].values():
            if 'prediction' in student_data and 'true_label' in student_data:
                pred = student_data['prediction']
                label = student_data['true_label']
                if pred is not None and label is not None:
                    predictions.append(float(pred))
                    true_labels.append(int(label))
    
    return predictions, true_labels


def calculate_metrics(predictions, true_labels, threshold=0.5):
    """Calculate all metrics"""
    if len(predictions) == 0:
        return None
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    pr_auc = auc(recall, precision)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(true_labels, predictions)
    except:
        roc_auc = 0.0
    
    # Binary predictions
    binary_preds = [1 if p >= threshold else 0 for p in predictions]
    
    # Accuracy, F1, Precision, Recall
    accuracy = accuracy_score(true_labels, binary_preds)
    f1 = f1_score(true_labels, binary_preds, zero_division=0)
    precision_val = precision_score(true_labels, binary_preds, zero_division=0)
    recall_val = recall_score(true_labels, binary_preds, zero_division=0)
    
    # Label distribution
    n_positive = sum(true_labels)
    n_negative = len(true_labels) - n_positive
    
    # Prediction statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    pred_min = np.min(predictions)
    pred_max = np.max(predictions)
    
    return {
        'n_samples': len(predictions),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision_val,
        'recall': recall_val,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_min': pred_min,
        'pred_max': pred_max
    }


def print_metrics(result_file, metrics, verbose=False):
    """Print metrics for a single file"""
    filename = Path(result_file).name
    
    if metrics is None:
        print(f"{filename}: No valid predictions")
        return
    
    print(f"\n{'=' * 70}")
    print(f"File: {filename}")
    print(f"{'=' * 70}")
    print(f"Samples:       {metrics['n_samples']}")
    print(f"Positive:      {metrics['n_positive']} ({metrics['n_positive']/metrics['n_samples']*100:.1f}%)")
    print(f"Negative:      {metrics['n_negative']} ({metrics['n_negative']/metrics['n_samples']*100:.1f}%)")
    print(f"{'-' * 70}")
    print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"F1 Score:      {metrics['f1_score']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    
    if verbose:
        print(f"{'-' * 70}")
        print(f"Pred Mean:     {metrics['pred_mean']:.4f}")
        print(f"Pred Std:      {metrics['pred_std']:.4f}")
        print(f"Pred Range:    [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}]")


def print_comparison_table(results):
    """Print comparison table for multiple files"""
    if len(results) <= 1:
        return
    
    print(f"\n{'=' * 120}")
    print("COMPARISON TABLE")
    print(f"{'=' * 120}")
    
    # Header
    print(f"{'File':<40} {'N':<6} {'PR-AUC':<8} {'ROC-AUC':<8} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8}")
    print(f"{'-' * 120}")
    
    # Sort by filename
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    for filename, metrics in sorted_results:
        if metrics is None:
            print(f"{filename:<40} {'N/A':<6}")
        else:
            print(f"{filename:<40} "
                  f"{metrics['n_samples']:<6} "
                  f"{metrics['pr_auc']:<8.4f} "
                  f"{metrics['roc_auc']:<8.4f} "
                  f"{metrics['accuracy']:<8.4f} "
                  f"{metrics['f1_score']:<8.4f} "
                  f"{metrics['precision']:<8.4f} "
                  f"{metrics['recall']:<8.4f}")


def extract_dayoff_from_filename(filename):
    """Extract days_to_cutoff from filename"""
    import re
    match = re.search(r'days(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def print_dayoff_trend(results):
    """Print trend analysis by days_to_cutoff"""
    dayoff_results = {}
    
    for filename, metrics in results.items():
        if metrics is not None:
            dayoff = extract_dayoff_from_filename(filename)
            if dayoff is not None:
                if dayoff not in dayoff_results:
                    dayoff_results[dayoff] = []
                dayoff_results[dayoff].append((filename, metrics))
    
    if len(dayoff_results) < 1:
        return
    
    print(f"\n{'=' * 120}")
    print("TREND ANALYSIS BY DAYS TO CUTOFF (横向对比)")
    print(f"{'=' * 120}")
    
    # Sort by dayoff
    sorted_dayoff = sorted(dayoff_results.items(), key=lambda x: x[0])
    
    # Header
    print(f"{'Days':<6} {'N':<6} {'Pos%':<8} {'PR-AUC':<10} {'ROC-AUC':<10} {'Accuracy':<10} {'F1':<10} {'Prec':<10} {'Rec':<10}")
    print(f"{'-' * 120}")
    
    for dayoff, file_metrics_list in sorted_dayoff:
        # If multiple files have the same dayoff, average the metrics
        if len(file_metrics_list) == 1:
            _, metrics = file_metrics_list[0]
            pos_pct = metrics['n_positive'] / metrics['n_samples'] * 100
            print(f"{dayoff:<6} "
                  f"{metrics['n_samples']:<6} "
                  f"{pos_pct:<8.1f} "
                  f"{metrics['pr_auc']:<10.4f} "
                  f"{metrics['roc_auc']:<10.4f} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f}")
        else:
            # Average metrics across multiple files with same dayoff
            n_samples = sum(m['n_samples'] for _, m in file_metrics_list)
            n_positive = sum(m['n_positive'] for _, m in file_metrics_list)
            pos_pct = n_positive / n_samples * 100 if n_samples > 0 else 0
            
            avg_pr_auc = np.mean([m['pr_auc'] for _, m in file_metrics_list])
            avg_roc_auc = np.mean([m['roc_auc'] for _, m in file_metrics_list])
            avg_accuracy = np.mean([m['accuracy'] for _, m in file_metrics_list])
            avg_f1 = np.mean([m['f1_score'] for _, m in file_metrics_list])
            avg_prec = np.mean([m['precision'] for _, m in file_metrics_list])
            avg_rec = np.mean([m['recall'] for _, m in file_metrics_list])
            
            print(f"{dayoff:<6} "
                  f"{n_samples:<6} "
                  f"{pos_pct:<8.1f} "
                  f"{avg_pr_auc:<10.4f} "
                  f"{avg_roc_auc:<10.4f} "
                  f"{avg_accuracy:<10.4f} "
                  f"{avg_f1:<10.4f} "
                  f"{avg_prec:<10.4f} "
                  f"{avg_rec:<10.4f}")


def print_overall_summary(results):
    """Print overall summary statistics"""
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) == 0:
        return
    
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    
    total_files = len(results)
    valid_files = len(valid_results)
    total_samples = sum(m['n_samples'] for m in valid_results.values())
    total_positive = sum(m['n_positive'] for m in valid_results.values())
    
    avg_pr_auc = np.mean([m['pr_auc'] for m in valid_results.values()])
    avg_roc_auc = np.mean([m['roc_auc'] for m in valid_results.values()])
    avg_accuracy = np.mean([m['accuracy'] for m in valid_results.values()])
    avg_f1 = np.mean([m['f1_score'] for m in valid_results.values()])
    avg_precision = np.mean([m['precision'] for m in valid_results.values()])
    avg_recall = np.mean([m['recall'] for m in valid_results.values()])
    
    print(f"Total Files:         {total_files}")
    print(f"Valid Files:         {valid_files}")
    print(f"Total Samples:       {total_samples}")
    print(f"Total Positive:      {total_positive} ({total_positive/total_samples*100:.1f}%)")
    print(f"Total Negative:      {total_samples - total_positive} ({(total_samples-total_positive)/total_samples*100:.1f}%)")
    print(f"{'-' * 80}")
    print(f"Average PR-AUC:      {avg_pr_auc:.4f}")
    print(f"Average ROC-AUC:     {avg_roc_auc:.4f}")
    print(f"Average Accuracy:    {avg_accuracy:.4f}")
    print(f"Average F1 Score:    {avg_f1:.4f}")
    print(f"Average Precision:   {avg_precision:.4f}")
    print(f"Average Recall:      {avg_recall:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from experiment results",
        epilog="Examples:\n"
               "  %(prog)s file1.json file2.json\n"
               "  %(prog)s results/\n"
               "  %(prog)s results/ --summary-only\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('paths', nargs='+', 
                       help='Result files or directories containing JSON files')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only show summary tables (skip individual file details)')
    
    args = parser.parse_args()
    
    # Find all result files
    result_files = find_result_files(args.paths)
    
    if len(result_files) == 0:
        print("No JSON result files found in the specified paths")
        return
    
    print(f"Found {len(result_files)} result file(s)")
    
    results = {}
    
    for result_file in result_files:
        try:
            predictions, true_labels = load_results(result_file)
            metrics = calculate_metrics(predictions, true_labels, args.threshold)
            
            filename = Path(result_file).name
            results[filename] = metrics
            
            if not args.summary_only:
                print_metrics(result_file, metrics, args.verbose)
        
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            results[Path(result_file).name] = None
    
    # Print summary tables
    print_overall_summary(results)
    
    if len(results) > 1:
        print_comparison_table(results)
    
    print_dayoff_trend(results)


if __name__ == "__main__":
    main()

