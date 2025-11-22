"""
Step 3: Evaluate zero-shot predictions.

This script:
1. Loads predictions from zero-shot inference
2. Loads ground truth labels
3. Computes token-level metrics (AUC, F1, Accuracy)
4. Optionally computes exercise-level metrics
5. Generates detailed analysis report
"""

import argparse
import os
import sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, log_loss

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_predictions(pred_file: str) -> dict:
    """
    Load predictions from file.
    
    Returns:
        Dictionary of instance_id -> probability
    """
    predictions = {}
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                instance_id, prob = parts
                predictions[instance_id] = float(prob)
    return predictions


def load_ground_truth(key_file: str) -> dict:
    """
    Load ground truth labels from .key file.
    
    Returns:
        Dictionary of instance_id -> label (0 or 1)
    """
    labels = {}
    with open(key_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                instance_id, label = parts
                labels[instance_id] = float(label)
    return labels


def compute_metrics(y_true: list, y_pred: list, y_pred_binary: list) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: List of ground truth labels (0 or 1)
        y_pred: List of predicted probabilities
        y_pred_binary: List of binary predictions (0 or 1)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Could not compute AUC: {e}")
        metrics['auc'] = None
    
    # F1 Score
    try:
        metrics['f1'] = f1_score(y_true, y_pred_binary)
    except Exception as e:
        print(f"Warning: Could not compute F1: {e}")
        metrics['f1'] = None
    
    # Accuracy
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    except Exception as e:
        print(f"Warning: Could not compute Accuracy: {e}")
        metrics['accuracy'] = None
    
    # Log Loss
    try:
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        metrics['log_loss'] = log_loss(y_true, y_pred_clipped)
    except Exception as e:
        print(f"Warning: Could not compute Log Loss: {e}")
        metrics['log_loss'] = None
    
    return metrics


def evaluate(pred_file: str, key_file: str, output_dir: str = None):
    """
    Evaluate predictions against ground truth.
    
    Args:
        pred_file: Path to predictions file
        key_file: Path to ground truth .key file
        output_dir: Optional directory to save detailed results
    """
    print(f"Loading predictions from: {pred_file}")
    predictions = load_predictions(pred_file)
    print(f"  Loaded {len(predictions)} predictions")
    
    print(f"\nLoading ground truth from: {key_file}")
    labels = load_ground_truth(key_file)
    print(f"  Loaded {len(labels)} labels")
    
    # Find common instances
    common_ids = set(predictions.keys()) & set(labels.keys())
    print(f"\nCommon instances: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("Error: No common instances found!")
        return
    
    # Prepare data for evaluation
    y_true = []
    y_pred = []
    y_pred_binary = []
    
    for instance_id in sorted(common_ids):
        y_true.append(labels[instance_id])
        y_pred.append(predictions[instance_id])
        # Use 0.5 as threshold for binary predictions
        y_pred_binary.append(1 if predictions[instance_id] >= 0.5 else 0)
    
    # Compute metrics
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS (Token-Level)")
    print(f"{'='*80}")
    
    metrics = compute_metrics(y_true, y_pred, y_pred_binary)
    
    print(f"AUC:      {metrics['auc']:.4f}" if metrics['auc'] is not None else "AUC:      N/A")
    print(f"F1:       {metrics['f1']:.4f}" if metrics['f1'] is not None else "F1:       N/A")
    print(f"Accuracy: {metrics['accuracy']:.4f}" if metrics['accuracy'] is not None else "Accuracy: N/A")
    print(f"Log Loss: {metrics['log_loss']:.4f}" if metrics['log_loss'] is not None else "Log Loss: N/A")
    
    # Class distribution
    num_positive = sum(y_true)
    num_negative = len(y_true) - num_positive
    print(f"\nClass Distribution:")
    print(f"  Positive (correct): {num_positive} ({num_positive/len(y_true)*100:.1f}%)")
    print(f"  Negative (incorrect): {num_negative} ({num_negative/len(y_true)*100:.1f}%)")
    
    # Prediction distribution
    pred_positive = sum(y_pred_binary)
    pred_negative = len(y_pred_binary) - pred_positive
    print(f"\nPrediction Distribution:")
    print(f"  Predicted positive: {pred_positive} ({pred_positive/len(y_pred_binary)*100:.1f}%)")
    print(f"  Predicted negative: {pred_negative} ({pred_negative/len(y_pred_binary)*100:.1f}%)")
    
    # Confusion matrix
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_binary[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred_binary[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred_binary[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_binary[i] == 0)
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    print(f"{'='*80}\n")
    
    # Save detailed results if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write("EVALUATION RESULTS (Token-Level)\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"AUC:       {metrics['auc']:.4f}\n" if metrics['auc'] is not None else "AUC:       N/A\n")
            f.write(f"F1:        {metrics['f1']:.4f}\n" if metrics['f1'] is not None else "F1:        N/A\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n" if metrics['accuracy'] is not None else "Accuracy:  N/A\n")
            f.write(f"Log Loss:  {metrics['log_loss']:.4f}\n" if metrics['log_loss'] is not None else "Log Loss:  N/A\n")
            f.write(f"\nPrecision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"\nClass Distribution:\n")
            f.write(f"  Positive: {num_positive} ({num_positive/len(y_true)*100:.1f}%)\n")
            f.write(f"  Negative: {num_negative} ({num_negative/len(y_true)*100:.1f}%)\n")
        
        print(f"Saved detailed metrics to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate zero-shot predictions'
    )
    parser.add_argument(
        '--pred',
        type=str,
        required=True,
        help='Path to predictions file'
    )
    parser.add_argument(
        '--key',
        type=str,
        required=True,
        help='Path to ground truth .key file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save detailed results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("STEP 3: EVALUATION")
    print("="*80)
    print(f"Predictions: {args.pred}")
    print(f"Ground Truth: {args.key}")
    if args.output_dir:
        print(f"Output Directory: {args.output_dir}")
    print("="*80)
    print()
    
    evaluate(args.pred, args.key, args.output_dir)
    
    print("Done!")


if __name__ == '__main__':
    main()
