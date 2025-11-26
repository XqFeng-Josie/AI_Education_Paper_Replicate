"""
Step 3: Evaluation
Calculate metrics and generate reports
"""
import pandas as pd
import json
import os
from utils import (
    calculate_metrics,
    plot_confusion_matrix,
    print_metrics_table,
    get_random_baseline_predictions,
    save_json,
)
import config


def evaluate_predictions(pred_file: str, model_name: str):
    """
    Evaluate predictions from a CSV file
    
    Args:
        pred_file: Path to predictions CSV
        model_name: Name of model for display
    
    Returns:
        Dictionary with metrics
    """
    # Load predictions
    df = pd.read_csv(pred_file)
    
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, config.LABELS)
    
    # Print results
    print_metrics_table(metrics, model_name)
    
    # Plot confusion matrix
    cm_path = pred_file.replace(".csv", "_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, config.LABELS, cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    
    return metrics


def evaluate_random_baseline():
    """
    Evaluate random baseline using training label distribution
    
    Returns:
        Dictionary with metrics
    """
    print("\n" + "="*60)
    print("Evaluating Random Baseline")
    print("="*60)
    
    # Load train and test data
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    
    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()
    
    # Generate random predictions
    random_preds = get_random_baseline_predictions(test_labels, train_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(test_labels, random_preds, config.LABELS)
    
    # Print results
    print_metrics_table(metrics, "Random Baseline")
    
    # Plot confusion matrix
    cm_path = os.path.join(config.RESULTS_DIR, "random_baseline_confusion_matrix.png")
    plot_confusion_matrix(test_labels, random_preds, config.LABELS, cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    
    return metrics


def main():
    """Main evaluation function"""
    print("="*60)
    print("CIMA Evaluation")
    print("="*60)
    
    all_results = {}
    
    # 1. Evaluate random baseline
    baseline_metrics = evaluate_random_baseline()
    all_results["random_baseline"] = baseline_metrics
    
    # 2. Evaluate Llama predictions (if exists)
    llama_pred_file = os.path.join(config.RESULTS_DIR, "llama_predictions.csv")
    if os.path.exists(llama_pred_file):
        print("\n" + "="*60)
        print("Evaluating Llama-3.1-405B")
        print("="*60)
        llama_metrics = evaluate_predictions(llama_pred_file, "Llama-3.1-405B")
        all_results["llama_3.1_405b"] = llama_metrics
    else:
        print(f"\nWarning: Llama predictions not found at {llama_pred_file}")
    
    # 3. Evaluate Mistral predictions (if exists)
    mistral_pred_file = os.path.join(config.RESULTS_DIR, "mistral_predictions.csv")
    if os.path.exists(mistral_pred_file):
        print("\n" + "="*60)
        print("Evaluating Mistral-7B")
        print("="*60)
        mistral_metrics = evaluate_predictions(mistral_pred_file, "Mistral-7B")
        all_results["mistral_7b"] = mistral_metrics
    else:
        print(f"\nWarning: Mistral predictions not found at {mistral_pred_file}")
    # 4. Evaluate Llama-3.1-405B 5-shot predictions (if exists)
    llama_5shot_pred_file = os.path.join(config.RESULTS_DIR, "llama_5shot_predictions.csv")
    if os.path.exists(llama_5shot_pred_file):
        print("\n" + "="*60)
        print("Evaluating Llama-3.1-405B 5-shot")
        print("="*60)
        llama_5shot_metrics = evaluate_predictions(llama_5shot_pred_file, "Llama-3.1-405B 5-shot")
        all_results["llama_3.1_405b_5shot"] = llama_5shot_metrics
    else:
        print(f"\nWarning: Llama-3.1-405B 5-shot predictions not found at {llama_5shot_pred_file}")
    # 5. Evaluate Llama-3.1-405B 20-shot predictions (if exists)
    llama_20shot_pred_file = os.path.join(config.RESULTS_DIR, "llama_20shot_predictions.csv")
    if os.path.exists(llama_20shot_pred_file):
        print("\n" + "="*60)
        print("Evaluating Llama-3.1-405B 20-shot")
        print("="*60)
        llama_20shot_metrics = evaluate_predictions(llama_20shot_pred_file, "Llama-3.1-405B 20-shot")
        all_results["llama_3.1_405b_20shot"] = llama_20shot_metrics
    else:
        print(f"\nWarning: Llama-3.1-405B 20-shot predictions not found at {llama_20shot_pred_file}")
    # 6. Evaluate Llama-3.1-405B full-context predictions (if exists)
    llama_full_context_pred_file = os.path.join(config.RESULTS_DIR, "llama_context_predictions.csv")
    if os.path.exists(llama_full_context_pred_file):
        print("\n" + "="*60)
        print("Evaluating Llama-3.1-405B full-context")
        print("="*60)
        llama_full_context_metrics = evaluate_predictions(llama_full_context_pred_file, "Llama-3.1-405B full-context")
        all_results["llama_3.1_405b_full_context"] = llama_full_context_metrics
    else:
        print(f"\nWarning: Llama-3.1-405B full-context predictions not found at {llama_full_context_pred_file}")
    # Save all metrics
    metrics_file = os.path.join(config.RESULTS_DIR, "metrics.json")
    save_json(all_results, metrics_file)
    print(f"\nSaved all metrics to {metrics_file}")
    
    # Print summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Macro F1 Comparison")
    print("="*60)
    for model, metrics in all_results.items():
        print(f"{model:25s}: {metrics['macro_f1']:.4f}")
    print("="*60)
    
    # Compare with paper results
    print("\nPaper Results (for reference):")
    print(f"{'Random baseline':25s}: 0.29")
    print(f"{'GPT-4 zero-shot':25s}: 0.49")
    print(f"{'Mistral-7B zero-shot':25s}: 0.11")
    print(f"{'RoBERTa fine-tuned':25s}: 0.63 (best)")
    print("="*60)


if __name__ == "__main__":
    main()
