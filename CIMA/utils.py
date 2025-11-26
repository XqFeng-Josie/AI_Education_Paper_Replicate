"""
Utility functions for CIMA experiment
"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple


def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save data to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of possible labels
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {},
    }
    
    # Per-class metrics
    for label in labels:
        y_true_binary = [1 if y == label else 0 for y in y_true]
        y_pred_binary = [1 if y == label else 0 for y in y_pred]
        
        metrics["per_class"][label] = {
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }
    
    return metrics


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str], save_path: str):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of possible labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_metrics_table(metrics: Dict, model_name: str = "Model"):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: Dictionary with metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'-'*60}")
    print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*60}")
    
    for label, scores in metrics["per_class"].items():
        print(f"{label:<15} {scores['precision']:<12.4f} {scores['recall']:<12.4f} {scores['f1']:<12.4f}")
    print(f"{'='*60}\n")


def get_random_baseline_predictions(y_true: List[str], train_labels: List[str]) -> List[str]:
    """
    Generate random predictions based on training label distribution
    
    Args:
        y_true: True labels (to get length)
        train_labels: Training labels to compute distribution
    
    Returns:
        List of random predictions
    """
    from collections import Counter
    
    # Calculate label distribution
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    label_probs = {label: count / total for label, count in label_counts.items()}
    
    # Generate random predictions
    labels = list(label_probs.keys())
    probs = [label_probs[label] for label in labels]
    
    np.random.seed(42)
    predictions = np.random.choice(labels, size=len(y_true), p=probs)
    
    return predictions.tolist()
