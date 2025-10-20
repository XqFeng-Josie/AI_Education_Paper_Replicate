"""
Evaluation metrics for at-risk student prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


def convert_risk_level_to_binary(risk_level: str) -> int:
    """
    Convert risk level string to binary prediction
    
    Following traditional ML convention (y = 1 - submitted):
    - 1 = at-risk (will NOT submit)
    - 0 = not at-risk (will submit)
    """
    if risk_level in ["High Risk", "high", "High"]:
        return 1  # At-risk (will NOT submit)
    else:
        return 0  # Not at-risk (will submit)


def convert_risk_score_to_binary(risk_score: float, threshold: float = 5.0) -> int:
    """
    Convert risk score to binary prediction
    
    Following traditional ML convention (y = 1 - submitted):
    - 1 = at-risk (high risk score, will NOT submit)
    - 0 = not at-risk (low risk score, will submit)
    """
    return 1 if risk_score >= threshold else 0


def extract_predictions_and_labels(results: List[Dict[str, Any]], 
                                   use_risk_level: bool = True,
                                   threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract predictions and ground truth labels from results
    
    Args:
        results: List of prediction results
        use_risk_level: Use categorical risk level vs numeric risk score
        threshold: Threshold for converting risk score to binary
        
    Returns:
        Tuple of (predictions, labels, risk_scores)
    """
    predictions = []
    labels = []
    risk_scores = []
    
    for result in results:
        if 'error' in result and not 'final_decision' in result:
            continue
        
        # Get ground truth and convert to at-risk label
        # Following traditional ML: y = 1 - submitted
        # submitted=1 (did submit) → y=0 (not at-risk)
        # submitted=0 (did NOT submit) → y=1 (at-risk)
        submitted = result.get('ground_truth', None)
        if submitted is None:
            continue
        label = 1 - submitted  # Convert to at-risk label
        
        # Get prediction
        final_decision = result.get('final_decision', {})
        
        if use_risk_level:
            risk_level = final_decision.get('final_risk_level', 'Medium Risk')
            pred = convert_risk_level_to_binary(risk_level)
        else:
            risk_score = final_decision.get('aggregated_risk_score', 5.0)
            pred = convert_risk_score_to_binary(risk_score, threshold)
            risk_scores.append(risk_score)
        
        predictions.append(pred)
        labels.append(label)
        
        # Also collect risk score for AUC calculation
        if 'aggregated_risk_score' in final_decision:
            risk_scores.append(final_decision['aggregated_risk_score'])
        else:
            risk_scores.append(5.0 if pred == 1 else 7.0)  # Default scores
    
    return np.array(predictions), np.array(labels), np.array(risk_scores)


def calculate_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate Precision-Recall AUC"""
    try:
        precision, recall, _ = precision_recall_curve(labels, -scores)  # Negative because higher score = higher risk
        pr_auc = auc(recall, precision)
        return pr_auc
    except Exception as e:
        logger.error(f"Error calculating PR-AUC: {e}")
        return 0.0


def calculate_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate ROC AUC"""
    try:
        roc_auc = roc_auc_score(labels, -scores)  # Negative because higher score = higher risk
        return roc_auc
    except Exception as e:
        logger.error(f"Error calculating ROC-AUC: {e}")
        return 0.0


def evaluate_predictions(results: List[Dict[str, Any]], 
                        use_risk_level: bool = True,
                        threshold: float = 5.0) -> Dict[str, Any]:
    """
    Comprehensive evaluation of predictions
    
    Args:
        results: List of prediction results
        use_risk_level: Use categorical risk level vs numeric risk score
        threshold: Threshold for risk score
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating predictions")
    
    # Extract predictions and labels
    predictions, labels, risk_scores = extract_predictions_and_labels(
        results, use_risk_level, threshold
    )
    
    if len(predictions) == 0:
        logger.error("No valid predictions found")
        return {}
    
    logger.info(f"Evaluating {len(predictions)} predictions")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    logger.info(f"Prediction distribution: {np.bincount(predictions)}")
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = np.mean(predictions == labels)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    # True/False Positives/Negatives
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC scores
    if len(risk_scores) == len(labels):
        metrics['pr_auc'] = calculate_pr_auc(labels, risk_scores)
        metrics['roc_auc'] = calculate_roc_auc(labels, risk_scores)
    
    # Classification report
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # Sample info
    metrics['n_samples'] = len(predictions)
    metrics['n_positive'] = int(np.sum(labels))
    metrics['n_negative'] = int(len(labels) - np.sum(labels))
    
    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]):
    """Print evaluation metrics summary"""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nSample Information:")
    print(f"  Total Samples: {metrics.get('n_samples', 'N/A')}")
    print(f"  Positive (Submitted): {metrics.get('n_positive', 'N/A')}")
    print(f"  Negative (Not Submitted): {metrics.get('n_negative', 'N/A')}")
    
    print(f"\nCore Metrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
    
    if 'specificity' in metrics:
        print(f"  Specificity: {metrics.get('specificity', 0):.4f}")
    
    print(f"\nAUC Scores:")
    print(f"  PR-AUC:  {metrics.get('pr_auc', 0):.4f}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    print(f"  TN: {cm[0][0]:4d}  FP: {cm[0][1]:4d}")
    print(f"  FN: {cm[1][0]:4d}  TP: {cm[1][1]:4d}")
    
    print("="*60 + "\n")


def compare_with_baseline(llm_metrics: Dict[str, Any], 
                         baseline_metrics: Dict[str, Any]) -> pd.DataFrame:
    """Compare LLM results with baseline"""
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'PR-AUC', 'ROC-AUC'],
        'LLM': [
            llm_metrics.get('accuracy', 0),
            llm_metrics.get('precision', 0),
            llm_metrics.get('recall', 0),
            llm_metrics.get('f1', 0),
            llm_metrics.get('pr_auc', 0),
            llm_metrics.get('roc_auc', 0)
        ],
        'Baseline': [
            baseline_metrics.get('accuracy', 0),
            baseline_metrics.get('precision', 0),
            baseline_metrics.get('recall', 0),
            baseline_metrics.get('f1', 0),
            baseline_metrics.get('pr_auc', 0),
            baseline_metrics.get('roc_auc', 0)
        ]
    })
    
    comparison['Improvement'] = comparison['LLM'] - comparison['Baseline']
    comparison['Improvement %'] = (comparison['Improvement'] / comparison['Baseline'] * 100).round(2)
    
    return comparison





