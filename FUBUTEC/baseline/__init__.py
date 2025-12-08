"""
Baseline实验模块
包含论文复现的所有基础代码
"""

from .data_preprocessing import DataPreprocessor
from .models import (
    NaiveBaselineClassifier, NaiveBaselineRegressor,
    get_decision_tree_classifier, get_decision_tree_regressor,
    get_random_forest_classifier, get_random_forest_regressor
)
from .evaluation import evaluate_model, cross_validate

__all__ = [
    'DataPreprocessor',
    'NaiveBaselineClassifier',
    'NaiveBaselineRegressor',
    'get_decision_tree_classifier',
    'get_decision_tree_regressor',
    'get_random_forest_classifier',
    'get_random_forest_regressor',
    'evaluate_model',
    'cross_validate',
]

