"""
LLM实验模块
提供LLM特征提取、统一单Agent预测等功能
"""

from .unified_agent import UnifiedAgent, FewShotExampleSelector

__all__ = [
    'LLMFeatureExtractor',
    'merge_llm_features',
    'UnifiedAgent',
    'FewShotExampleSelector'
]

