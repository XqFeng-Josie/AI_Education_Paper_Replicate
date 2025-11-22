"""
Utility functions for LLM+MLP pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
import os

# Add parent directory to path to import data_preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm_mlp.data_loader import InstanceData


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    model_name: str
    model_path: str
    hidden_size: int = None


MODEL_MAPPING = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.3-70b-instruct": "meta-llama/Meta-Llama-3.3-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


class UserHistory:
    """Aggregate user learning history from train data."""
    
    def __init__(self):
        self.user_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'correct': 0,
            'by_format': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'by_pos': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'days': 0.0,
        })
    
    def add_instance(self, instance: InstanceData, label: float):
        """Add an instance to the user's history."""
        user_id = instance.user
        stats = self.user_stats[user_id]
        
        # Update overall stats
        stats['total_attempts'] += 1
        if label > 0.5:
            stats['correct'] += 1
        
        # Update format-specific stats
        format_stats = stats['by_format'][instance.format]
        format_stats['total'] += 1
        if label > 0.5:
            format_stats['correct'] += 1
        
        # Update POS-specific stats
        pos_stats = stats['by_pos'][instance.part_of_speech]
        pos_stats['total'] += 1
        if label > 0.5:
            pos_stats['correct'] += 1
        
        # Update days (use the maximum days seen for this user)
        stats['days'] = max(stats['days'], instance.days if instance.days else 0.0)
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get aggregated stats for a user."""
        if user_id not in self.user_stats:
            return {
                'total_attempts': 0,
                'accuracy': 0.0,
                'days': 0.0,
                'format_acc': {},
                'pos_acc': {},
            }
        
        stats = self.user_stats[user_id]
        total = stats['total_attempts']
        accuracy = (stats['correct'] / total * 100) if total > 0 else 0.0
        
        # Calculate format-specific accuracies
        format_acc = {}
        for fmt, fmt_stats in stats['by_format'].items():
            if fmt_stats['total'] > 0:
                format_acc[fmt] = fmt_stats['correct'] / fmt_stats['total'] * 100
            else:
                format_acc[fmt] = 0.0
        
        # Calculate POS-specific accuracies
        pos_acc = {}
        for pos, pos_stats in stats['by_pos'].items():
            if pos_stats['total'] > 0:
                pos_acc[pos] = pos_stats['correct'] / pos_stats['total'] * 100
            else:
                pos_acc[pos] = 0.0
        
        return {
            'total_attempts': total,
            'accuracy': accuracy,
            'days': stats['days'],
            'format_acc': format_acc,
            'pos_acc': pos_acc,
        }


def create_prompt(instance: InstanceData, user_history: UserHistory) -> str:
    """
    Create a contextualized prompt for LLM with user history.
    
    Args:
        instance: The instance to create a prompt for
        user_history: User learning history from train data
    
    Returns:
        Formatted prompt string
    """
    user_stats = user_history.get_user_stats(instance.user)
    
    # Get morphological features as a string
    morph_keys = list(instance.morphological_features.keys())
    morph_str = ','.join(morph_keys) if morph_keys else 'none'
    
    # Build format accuracies string
    format_acc_parts = []
    for fmt in ['listen', 'reverse_translate', 'reverse_tap']:
        acc = user_stats['format_acc'].get(fmt, 0.0)
        format_acc_parts.append(f"{fmt}={acc:.1f}%")
    format_acc_str = ', '.join(format_acc_parts)
    
    # Build the prompt (without specific user_id for better generalization)
    prompt = (
        f"User {instance.user} learning history: {user_stats['days']:.1f} days, "
        # f"Learner's history: {user_stats['days']:.1f} days of practice, " # TODO: remove this line
        f"{user_stats['total_attempts']} attempts, {user_stats['accuracy']:.1f}% correct. "
        f"Performance by format: {format_acc_str}. "
        f"Current token: '{instance.token}' (POS: {instance.part_of_speech}, "
        f"Format: {instance.format}, Morphology: {morph_str}, "
        f"DepLabel: {instance.dependency_label}). "
        f"Will the learner answer this correctly?"
    )
    
    return prompt


def create_exercise_aware_prompt(
    instance: InstanceData,
    all_instances: List[InstanceData],
    current_idx: int,
    user_history: UserHistory
) -> str:
    """
    Create a contextualized prompt with full exercise context.
    
    Args:
        instance: The current token instance
        all_instances: All instances in this exercise
        current_idx: Index of current instance in the exercise
        user_history: User learning history from train data
    
    Returns:
        Formatted prompt string with exercise context
    """
    user_stats = user_history.get_user_stats(instance.user)
    
    # Build format accuracies string
    format_acc_parts = []
    for fmt in ['listen', 'reverse_translate', 'reverse_tap']:
        acc = user_stats['format_acc'].get(fmt, 0.0)
        format_acc_parts.append(f"{fmt}={acc:.1f}%")
    format_acc_str = ', '.join(format_acc_parts)
    
    # Get all tokens in the exercise to provide context
    exercise_tokens = [inst.token for inst in all_instances]
    exercise_str = ' '.join(exercise_tokens)
    
    # Get morphological features for current token
    morph_keys = list(instance.morphological_features.keys())
    morph_str = ','.join(morph_keys) if morph_keys else 'none'
    
    # Build the prompt with exercise context
    prompt = (
        # f"User {instance.user} learning history: {user_stats['days']:.1f} days, "
        f"Learner's history: {user_stats['days']:.1f} days of practice, " # TODO: remove this line
        f"{user_stats['total_attempts']} attempts, {user_stats['accuracy']:.1f}% correct. "
        f"Performance by format: {format_acc_str}. "
        f"Exercise (all tokens): \"{exercise_str}\". "
        f"Current token #{current_idx + 1}/{len(all_instances)}: '{instance.token}' "
        f"(POS: {instance.part_of_speech}, Format: {instance.format}, "
        f"Morphology: {morph_str}, DepLabel: {instance.dependency_label}). "
        f"Will the learner answer this token correctly?"
    )
    
    return prompt



def build_user_history_from_train(train_instances: List[InstanceData], 
                                   train_labels: Dict[str, float]) -> UserHistory:
    """
    Build user history from training data.
    
    Args:
        train_instances: List of training instances
        train_labels: Dictionary of instance_id -> label
    
    Returns:
        UserHistory object with aggregated statistics
    """
    history = UserHistory()
    
    for instance in train_instances:
        if instance.instance_id in train_labels:
            label = train_labels[instance.instance_id]
            history.add_instance(instance, label)
    
    return history
