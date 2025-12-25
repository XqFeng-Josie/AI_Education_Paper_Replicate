"""
Unified Single-Agent System Usage Examples
Demonstrates how to use training set as few-shot source, ensuring test set consistency with paper
Supports OpenRouter API integration with command-line controls
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import argparse
import json
import time
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from pathlib import Path

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import unified LLM client
from llm_experiments.llm_client import create_llm_client
from llm_experiments.unified_agent import UnifiedAgent, FewShotExampleSelector
from llm_experiments.behavior_to_text import BehaviorToTextConverter
from selflearner.problem_definition import ProblemDefinition, TrainingType
from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultStorage:
    """Manages result storage and resume functionality (thread-safe)"""
    
    def __init__(self, result_file: str):
        """
        Initialize result storage
        
        Args:
            result_file: Path to result file (JSON format)
        """
        self.result_file = Path(result_file)
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}
        self.lock = threading.Lock()  # Thread-safe lock for concurrent access
        self.load()
    
    def load(self):
        """Load existing results from file"""
        if self.result_file.exists():
            try:
                with open(self.result_file, 'r') as f:
                    self.results = json.load(f)
                num_completed = len(self.results.get('student_results', {}))
                logger.info(f"Loaded {num_completed} existing predictions from {self.result_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}, starting fresh")
                self.results = {}
        else:
            self.results = {}
    
    def save(self):
        """Save results to file (thread-safe)"""
        with self.lock:
            try:
                with open(self.result_file, 'w') as f:
                    json.dump(self.results, f, indent=2)
                logger.debug(f"Saved results to {self.result_file}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
    
    def get_completed_indices(self) -> set:
        """Get set of completed student indices"""
        if 'student_results' not in self.results:
            return set()
        return set(self.results['student_results'].keys())
    
    def get_completed_label_distribution(self) -> Dict[int, int]:
        """Get label distribution of completed students"""
        if 'student_results' not in self.results:
            return {}
        
        label_counts = {}
        for idx_str, result in self.results['student_results'].items():
            true_label = result.get('true_label')
            if true_label is not None:
                label_counts[true_label] = label_counts.get(true_label, 0) + 1
        
        return label_counts
    
    def save_student_result(self, student_id: str, index: int, prediction: float, true_label: int,
                           prompt: Optional[str] = None, response: Optional[str] = None):
        """Save result for a single student (thread-safe)"""
        with self.lock:
            if 'student_results' not in self.results:
                self.results['student_results'] = {}
            if 'predictions' not in self.results:
                self.results['predictions'] = []
            if 'true_labels' not in self.results:
                self.results['true_labels'] = []
            
            student_result = {
                'student_id': student_id,
                'prediction': prediction,
                'true_label': true_label
            }
            
            # Save prompt and response if provided
            if prompt is not None:
                student_result['prompt'] = prompt
            if response is not None:
                student_result['response'] = response
            
            self.results['student_results'][str(index)] = student_result
        
        # Save outside the lock to reduce lock contention
        self.save()
    
    def get_all_results(self) -> tuple:
        """Get all predictions and true labels"""
        if 'student_results' not in self.results:
            return [], []
        
        # Sort by index to maintain order
        sorted_results = sorted(
            self.results['student_results'].items(),
            key=lambda x: int(x[0])
        )
        
        predictions = [r['prediction'] for _, r in sorted_results]
        true_labels = [r['true_label'] for _, r in sorted_results]
        
        return predictions, true_labels
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save experiment metadata (thread-safe)"""
        with self.lock:
            if 'metadata' not in self.results:
                self.results['metadata'] = {}
            self.results['metadata'].update(metadata)
        self.save()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata"""
        return self.results.get('metadata', {})


def load_oulad_data(module: str,
                    presentation: str,
                    assessment_name: str,
                    days_to_cutoff: int,
                    features: list):
    """
    Load OULAD data, ensuring test set consistency with paper
    
    Args:
        module: Module code (e.g., "BBB")
        presentation: Course presentation (e.g., "2014J")
        assessment_name: Assessment name (e.g., "TMA 1")
        days_to_cutoff: Days to cutoff (0-11)
        features: Feature list
        
    Returns:
        dict containing:
            - 'train_data': Training DataFrame
            - 'test_data': Test DataFrame
            - 'y_train': Training labels
            - 'y_test': Test labels
            - 'problem_definition': ProblemDefinition object
    """
    # Create problem definition (data split consistent with paper)
    problem_def = ProblemDefinition(
        module=module,
        presentation=presentation,
        assessment_name=assessment_name,
        days_to_cutoff=days_to_cutoff,
        y_column='submitted',
        grouping_column='submit_in',
        id_column='id_student',
        presentation_train=presentation,  # Use same presentation as training set
        training_type=TrainingType.SELFLEARNER
    )
    
    # Extract features
    fe = FeatureExtractionOulad(problem_def)
    data = fe.extract_features(features=features)
    
    train_data = data["all_train"]
    test_data = data["all_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    # Add column aliases for behavior converter compatibility
    # The behavior converter expects 'sum_click' and 'num_days', but the data uses 'sum_click_fromstart' and 'count_days_fromstart'
    if 'sum_click_fromstart' in train_data.columns:
        train_data['sum_click'] = train_data['sum_click_fromstart']
        test_data['sum_click'] = test_data['sum_click_fromstart']
    
    if 'count_days_fromstart' in train_data.columns:
        train_data['num_days'] = train_data['count_days_fromstart']
        test_data['num_days'] = test_data['count_days_fromstart']
    
    logger.info(f"Loaded data: train={len(train_data)}, test={len(test_data)}")
    logger.info(f"Train label distribution: {train_data['submitted'].value_counts().to_dict()}")
    logger.info(f"Test label distribution: {test_data['submitted'].value_counts().to_dict()}")
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'y_train': y_train,
        'y_test': y_test,
        'problem_definition': problem_def
    }


def balanced_sample_test_data(test_data: pd.DataFrame,
                               label_column: str,
                               max_students: int,
                               random_state: int = 42,
                               completed_indices: Optional[set] = None,
                               completed_label_dist: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """
    Balanced sampling from test data, maintaining label distribution as balanced as possible.
    If completed_indices and completed_label_dist are provided, considers already processed data
    to maintain overall balance (completed + newly sampled).
    
    Args:
        test_data: Test DataFrame (original indices, before any filtering)
        label_column: Label column name (e.g., 'submitted')
        max_students: Maximum number of students to have in total (completed + newly sampled)
        random_state: Random seed for reproducibility
        completed_indices: Set of already processed student indices (as strings)
        completed_label_dist: Label distribution of already processed students {label: count}
        
    Returns:
        Balanced sampled DataFrame (only newly sampled rows, excluding already completed ones)
    """
    np.random.seed(random_state)
    
    # Convert completed_indices to integers if provided
    completed_indices_int = set()
    if completed_indices:
        completed_indices_int = {int(idx) for idx in completed_indices if idx.isdigit()}
    
    # Filter out already completed samples from available test data
    available_test_data = test_data.drop(index=completed_indices_int, errors='ignore') if completed_indices_int else test_data.copy()
    
    # Count completed samples per label
    completed_counts = completed_label_dist if completed_label_dist else {}
    total_completed = sum(completed_counts.values()) if completed_counts else 0
    
    if total_completed > 0:
        logger.info(f"Already processed {total_completed} students with label distribution: {completed_counts}")
    
    # Calculate how many new samples we need
    if total_completed >= max_students:
        logger.info(f"Already have {total_completed} processed students >= max_students ({max_students}), no new sampling needed")
        # Return empty dataframe with same columns (will be combined with completed in caller)
        return test_data.iloc[0:0].copy()
    
    new_samples_needed = max_students - total_completed
    
    if len(available_test_data) <= new_samples_needed:
        logger.info(f"Available test data ({len(available_test_data)}) <= new samples needed ({new_samples_needed}), using all available")
        # Return all available data (excluding completed)
        return available_test_data.copy()
    
    # Get unique labels from original test data (to ensure we consider all possible labels)
    unique_labels = sorted(test_data[label_column].unique())
    if not unique_labels:
        logger.warning("No labels found in test data")
        return test_data.iloc[0:0].copy()
    
    # Calculate target distribution for overall (completed + new) to be balanced
    target_per_label = max_students // len(unique_labels)
    remainder = max_students % len(unique_labels)
    
    # Calculate how many new samples needed per label (compensating for already completed)
    selected_indices = []
    
    logger.info(f"Available test set (excluding {len(completed_indices_int)} completed) label distribution: {available_test_data[label_column].value_counts().to_dict()}")
    
    for i, label in enumerate(unique_labels):
        # Target total for this label (including completed)
        target_total = target_per_label + (1 if i < remainder else 0)
        
        # Already completed for this label
        already_have = completed_counts.get(label, 0)
        
        # How many more we need to sample for this label
        target_new_samples = max(0, target_total - already_have)
        
        # Get available samples with this label
        label_data = available_test_data[available_test_data[label_column] == label]
        
        # If insufficient available samples, use all available
        n_samples = min(target_new_samples, len(label_data))
        
        if n_samples > 0:
            selected_label_indices = np.random.choice(
                label_data.index,
                size=n_samples,
                replace=False
            )
            selected_indices.extend(selected_label_indices)
            
            # Log sampling info for this label
            if n_samples < target_new_samples:
                logger.info(f"  Label {label}: target total={target_total}, already have={already_have}, need {target_new_samples} more, available {len(label_data)}, sampled {n_samples} (limited by availability)")
            else:
                logger.info(f"  Label {label}: target total={target_total}, already have={already_have}, sampled {n_samples} more")
        elif target_new_samples > 0:
            logger.info(f"  Label {label}: target total={target_total}, already have={already_have}, need {target_new_samples} more, but no available samples")
    
    # Create sampled dataframe (only newly sampled rows)
    sampled_data = test_data.loc[selected_indices].copy() if selected_indices else test_data.iloc[0:0].copy()
    
    # Calculate combined distribution (completed + newly sampled)
    combined_counts = completed_counts.copy()
    if len(sampled_data) > 0:
        new_label_counts = sampled_data[label_column].value_counts().to_dict()
        for label, count in new_label_counts.items():
            combined_counts[label] = combined_counts.get(label, 0) + count
    
    total_combined = sum(combined_counts.values())
    
    # Log the distributions
    if len(sampled_data) > 0:
        new_label_counts = sampled_data[label_column].value_counts()
        logger.info(f"Newly sampled {len(sampled_data)} students with label distribution: {new_label_counts.to_dict()}")
    
    if total_combined > 0:
        logger.info(f"Combined (completed + newly sampled) total: {total_combined} students with distribution: {combined_counts}")
        combined_props = {label: count / total_combined for label, count in combined_counts.items()}
        logger.info(f"Combined label proportions: {combined_props}")
    
    return sampled_data


def run_experiment(llm_wrapper,
                   mode: str = "few_shot",
                   module: str = "BBB",
                   presentation: str = "2014J",
                   assessment_name: str = "TMA 1",
                   days_to_cutoff: int = 0,
                   features: list = None,
                   behavior_converter=None,
                   num_few_shot: int = 5,
                   max_students: Optional[int] = None,
                   selection_strategy: str = "balanced",
                   random_state: int = 42,
                   result_file: Optional[str] = None,
                   save_interval: int = 10,
                   num_workers: int = 1):
    """
    Run experiment (zero-shot or few-shot mode)
    
    Args:
        llm_wrapper: LLM wrapper instance
        mode: "zero_shot" or "few_shot"
        module: Module code
        presentation: Course presentation
        assessment_name: Assessment name
        days_to_cutoff: Days to cutoff
        features: Feature list
        behavior_converter: BehaviorToTextConverter instance (optional)
        num_few_shot: Number of few-shot examples (only for few-shot mode)
        max_students: Maximum number of students to test (None = all)
        selection_strategy: Example selection strategy ("balanced", "diverse", "similar")
        random_state: Random seed
        result_file: Path to result file for resume support (None = no saving)
        save_interval: Save results every N students
        num_workers: Number of parallel workers for concurrent processing (1 = sequential)
        
    Returns:
        Experiment result dictionary
    """
    if features is None:
        features = [
            "demog",
            "vle_statistics",
            "vle_statistics_beforestart",
            "vle_day_activity_type_flags",
            "vle_day_activity_type",
            "vle_day",
            "vle_day_flags",
            "reg_statistics",
        ]
    
    logger.info("=" * 60)
    logger.info(f"{mode.upper().replace('_', '-')} Mode Experiment")
    logger.info("=" * 60)
    logger.info(f"Task: {module} / {presentation} / {assessment_name} / days={days_to_cutoff}")
    if mode == "few_shot":
        logger.info(f"Few-shot examples: {num_few_shot}")
    if max_students:
        logger.info(f"Max students to test: {max_students}")
    logger.info("=" * 60)
    
    # 1. Load data (ensure test set consistency with paper)
    data = load_oulad_data(module, presentation, assessment_name, days_to_cutoff, features)
    train_data = data['train_data']
    test_data = data['test_data']
    y_test = data['y_test']
    
    # 1.5. Initialize result storage early to check for completed data (if result_file provided)
    storage = None
    completed_indices = set()
    completed_label_dist = {}
    if result_file:
        storage = ResultStorage(result_file)
        completed_indices = storage.get_completed_indices()
        if completed_indices:
            completed_label_dist = storage.get_completed_label_distribution()
            logger.info(f"Found {len(completed_indices)} already processed students with label distribution: {completed_label_dist}")
    
    # 1.6. Create behavior converter if not provided
    if behavior_converter is None:
        logger.info("Creating BehaviorToTextConverter...")
        behavior_converter = BehaviorToTextConverter(
            include_peer_context=True,
            days_to_cutoff=days_to_cutoff
        )
        # Set cohort statistics from training data
        behavior_converter.set_cohort_statistics(train_data)
        logger.info("✓ BehaviorToTextConverter initialized with cohort statistics")
    
    # 1.7. Limit test set size if specified (using balanced sampling, considering completed data)
    if max_students and len(test_data) > max_students:
        logger.info(f"Balanced sampling test set from {len(test_data)} to {max_students} students (considering {len(completed_indices)} already processed)")
        # Note: This will exclude already completed indices and sample new ones to maintain balance
        new_samples = balanced_sample_test_data(
            test_data=test_data,
            label_column='submitted',
            max_students=max_students,
            random_state=random_state,
            completed_indices=completed_indices,
            completed_label_dist=completed_label_dist
        )
        
        # Combine completed indices and newly sampled indices
        completed_indices_int = {int(idx) for idx in completed_indices if idx.isdigit()}
        if len(new_samples) > 0:
            new_indices = set(new_samples.index)
            all_indices = completed_indices_int | new_indices
            test_data = test_data.loc[sorted(all_indices)].copy()
        else:
            # Only use completed indices if no new samples needed
            test_data = test_data.loc[sorted(completed_indices_int)].copy()
        
        y_test = test_data['submitted'].values
    
    # 2. Prepare few-shot examples if needed
    few_shot_examples = []
    if mode == "few_shot":
        logger.info(f"Balanced sampling {num_few_shot} few-shot examples from training set ({len(train_data)} samples)...")
        selector = FewShotExampleSelector(
            strategy=selection_strategy,
            num_examples=num_few_shot
        )
        
        few_shot_examples = selector.select_from_train_dataframe(
            train_df=train_data,
            label_column='submitted',
            behavior_converter=behavior_converter,
            random_state=random_state
        )
        
        logger.info(f"Successfully sampled {len(few_shot_examples)} few-shot examples")
        logger.info(f"Example label distribution: {pd.Series([e['label'] for e in few_shot_examples]).value_counts().to_dict()}")
    
    # 3. Initialize UnifiedAgent
    agent = UnifiedAgent(
        llm_wrapper=llm_wrapper,
        behavior_converter=behavior_converter,
        mode=mode,
        num_few_shot_examples=num_few_shot
    )
    
    # 4. Set few-shot examples if needed
    if mode == "few_shot" and len(few_shot_examples) > 0:
        agent.set_few_shot_examples(few_shot_examples)
    
    # 5. Save metadata if storage is initialized
    if storage:
        if completed_indices:
            logger.info(f"Resuming from checkpoint: {len(completed_indices)}/{len(test_data)} students already processed")
        
        # Save metadata
        storage.save_metadata({
            'mode': mode,
            'module': module,
            'presentation': presentation,
            'assessment_name': assessment_name,
            'days_to_cutoff': days_to_cutoff,
            'num_few_shot': num_few_shot if mode == "few_shot" else 0,
            'selection_strategy': selection_strategy,
            'random_state': random_state,
            'total_students': len(test_data)
        })
    
    # 6. Predict on test set
    logger.info(f"Starting prediction on test set (total {len(test_data)} samples)...")
    logger.info(f"Using {num_workers} parallel workers")
    
    # Initialize predictions and true_labels with None to maintain order
    predictions = [None] * len(test_data)
    true_labels = [None] * len(test_data)
    
    # Load existing results if resuming
    if storage and completed_indices:
        for idx_str in completed_indices:
            idx = int(idx_str)
            if idx < len(test_data):
                student_result = storage.results['student_results'][idx_str]
                predictions[idx] = student_result['prediction']
                true_labels[idx] = student_result['true_label']
    
    # Define student processing function
    def process_student(idx_row_tuple):
        """Process a single student (for parallel execution)"""
        idx, row = idx_row_tuple
        
        # Skip if already processed
        if str(idx) in completed_indices:
            logger.debug(f"Skipping student {idx} (already processed)")
            return idx, None, None
        
        # Generate student behavior description
        student_id = str(row.get('id_student', f'student_{idx}'))
        if behavior_converter is not None:
            try:
                narrative = behavior_converter.convert_to_text(row)
            except Exception as e:
                logger.warning(f"Failed to convert row {idx}: {e}")
                narrative = f"Student ID: {student_id}"
        else:
            # Simple fallback: use student ID
            narrative = f"Student ID: {student_id}"
        
        # Predict with retry logic (handled in Client)
        try:
            result = agent.predict(student_narrative=narrative)
            risk_prob = agent.get_risk_probability(result)
            
            # Extract prompt and response from result
            prompt = result.get('prompt', None)
            response = result.get('raw_response', None)
            
            return idx, risk_prob, int(row['submitted']), student_id, prompt, response
            
        except Exception as e:
            logger.error(f"Error predicting for student {idx} (after all retries): {e}")
            # Return None to indicate failure
            return idx, None, None, student_id, None, None
    
    # Process students (sequential or parallel based on num_workers)
    if num_workers == 1:
        # Sequential processing
        logger.info("Using sequential processing")
        for idx, (_, row) in enumerate(test_data.iterrows()):
            result = process_student((idx, row))
            if result[1] is not None:  # risk_prob is not None
                idx, risk_prob, true_label, student_id, prompt, response = result
                predictions[idx] = risk_prob
                true_labels[idx] = true_label
                
                if storage:
                    storage.save_student_result(
                        student_id, idx, risk_prob, true_label,
                        prompt=prompt, response=response
                    )
                
                completed_count = sum(1 for p in predictions if p is not None)
                if (idx + 1) % save_interval == 0:
                    logger.info(f"Processed {idx + 1}/{len(test_data)} samples (completed: {completed_count})")
    else:
        # Parallel processing using ThreadPoolExecutor
        logger.info(f"Using parallel processing with {num_workers} workers")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = []
            for idx, (_, row) in enumerate(test_data.iterrows()):
                if str(idx) not in completed_indices:
                    future = executor.submit(process_student, (idx, row))
                    futures.append(future)
            
            # Process results as they complete
            completed_count = len(completed_indices)
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result[1] is not None:  # risk_prob is not None
                        idx, risk_prob, true_label, student_id, prompt, response = result
                        predictions[idx] = risk_prob
                        true_labels[idx] = true_label
                        
                        if storage:
                            storage.save_student_result(
                                student_id, idx, risk_prob, true_label,
                                prompt=prompt, response=response
                            )
                        
                        completed_count += 1
                        if completed_count % save_interval == 0:
                            logger.info(f"Completed {completed_count}/{len(test_data)} samples")
                except Exception as e:
                    logger.error(f"Task failed with exception: {e}")
                    # Continue processing other tasks
    
    # 7. Final save if storage is enabled
    if storage:
        storage.save()
        logger.info(f"Final results saved to {storage.result_file}")
    
    # 8. Filter out None values (unprocessed students)
    valid_predictions = [p for p in predictions if p is not None]
    valid_true_labels = [l for l in true_labels if l is not None]
    
    if len(valid_predictions) == 0:
        logger.error("No predictions available. Cannot calculate metrics.")
        return {
            'predictions': [],
            'true_labels': [],
            'pr_auc': 0.0,
            'roc_auc': 0.0,
            'accuracy': 0.0,
            'few_shot_examples': few_shot_examples if mode == "few_shot" else None,
            'agent': agent,
            'mode': mode,
            'num_tested': 0
        }
    
    logger.info(f"Calculating metrics for {len(valid_predictions)} completed predictions")
    
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score
    
    precision, recall, thresholds = precision_recall_curve(valid_true_labels, valid_predictions)
    pr_auc = auc(recall, precision)
    
    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(valid_true_labels, valid_predictions)
    except:
        roc_auc = 0.0
    
    # Calculate accuracy at threshold 0.5
    binary_predictions = [1 if p >= 0.5 else 0 for p in valid_predictions]
    accuracy = accuracy_score(valid_true_labels, binary_predictions)
    
    logger.info("=" * 60)
    logger.info("Experiment Results:")
    logger.info(f"Total predictions: {len(valid_predictions)}/{len(test_data)}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("=" * 60)
    
    return {
        'predictions': valid_predictions,
        'true_labels': valid_true_labels,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'few_shot_examples': few_shot_examples if mode == "few_shot" else None,
        'agent': agent,
        'mode': mode,
        'num_tested': len(valid_predictions),
        'num_total': len(test_data),
        'result_file': result_file
    }


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run Unified Agent experiments with LLM (OpenRouter or Local)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # LLM provider settings
    parser.add_argument("--provider", type=str, choices=["openrouter", "local", "multi_local"], 
                       default="local",
                       help="LLM provider: 'openrouter', 'local' (single server), or 'multi_local' (multiple servers)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenRouter API key (only for OpenRouter, or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-70b-instruct",
                       help="Model name (only for OpenRouter)")
    parser.add_argument("--server_url", type=str, default="http://localhost:8004",
                       help="Local Llama server URL (only for local provider)")
    parser.add_argument("--base_port", type=int, default=8000,
                       help="Base port for multi-server setup (only for multi_local)")
    parser.add_argument("--num_servers", type=int, default=4,
                       help="Number of servers for multi-server setup (only for multi_local)")
    parser.add_argument("--load_balance_strategy", type=str, default="round_robin",
                       choices=["round_robin", "random"],
                       help="Load balancing strategy for multi-server setup")
    
    # Experiment mode
    parser.add_argument("--mode", type=str, choices=["zero_shot", "few_shot"], default="few_shot",
                       help="Experiment mode: zero_shot or few_shot")
    parser.add_argument("--num_few_shot", type=int, default=5,
                       help="Number of few-shot examples (only for few-shot mode)")
    
    # Task selection
    parser.add_argument("--module", type=str, default="BBB",
                       help="Module code (e.g., BBB, AAA, CCC)")
    parser.add_argument("--presentation", type=str, default="2014J",
                       help="Course presentation (e.g., 2014J, 2013J, 2014B)")
    parser.add_argument("--assessment_name", type=str, default="TMA 1",
                       help="Assessment name (e.g., 'TMA 1', 'TMA 2', 'CMA 1')")
    parser.add_argument("--days_to_cutoff", type=int, default=0,
                       help="Days to cutoff (0-11)")
    
    # Student control
    parser.add_argument("--max_students", type=int, default=None,
                       help="Maximum number of students to test (None = all)")
    
    # Other settings
    parser.add_argument("--selection_strategy", type=str, default="balanced",
                       choices=["balanced", "diverse", "similar"],
                       help="Few-shot example selection strategy")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Result storage settings
    parser.add_argument("--result_file", type=str, default=None,
                       help="Path to result file for resume support (JSON format)")
    parser.add_argument("--save_interval", type=int, default=1,
                       help="Save results every N students")
    
    # API retry settings
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of API retries")
    parser.add_argument("--retry_delay", type=int, default=60,
                       help="Delay in seconds between API retries")
    
    # Concurrency settings
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of parallel workers for concurrent processing (1=sequential, >1=parallel)")
    
    args = parser.parse_args()
    
    # Initialize LLM client
    try:
        logger.info(f"Initializing {args.provider} LLM client...")
        llm_client = create_llm_client(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            server_url=args.server_url,
            base_port=args.base_port,
            num_servers=args.num_servers,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            load_balance_strategy=args.load_balance_strategy
        )
        
        if args.provider == "openrouter":
            logger.info(f"✅ Connected to OpenRouter API, model: {args.model}")
        elif args.provider == "local":
            logger.info(f"✅ Connected to local Llama server at {args.server_url}")
        elif args.provider == "multi_local":
            logger.info(f"✅ Connected to {args.num_servers} local Llama servers (base port: {args.base_port})")
            logger.info(f"   Load balancing: {args.load_balance_strategy}")
        
        logger.info(f"   Retry settings: max_retries={args.max_retries}, delay={args.retry_delay}s")
        logger.info(f"   Concurrency: {args.num_workers} parallel workers")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize {args.provider} LLM client: {e}")
        if args.provider == "openrouter":
            logger.error("Please check:")
            logger.error("1. Is OPENROUTER_API_KEY set? (or use --api_key)")
            logger.error("2. Is the model name correct?")
            logger.error("3. Is the network connection normal?")
        elif args.provider == "multi_local":
            logger.error("Please check:")
            logger.error("1. Are the local Llama servers running? (bash start_multi_gpu_servers.sh)")
            logger.error("2. Is the base_port correct?")
            logger.error("3. Are the servers accessible from this machine?")
        else:
            logger.error("Please check:")
            logger.error("1. Is the local Llama server running? (bash start_llama_server.sh)")
            logger.error("2. Is the server_url correct?")
            logger.error("3. Can you access the server from this machine?")
        return 1
    
    # Generate default result file if not provided
    if args.result_file is None:
        # Generate result file name based on experiment parameters
        result_dir = Path("results")
        result_dir.mkdir(exist_ok=True)
        result_filename = f"results_{args.module}_{args.presentation}_{args.assessment_name.replace(' ', '_')}_days{args.days_to_cutoff}_{args.mode}.json"
        args.result_file = str(result_dir / result_filename)
        logger.info(f"Using default result file: {args.result_file}")
    
    # Run experiment
    try:
        result = run_experiment(
            llm_wrapper=llm_client,
            mode=args.mode,
            module=args.module,
            presentation=args.presentation,
            assessment_name=args.assessment_name,
            days_to_cutoff=args.days_to_cutoff,
            behavior_converter=None,  # Can be added later if needed
            num_few_shot=args.num_few_shot,
            max_students=args.max_students,
            selection_strategy=args.selection_strategy,
            random_state=args.random_state,
            result_file=args.result_file,
            save_interval=args.save_interval,
            num_workers=args.num_workers
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Mode: {result['mode']}")
        print(f"Task: {args.module} / {args.presentation} / {args.assessment_name} / days={args.days_to_cutoff}")
        print(f"Students tested: {result['num_tested']}")
        if result['mode'] == "few_shot":
            print(f"Few-shot examples: {args.num_few_shot}")
        print(f"\nResults:")
        print(f"  PR-AUC:  {result['pr_auc']:.4f}")
        print(f"  ROC-AUC: {result['roc_auc']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        if result.get('result_file'):
            print(f"\nResults saved to: {result['result_file']}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
