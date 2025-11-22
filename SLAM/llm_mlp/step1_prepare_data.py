"""
Step 1: Prepare exercise-grouped data with user history context.

This script:
1. Loads train data to build user history
2. Loads dev and test data
3. Groups tokens by exercise (exercise_id = instance_id[:10])
4. Creates contextualized prompts for each token
5. Aggregates exercise-level labels (all-correct policy)
6. Outputs exercise-grouped dev.jsonl and test.jsonl
"""

import argparse
import json
import os
import sys
from typing import Dict, List
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm_mlp.data_loader import load_data, InstanceData
from llm_mlp.utils import UserHistory, create_exercise_aware_prompt, build_user_history_from_train


def group_by_exercise(
    instances: List[InstanceData],
    labels: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Group instances by exercise_id.
    
    Args:
        instances: List of instances
        labels: Dictionary of instance_id -> label
    
    Returns:
        Dictionary of exercise_id -> {
            'instances': List[InstanceData],
            'labels': List[float],
            'exercise_label': float (1 if all correct, 0 otherwise)
        }
    """
    exercises = defaultdict(lambda: {'instances': [], 'labels': []})
    
    for instance in instances:
        exercise_id = instance.exercise_id  # First 10 chars
        exercises[exercise_id]['instances'].append(instance)
        
        if instance.instance_id in labels:
            label = labels[instance.instance_id]
            exercises[exercise_id]['labels'].append(label)
    
    # Compute exercise-level label (all-correct policy)
    for exercise_id, data in exercises.items():
        if data['labels']:
            # Exercise is correct (1) only if ALL tokens are correct
            data['exercise_label'] = 1.0 if all(l > 0.5 for l in data['labels']) else 0.0
        else:
            data['exercise_label'] = None
    
    return exercises


def prepare_exercise_data(
    exercises: Dict[str, Dict],
    user_history: UserHistory,
    output_path: str,
    split_name: str
):
    """
    Prepare exercise-grouped data with user history context.
    
    Args:
        exercises: Dictionary from group_by_exercise()
        user_history: User learning history from train data
        output_path: Path to output JSONL file
        split_name: Name of the split (for logging)
    """
    print(f"\nPreparing {split_name} data (exercise-level)...")
    print(f"  Total exercises: {len(exercises)}")
    
    total_tokens = sum(len(ex['instances']) for ex in exercises.values())
    print(f"  Total tokens: {total_tokens}")
    
    # Count exercise labels
    if any(ex['exercise_label'] is not None for ex in exercises.values()):
        labeled_exercises = [ex for ex in exercises.values() if ex['exercise_label'] is not None]
        num_correct = sum(1 for ex in labeled_exercises if ex['exercise_label'] > 0.5)
        num_incorrect = len(labeled_exercises) - num_correct
        print(f"  Correct exercises: {num_correct} ({num_correct/len(labeled_exercises)*100:.1f}%)")
        print(f"  Incorrect exercises: {num_incorrect} ({num_incorrect/len(labeled_exercises)*100:.1f}%)")
    
    with open(output_path, 'w') as f:
        for idx, (exercise_id, ex_data) in enumerate(sorted(exercises.items())):
            # Create prompts for all tokens in the exercise (with full exercise context)
            prompts = []
            instance_ids = []
            
            all_instances = ex_data['instances']
            for token_idx, instance in enumerate(all_instances):
                # Use exercise-aware prompt that includes all tokens in the exercise
                prompt = create_exercise_aware_prompt(
                    instance=instance,
                    all_instances=all_instances,
                    current_idx=token_idx,
                    user_history=user_history
                )
                prompts.append(prompt)
                instance_ids.append(instance.instance_id)
            
            # Get user_id from first instance
            user_id = ex_data['instances'][0].user if ex_data['instances'] else None
            
            # Create JSON record
            record = {
                'exercise_id': exercise_id,
                'instance_ids': instance_ids,
                'prompts': prompts,
                'token_labels': ex_data['labels'],
                'exercise_label': int(ex_data['exercise_label']) if ex_data['exercise_label'] is not None else None,
                'user_id': user_id,
                'num_tokens': len(instance_ids)
            }
            
            f.write(json.dumps(record) + '\n')
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} exercises...")
    
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare SLAM data at exercise level with user history context'
    )
    parser.add_argument(
        '--track',
        type=str,
        default='en_es',
        choices=['en_es', 'es_en', 'fr_en'],
        help='Dataset track to use'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset',
        help='Directory containing the dataset files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='llm_mlp/data',
        help='Output directory for prepared data'
    )
    
    args = parser.parse_args()
    
    # Construct file paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', args.data_dir)
    train_file = os.path.join(data_dir, f'{args.track}.slam.20190204.train')
    dev_file = os.path.join(data_dir, f'{args.track}.slam.20190204.dev')
    test_file = os.path.join(data_dir, f'{args.track}.slam.20190204.test')
    
    # Check if files exist
    for filepath, name in [(train_file, 'train'), (dev_file, 'dev'), (test_file, 'test')]:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{name} file not found: {filepath}")
    
    print("="*80)
    print("STEP 1: EXERCISE-LEVEL DATA PREPARATION WITH USER HISTORY")
    print("="*80)
    print(f"Track: {args.track}")
    print(f"Data directory: {data_dir}")
    
    # Load train data to build user history
    print("\n[1/6] Loading train data to build user history...")
    train_instances, train_labels = load_data(train_file)
    print(f"  Loaded {len(train_instances)} train instances")
    
    print("\n[2/6] Building user history from train data...")
    user_history = build_user_history_from_train(train_instances, train_labels)
    num_users = len(user_history.user_stats)
    print(f"  Built history for {num_users} users")
    
    # Load dev data
    print("\n[3/6] Loading dev data...")
    dev_instances = load_data(dev_file)
    # Load dev labels from .key file
    dev_file_key = dev_file + '.key'
    dev_labels = {}
    if os.path.isfile(dev_file_key):
        with open(dev_file_key, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    dev_labels[parts[0]] = float(parts[1])
        print(f"  Loaded {len(dev_instances)} dev instances with labels from .key file")
    else:
        print(f"  Warning: No .key file found for dev, labels will be None")
    
    # Load test data
    print("\n[4/6] Loading test data...")
    test_instances = load_data(test_file)
    # Load test labels from .key file
    test_file_key = test_file + '.key'
    test_labels = {}
    if os.path.isfile(test_file_key):
        with open(test_file_key, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    test_labels[parts[0]] = float(parts[1])
        print(f"  Loaded {len(test_instances)} test instances with labels from .key file")
    else:
        print(f"  Loaded {len(test_instances)} test instances (no labels)")
        test_labels = None
    
    # Group by exercise
    print("\n[5/6] Grouping instances by exercise...")
    dev_exercises = group_by_exercise(dev_instances, dev_labels)
    test_exercises = group_by_exercise(test_instances, test_labels if test_labels else {})
    print(f"  Dev: {len(dev_exercises)} exercises")
    print(f"  Test: {len(test_exercises)} exercises")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare exercise-level data
    print("\n[6/6] Creating exercise-level JSONL files...")
    dev_output = os.path.join(output_dir, f'{args.track}_dev_exercise.jsonl')
    prepare_exercise_data(dev_exercises, user_history, dev_output, 'dev')
    
    test_output = os.path.join(output_dir, f'{args.track}_test_exercise.jsonl')
    prepare_exercise_data(test_exercises, user_history, test_output, 'test')
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Train instances (for history): {len(train_instances)}")
    print(f"Dev exercises: {len(dev_exercises)}")
    print(f"Test exercises: {len(test_exercises)}")
    print(f"Unique users in history: {num_users}")
    print(f"\nOutput files:")
    print(f"  - {dev_output}")
    print(f"  - {test_output}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
