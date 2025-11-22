"""
Step 1: Prepare zero-shot prompts from exercise-grouped data.

This script:
1. Reads exercise-grouped data from llm_mlp/data/{track}_{split}_exercise.jsonl
2. Creates exercise-level zero-shot prompts
3. Saves to llm_zeroshot/data/{track}_{split}_zeroshot.jsonl
"""

import argparse
import json
import os
import sys
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm_zeroshot.utils import create_zeroshot_prompt


def prepare_zeroshot_data(
    input_path: str,
    output_path: str,
    limit: int = None
):
    """
    Prepare zero-shot prompts from exercise-grouped data.
    
    Args:
        input_path: Path to input exercise.jsonl file
        output_path: Path to output zeroshot.jsonl file
        limit: Optional limit on number of exercises to process
    """
    print(f"Reading exercise data from: {input_path}")
    
    exercises = []
    with open(input_path, 'r') as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            
            exercise_data = json.loads(line.strip())
            exercises.append(exercise_data)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Loaded {idx + 1} exercises...")
    
    print(f"Loaded {len(exercises)} exercises")
    
    # Prepare zero-shot prompts
    print(f"\nGenerating zero-shot prompts...")
    zeroshot_data = []
    
    for idx, exercise_data in enumerate(exercises):
        # Create zero-shot prompt
        prompt = create_zeroshot_prompt(exercise_data)
        
        # Create new data record
        zeroshot_record = {
            'exercise_id': exercise_data['exercise_id'],
            'instance_ids': exercise_data['instance_ids'],
            'prompt': prompt,
            'token_labels': exercise_data.get('token_labels', None),
            'exercise_label': exercise_data.get('exercise_label', None),
            'num_tokens': exercise_data['num_tokens'],
            'user_id': exercise_data.get('user_id', None)
        }
        
        zeroshot_data.append(zeroshot_record)
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} exercises...")
    
    # Save to output file
    print(f"\nSaving to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for record in zeroshot_data:
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved {len(zeroshot_data)} zero-shot prompts")
    
    # Print statistics
    total_tokens = sum(r['num_tokens'] for r in zeroshot_data)
    avg_tokens = total_tokens / len(zeroshot_data) if zeroshot_data else 0
    
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Total exercises: {len(zeroshot_data)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per exercise: {avg_tokens:.2f}")
    
    if any(r['token_labels'] is not None for r in zeroshot_data):
        labeled_exercises = [r for r in zeroshot_data if r['exercise_label'] is not None]
        if labeled_exercises:
            num_correct = sum(1 for r in labeled_exercises if r['exercise_label'] == 1)
            print(f"Correct exercises: {num_correct} / {len(labeled_exercises)} ({num_correct/len(labeled_exercises)*100:.1f}%)")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare zero-shot prompts from exercise-grouped data'
    )
    parser.add_argument(
        '--track',
        type=str,
        default='en_es',
        choices=['en_es', 'es_en', 'fr_en'],
        help='Dataset track'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        choices=['dev', 'test'],
        help='Data split to prepare'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='llm_mlp/data',
        help='Directory containing exercise-grouped data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='llm_zeroshot/data',
        help='Output directory for zero-shot prompts'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of exercises to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Construct file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    input_path = os.path.join(project_root, args.data_dir, f'{args.track}_{args.split}_exercise.jsonl')
    output_path = os.path.join(project_root, args.output_dir, f'{args.track}_{args.split}_zeroshot.jsonl')
    
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please run step1_prepare_data.py first to generate exercise-grouped data."
        )
    
    print("="*80)
    print("STEP 1: ZERO-SHOT PROMPT PREPARATION")
    print("="*80)
    print(f"Track: {args.track}")
    print(f"Split: {args.split}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if args.limit:
        print(f"Limit: {args.limit} exercises")
    print("="*80)
    print()
    
    # Prepare data
    prepare_zeroshot_data(input_path, output_path, args.limit)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
