"""
Few-shot prompt experiments
Directly predict target values from student data using natural language prompts
"""
import pandas as pd
import numpy as np
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add baseline module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'baseline'))
from data_preprocessing import DataPreprocessor

# Add llm_data_generation to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'llm_data_generation'))
from llm_client import OpenRouterClient

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))
from prompt_predictor import PromptPredictor


def save_result_immediately(
    result: dict,
    results_dir: Path,
    timestamp: str,
    all_results: list,
    all_detailed_results: dict
):
    """
    Persist a single task result immediately and update aggregate artifacts.
    
    Args:
        result: Task result dictionary
        results_dir: Directory for result files
        timestamp: Timestamp string
        all_results: List of all results (will be updated)
        all_detailed_results: Dict of all detailed results (will be updated)
    """
    all_results.append(result)
    
    setup_task_key = f"{result['setup']}_{result['task']}"
    all_detailed_results[setup_task_key] = {
        'setup': result['setup'],
        'task': result['task'],
        'model_name': result['model_name'],
        'n_examples': result['n_examples'],
        'random_state': result['random_state'],
        'temperature': result['temperature'],
        'score': result['score'],
        'samples': result['detailed_results']
    }
    
    results_file = results_dir / f"results_prompt_{timestamp}.json"
    summary_results = []
    for r in all_results:
        summary_result = {k: v for k, v in r.items() if k != 'detailed_results'}
        summary_results.append(summary_result)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    detailed_file = results_dir / f"results_prompt_detailed_{timestamp}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(all_detailed_results, f, indent=2, ensure_ascii=False)
    
    results_for_summary = []
    for r in all_results:
        results_for_summary.append({
            'Model': r['model_name'],
            'Setup': f"Setup {r['setup']}",
            'Task': r['task'],
            'Score': f"{r['score']:.4f}",
            'Few-shot examples': r['n_examples'],
            'Random state': r['random_state'],
            'Temperature': r['temperature']
        })
    
    summary_file = results_dir / f"results_prompt_summary_{timestamp}.csv"
    summary_df = pd.DataFrame(results_for_summary)
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'=' * 80}")
    print(f"✅ Task finished: Setup {result['setup']} - {result['task']}")
    print(f"{'=' * 80}")
    print(f"Score: {result['score']:.4f}")
    print("Results saved:")
    print(f"  - Summary JSON: {results_file}")
    print(f"  - Detailed JSON: {detailed_file}")
    print(f"  - CSV summary: {summary_file}")
    print("\nProgress so far:")
    print(summary_df.to_string(index=False))


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint file if present."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'completed_tasks': [],
        'all_results': [],
        'all_detailed_results': {},
        'timestamp': None
    }


def save_checkpoint(
    checkpoint_file: Path,
    completed_tasks: list,
    all_results: list,
    all_detailed_results: dict,
    timestamp: str
):
    """Persist checkpoint data to disk."""
    checkpoint_data = {
        'completed_tasks': completed_tasks,
        'all_results': all_results,
        'all_detailed_results': all_detailed_results,
        'timestamp': timestamp
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)


def run_prompt_experiments(
    data_path: str = '../data/student-por.csv',
    model: str = 'meta-llama/llama-3.3-70b-instruct',
    n_examples: int = 5,
    random_state: int = 42,
    temperature: float = 0, # 0 is the most deterministic
    resume: bool = False,
    checkpoint_file: str = None,
    use_feature_selection: bool = True,
    feature_selection_model: str = 'rf',
    n_top_features: int = 10,
    # Self-Consistency parameters
    use_self_consistency: bool = True,
    n_consistency_samples: int = 5,
    consistency_temperature: float = 0.7,
    # COT output parameter
    output_cot: bool = False
):
    """
    Run few-shot prompt experiments with immediate output and resume support
    
    Uses the same data split as baseline (single split, not cross-validation):
    - Uses KFold(n_splits=10, shuffle=True, random_state=42) first fold as test set
    - Train set is used for few-shot examples
    - Test set is used for prediction
    
    Args:
        data_path: Path to student data CSV
        model: LLM model name
        n_examples: Number of few-shot examples (from train set)
        random_state: Random seed (default: 42, same as baseline first run)
        temperature: LLM temperature
        resume: Whether to resume from checkpoint
        checkpoint_file: Path to checkpoint file (default: auto-generated)
    """
    print("=" * 80)
    print("Few-shot prompt experiments")
    print("=" * 80)
    
    # Create results directory
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp and checkpoint file path
    if checkpoint_file is None:
        # Try to find existing checkpoint or create new one
        checkpoint_file = results_dir / "checkpoint_prompt.json"
    else:
        checkpoint_file = Path(checkpoint_file)
    
    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        print(f"\n📂 Resuming from checkpoint: {checkpoint_file}")
        checkpoint = load_checkpoint(checkpoint_file)
        completed_tasks = set(checkpoint['completed_tasks'])
        all_results = checkpoint['all_results']
        all_detailed_results = checkpoint['all_detailed_results']
        timestamp = checkpoint.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"Completed tasks: {sorted(completed_tasks)}")
        print("Skipping completed tasks and continuing pending ones")
    else:
        if resume:
            print(f"\n⚠️  Checkpoint not found: {checkpoint_file}")
            print("Starting all tasks from scratch")
        completed_tasks = set()
        all_results = []
        all_detailed_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize LLM client
    print(f"\nInitializing LLM client: {model}")
    try:
        llm_client = OpenRouterClient(model=model)
    except Exception as e:
        print(f"Error: failed to initialize LLM client: {e}")
        print("Ensure OPENROUTER_API_KEY environment variable is set")
        return
    
    # Load and preprocess data
    print(f"\nLoading data: {data_path}")
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    
    # Define all tasks
    all_task_configs = [
        {'setup': 'A', 'task': 'classification', 'strategy': 'balanced'},
        {'setup': 'A', 'task': 'regression', 'strategy': 'diverse'},
        {'setup': 'C', 'task': 'classification', 'strategy': 'balanced'},
        {'setup': 'C', 'task': 'regression', 'strategy': 'diverse'},
    ]
    
    # Run experiments for each task
    for task_config in all_task_configs:
        setup = task_config['setup']
        task = task_config['task']
        strategy = task_config['strategy']
        task_key = f"{setup}_{task}"
        
        # Check if task is already completed
        if task_key in completed_tasks:
            print(f"\n{'=' * 80}")
            print(f"⏭️  Skipping completed task: Setup {setup} - {task}")
            print(f"{'=' * 80}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Starting task: Setup {setup} - {task}")
        print(f"{'=' * 80}")
        
        # Get data for this setup
        X, y_binary, y_regression, feature_names = preprocessor.get_setup_data(setup)
        
        # Get original (non-encoded) data for prompt generation
        if setup == 'A':
            X_df = preprocessor.data.drop(columns=['G3']).copy()
        else:  # setup == 'C'
            X_df = preprocessor.data.drop(columns=['G1', 'G2', 'G3']).copy()
        
        print(f"\nData shape: {X_df.shape}")
        print(f"Feature count: {len(feature_names)}")
        
        # Create predictor with all features
        predictor = PromptPredictor(
            llm_client=llm_client,
            n_examples=n_examples,
            temperature=temperature,
            use_cot=True,
            output_cot=output_cot,
            example_strategy='hybrid',
            use_feature_selection=use_feature_selection,
            feature_selection_model=feature_selection_model,
            n_top_features=n_top_features,
            use_self_consistency=use_self_consistency,
            n_consistency_samples=n_consistency_samples,
            consistency_temperature=consistency_temperature,
            use_dynamic_examples=False,
            use_adaptive_temperature=False,
            use_error_correction=False
        )
        # Run evaluation
        if task == 'classification':
            y_target = y_binary
            task_name = "Classification - Pass/Fail"
        else:  # regression
            y_target = y_regression
            task_name = "Regression - G3 Grade"
        
        print(f"\n{'-' * 80}")
        print(f"{task_name}")
        print(f"{'-' * 80}")
        
        try:
            score, y_test, y_pred, detailed_results = predictor.evaluate(
                X=X_df,  # Original features for prompt construction
                y=y_target,
                task=task,
                setup=setup,
                random_state=random_state,
                strategy=strategy,
                X_encoded=X  # Encoded features for feature selection model
            )
            
            # Create result dict
            model_name = f'LLM-Prompt (Few-shot, {n_examples} examples)'
            if use_self_consistency:
                model_name += f' [Self-Consistency: {n_consistency_samples} samples]'
            if output_cot:
                model_name += ' [COT Output: JSON]'
            
            result = {
                'model_name': model_name,
                'setup': setup,
                'task': task,
                'score': float(score),
                'n_examples': n_examples,
                'random_state': random_state,
                'temperature': temperature,
                'use_self_consistency': use_self_consistency,
                'n_consistency_samples': n_consistency_samples if use_self_consistency else None,
                'consistency_temperature': consistency_temperature if use_self_consistency else None,
                'output_cot': output_cot,
                'detailed_results': detailed_results
            }
            
            # Save immediately
            save_result_immediately(
                result=result,
                results_dir=results_dir,
                timestamp=timestamp,
                all_results=all_results,
                all_detailed_results=all_detailed_results
            )
            
            # Mark task as completed and save checkpoint
            completed_tasks.add(task_key)
            save_checkpoint(
                checkpoint_file=checkpoint_file,
                completed_tasks=list(completed_tasks),
                all_results=all_results,
                all_detailed_results=all_detailed_results,
                timestamp=timestamp
            )
            
        except Exception as e:
            print(f"\n❌ Task failed: Setup {setup} - {task}")
            print(f"Error: {e}")
            print("Checkpoint saved; you can resume later with --resume")
            import traceback
            traceback.print_exc()
            # Save checkpoint even on error
            save_checkpoint(
                checkpoint_file=checkpoint_file,
                completed_tasks=list(completed_tasks),
                all_results=all_results,
                all_detailed_results=all_detailed_results,
                timestamp=timestamp
            )
            raise
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("🎉 All tasks completed!")
    print(f"{'=' * 80}")
    
    if all_results:
        results_for_summary = []
        for r in all_results:
            results_for_summary.append({
                'Model': r['model_name'],
                'Setup': f"Setup {r['setup']}",
                'Task': r['task'],
                'Score': f"{r['score']:.4f}",
                'Few-shot examples': r['n_examples'],
                'Random state': r['random_state'],
                'Temperature': r['temperature']
            })
        
        summary_df = pd.DataFrame(results_for_summary)
        print("\nFinal result summary:")
        print(summary_df.to_string(index=False))
        
        # Final save
        results_file = results_dir / f"results_prompt_{timestamp}.json"
        detailed_file = results_dir / f"results_prompt_detailed_{timestamp}.json"
        summary_file = results_dir / f"results_prompt_summary_{timestamp}.csv"
        
        print(f"\nFinal result files:")
        print(f"  - Summary JSON: {results_file}")
        print(f"  - Detailed JSON: {detailed_file}")
        print(f"  - CSV summary: {summary_file}")
    
    # Clean up checkpoint if all tasks completed
    if len(completed_tasks) == len(all_task_configs):
        if checkpoint_file.exists():
            print(f"\n🧹 Removing checkpoint file: {checkpoint_file}")
            checkpoint_file.unlink()
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run few-shot prompt experiments')
    parser.add_argument(
        '--data_path',
        type=str,
        default='../data/student-por.csv',
        help='Path to student data CSV'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/llama-3.3-70b-instruct',
        help='LLM model name'
    )
    parser.add_argument(
        '--n_examples',
        type=int,
        default=5,
        help='Number of few-shot examples (default: 5, sampled from train)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed (default: 42, matches baseline first run)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='LLM temperature (default: 0.3)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (skip completed tasks)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (default: ../results/checkpoint_prompt.json)'
    )
    parser.add_argument(
        '--use_feature_selection',
        action='store_true',
        default=True,
        help='Enable feature selection based on importance (default: True)'
    )
    parser.add_argument(
        '--feature_selection_model',
        type=str,
        default='rf',
        choices=['rf', 'xgb'],
        help='Model for feature selection importance: rf or xgb (default: rf)'
    )
    parser.add_argument(
        '--n_top_features',
        type=int,
        default=10,
        help='Select top-N important features for prompts (default: 10)'
    )
    parser.add_argument(
        '--use_self_consistency',
        action='store_true',
        default=True,
        help='Enable self-consistency (multiple samples + voting) (default: True)'
    )
    parser.add_argument(
        '--no_self_consistency',
        action='store_false',
        dest='use_self_consistency',
        help='Disable self-consistency'
    )
    parser.add_argument(
        '--n_consistency_samples',
        type=int,
        default=5,
        help='Number of self-consistency samples (default: 5, recommend 3-7)'
    )
    parser.add_argument(
        '--consistency_temperature',
        type=float,
        default=0.7,
        help='Self-consistency sampling temperature (default: 0.7, recommend 0.5-0.8)'
    )
    parser.add_argument(
        '--output_cot',
        action='store_true',
        default=False,
        help='Enable Chain-of-Thought output in JSON format (default: False)'
    )
    parser.add_argument(
        '--no_output_cot',
        action='store_false',
        dest='output_cot',
        help='Disable Chain-of-Thought output'
    )
    
    args = parser.parse_args()
    
    run_prompt_experiments(
        data_path=args.data_path,
        model=args.model,
        n_examples=args.n_examples,
        random_state=args.random_state,
        temperature=args.temperature,
        resume=args.resume,
        checkpoint_file=args.checkpoint,
        use_feature_selection=args.use_feature_selection,
        feature_selection_model=args.feature_selection_model,
        n_top_features=args.n_top_features,
        use_self_consistency=args.use_self_consistency,
        n_consistency_samples=args.n_consistency_samples,
        consistency_temperature=args.consistency_temperature,
        output_cot=args.output_cot
    )

