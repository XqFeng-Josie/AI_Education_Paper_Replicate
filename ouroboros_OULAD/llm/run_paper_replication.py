"""
Run LLM multi-agent experiments on OULAD dataset

This script runs experiments matching the original Ouroboros paper's experimental setup
to enable direct comparison with traditional ML methods.

Default Paper Settings (matching traditional ML experiments):
- modules = ["BBB", "DDD", "EEE", "FFF"]
- presentations = ["2014J"]
- assessment = "TMA 1" (single assessment)
- max_days = 11 (days 0-11, total 12 time points)
- label_name = "submitted"

Usage:
  # Pilot test (small sample)
  python -m llm_experiments.experiments.run_paper_replication --pilot --n_students 10
  
  # Single configuration
  python -m llm_experiments.experiments.run_paper_replication --module BBB --presentation 2014J --day 0
  
  # Full paper replication
  python -m llm_experiments.experiments.run_paper_replication
"""

import os
import sys
import argparse
import logging
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.models.llm_wrapper import create_llm_wrapper
from llm.models.multi_agent_system import MultiAgentSystem
from llm.behavior_to_text import BehaviorToTextConverter
from llm.metrics import evaluate_predictions, print_evaluation_summary
from selflearner.problem_definition import ProblemDefinition
from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Paper Configuration
# ============================================================================

PAPER_CONFIG = {
    'modules': ["BBB", "DDD", "EEE", "FFF"],
    'presentations': ["2014J"],
    'assessment': "TMA 1",  # Single assessment, matching traditional ML experiments
    'max_days': 11,  # Days 0-11, matching traditional ML experiments
    'label_name': "submitted",
    'features': [
        'demog',                    # Demographics
        'vle_statistics',           # VLE activity statistics
        'vle_statistics_beforestart',  # Pre-course VLE activity
        'reg_statistics'            # Registration statistics
        # Note: Focus on interpretable features for LLM narrative generation
        # Excluding detailed daily features for better narrative generation:
        # 'vle_day_activity_type_flags', 'vle_day_activity_type', 'vle_day', 'vle_day_flags'
    ]
}


def load_oulad_data(module: str, presentation: str, assessment_name: str, 
                    days_to_cutoff: int, hdf5_path: str, features: List[str]):
    """Load and prepare OULAD data for specific configuration
    
    Uses SELFLEARNER training type (same as traditional ML experiments):
    - Training data: from earlier students in the same presentation
    - Test data: later students in the same presentation
    - Ensures no data leakage through temporal split
    """
    logger.info(f"Loading data: {module}/{presentation}/{assessment_name}, day_cutoff={days_to_cutoff}")
    
    try:
        # Import TrainingType here to be explicit
        from selflearner.problem_definition import TrainingType
        
        # Define problem with SELFLEARNER training type (matching traditional ML)
        problem_def = ProblemDefinition(
            module=module,
            presentation=presentation,
            assessment_name=assessment_name,
            days_to_cutoff=days_to_cutoff,
            training_type=TrainingType.SELFLEARNER,  # Explicitly set to match traditional ML
            presentation_train=presentation,  # Train on same presentation (temporal split)
            y_column='submitted',  # Same label as traditional ML
            grouping_column='submit_in',
            id_column='id_student'
        )
        
        # Extract features using same extractor as traditional ML
        feature_extractor = FeatureExtractionOulad(problem_def, hdf5_path=hdf5_path)
        data = feature_extractor.extract_features(features)
        
        logger.info(f"Data loaded - Train: {len(data['x_train'])}, Test: {len(data['x_test'])}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data for {module}/{presentation}: {e}")
        return None


def prepare_student_narratives(data: dict, max_students: int = None, current_day: int = 0):
    """Convert student data to narratives"""
    logger.info(f"Converting student data to narratives (max={max_students}, day={current_day})")
    
    converter = BehaviorToTextConverter()
    
    # Get test data
    X_test = data['x_test']
    y_test = data['y_test']
    
    # Limit if specified
    if max_students:
        X_test = X_test.head(max_students)
        y_test = y_test.head(max_students)
    
    narratives = []
    labels = []
    student_ids = []
    
    for idx, row in tqdm(X_test.iterrows(), total=len(X_test), desc="Creating narratives"):
        student_id = row['id_student']
        
        # Prepare student data dict with course context
        student_data = {
            'id_student': student_id,
            'course_info': {
                'current_day': current_day,
                'assessment_name': data.get('assessment_name', 'TMA 1')
            },
            'demographics': {
                'gender': row.get('gender', 'Unknown'),
                'age_band': row.get('age_band', 'Unknown'),
                'highest_education': row.get('highest_education', 'Unknown'),
                'region': row.get('region', 'Unknown'),
                'num_of_prev_attempts': row.get('num_of_prev_attempts', 0),
                'disability': row.get('disability', 'N')
            },
            'vle_statistics': {
                'total_clicks': row.get('sum_click_fromvleopen', 0),
                'active_days': row.get('count_days_fromvleopen', 0),
                'total_days': current_day if current_day > 0 else row.get('count_days_fromvleopen', 0),
                'last_login': row.get('last_login_rel', 999),
                'first_login': row.get('first_login', 0),
                'unique_materials': row.get('sum_material_fromvleopen', 0),
                'avg_clicks_per_day': row.get('sum_click_fromvleopen', 0) / max(row.get('count_days_fromvleopen', 1), 1),
                'max_consecutive_days': row.get('consecutive_days', 0)
            }
        }
        
        narrative = converter.convert_student_to_narrative(
            student_data=student_data,
            include_demographics=True,
            include_statistics=True
        )
        
        narratives.append(narrative)
        student_ids.append(student_id)
        
        # Get label
        label_row = y_test[y_test['id_student'] == student_id]
        if len(label_row) > 0:
            labels.append(label_row.iloc[0]['submitted'])
        else:
            labels.append(0)
    
    return narratives, labels, student_ids


def run_single_configuration(module: str, presentation: str, assessment: str,
                            days_to_cutoff: int, multi_agent_system: MultiAgentSystem,
                            hdf5_path: str, features: List[str],
                            max_students: int = None,
                            output_dir: Path = None,
                            save_incrementally: bool = True,
                            n_workers: int = None):
    """Run experiment for a single configuration with resume support"""
    
    config_id = f"{module}_{presentation}_day{days_to_cutoff}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Running configuration: {config_id}")
    logger.info(f"{'='*60}")
    
    # Load data
    data = load_oulad_data(
        module=module,
        presentation=presentation,
        assessment_name=assessment,
        days_to_cutoff=days_to_cutoff,
        hdf5_path=hdf5_path,
        features=features
    )
    
    if data is None:
        logger.error(f"Failed to load data for {config_id}")
        return None
    
    # Prepare narratives
    narratives, labels, student_ids = prepare_student_narratives(
        data, 
        max_students=max_students,
        current_day=days_to_cutoff
    )
    
    if len(narratives) == 0:
        logger.warning(f"No students found for {config_id}")
        return None
    
    # Prepare incremental save file (no timestamp, fixed name for resume)
    completed_student_ids = set()
    if save_incrementally and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        incremental_file = output_dir / f"incremental_{config_id}.jsonl"
        
        # Check if file exists and load completed student IDs
        if incremental_file.exists():
            logger.info(f"Found existing results file: {incremental_file}")
            with open(incremental_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if 'student_id' in result and 'error' not in result:
                            completed_student_ids.add(result['student_id'])
                    except json.JSONDecodeError:
                        continue
            
            if completed_student_ids:
                logger.info(f"✓ Found {len(completed_student_ids)} completed students, will skip them")
                logger.info(f"✓ Resuming from where we left off...")
            else:
                logger.info(f"✓ File exists but no valid results found, starting fresh")
        else:
            logger.info(f"✓ Starting new experiment, results will be saved to: {incremental_file}")
    
    # Filter out already completed students
    students_to_process = []
    for i, (narrative, label, student_id) in enumerate(zip(narratives, labels, student_ids)):
        if student_id not in completed_student_ids:
            students_to_process.append((i, narrative, label, student_id))
    
    total_students = len(narratives)
    remaining_students = len(students_to_process)
    completed_count = total_students - remaining_students
    
    if completed_count > 0:
        logger.info(f"Progress: {completed_count}/{total_students} students already completed")
    
    if remaining_students == 0:
        logger.info(f"✓ All {total_students} students already processed, skipping this configuration")
        # Load existing results for metrics
        existing_results = []
        if save_incrementally and output_dir and incremental_file.exists():
            with open(incremental_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        existing_results.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        if existing_results:
            try:
                metrics = evaluate_predictions(existing_results)
                logger.info(f"\n{config_id} Results (from cache):")
                logger.info(f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}")
                logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                logger.info(f"  F1: {metrics.get('f1', 0):.4f}")
                
                return {
                    'config_id': config_id,
                    'module': module,
                    'presentation': presentation,
                    'days_to_cutoff': days_to_cutoff,
                    'predictions': existing_results,
                    'metrics': metrics,
                    'n_students': len(existing_results),
                    'resumed': True
                }
            except Exception as e:
                logger.error(f"Error evaluating cached results for {config_id}: {e}")
        
        return None
    
    logger.info(f"Processing {remaining_students}/{total_students} remaining students")
    
    # Determine number of workers for concurrent processing
    # Priority: 1) user-specified n_workers, 2) auto-detect from GPU count, 3) default to 1
    if n_workers is not None:
        logger.info(f"Using user-specified {n_workers} workers for concurrent processing")
    elif hasattr(multi_agent_system, 'llm') and hasattr(multi_agent_system.llm, 'server_urls'):
        n_workers = len(multi_agent_system.llm.server_urls)
        logger.info(f"Auto-detected {n_workers} workers (matching {n_workers} GPU servers)")
    elif hasattr(multi_agent_system, 'llm') and hasattr(multi_agent_system.llm, 'client') and hasattr(multi_agent_system.llm.client, 'server_urls'):
        n_workers = len(multi_agent_system.llm.client.server_urls)
        logger.info(f"Auto-detected {n_workers} workers (matching {n_workers} GPU servers)")
    else:
        n_workers = 1  # Default: serial processing
        logger.info(f"Using 1 worker (serial processing)")
    
    # Run predictions
    results = []
    
    # Thread-safe file writing lock
    file_lock = threading.Lock()
    
    def process_student(i, narrative, label, student_id):
        """Process a single student prediction"""
        try:
            prediction = multi_agent_system.predict(
                student_narrative=narrative,
                return_intermediate=True
            )
            
            # Add metadata
            prediction['ground_truth'] = int(label)
            prediction['student_id'] = int(student_id)
            prediction['student_index'] = i
            prediction['module'] = module
            prediction['presentation'] = presentation
            prediction['assessment'] = assessment
            prediction['days_to_cutoff'] = days_to_cutoff
            prediction['config_id'] = config_id
            
            # Save incrementally (thread-safe)
            if save_incrementally and output_dir:
                with file_lock:
                    with open(incremental_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                        f.flush()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing student {i} ({student_id}): {e}")
            error_result = {
                'error': str(e),
                'ground_truth': int(label),
                'student_id': int(student_id),
                'student_index': i,
                'module': module,
                'presentation': presentation,
                'assessment': assessment,
                'days_to_cutoff': days_to_cutoff,
                'config_id': config_id
            }
            
            # Save error too (thread-safe)
            if save_incrementally and output_dir:
                with file_lock:
                    with open(incremental_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                        f.flush()
            
            return error_result
    
    # Process only remaining students (skip already completed ones)
    if n_workers > 1:
        logger.info(f"🚀 Concurrent processing enabled with {n_workers} workers")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit tasks only for students that haven't been processed
            future_to_student = {
                executor.submit(process_student, i, narrative, label, student_id): i
                for i, narrative, label, student_id in students_to_process
            }
            
            # Collect results with progress bar
            with tqdm(total=remaining_students, desc=f"Predicting {config_id} ({n_workers} workers, {completed_count} done)") as pbar:
                for future in as_completed(future_to_student):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
    else:
        # Serial processing (original behavior)
        for i, narrative, label, student_id in tqdm(
            students_to_process, 
            total=remaining_students, 
            desc=f"Predicting {config_id} ({completed_count} done)"
        ):
            result = process_student(i, narrative, label, student_id)
            results.append(result)
    
    # Evaluate all results (including previously completed ones)
    # Load all results from file for complete evaluation
    all_results = []
    if save_incrementally and output_dir and incremental_file.exists():
        with open(incremental_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    
    if len(all_results) > 0:
        try:
            metrics = evaluate_predictions(all_results)
            logger.info(f"\n{config_id} Results (total {len(all_results)} students):")
            logger.info(f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}")
            logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            logger.info(f"  F1: {metrics.get('f1', 0):.4f}")
            
            return {
                'config_id': config_id,
                'module': module,
                'presentation': presentation,
                'days_to_cutoff': days_to_cutoff,
                'predictions': all_results,
                'metrics': metrics,
                'n_students': len(all_results),
                'resumed': completed_count > 0
            }
        except Exception as e:
            logger.error(f"Error evaluating {config_id}: {e}")
            return {
                'config_id': config_id,
                'module': module,
                'presentation': presentation,
                'days_to_cutoff': days_to_cutoff,
                'predictions': all_results,
                'n_students': len(all_results),
                'error': str(e)
            }
    
    return None


def run_paper_replication_experiments(args):
    """Run all experiments matching paper configuration"""
    
    logger.info("="*70)
    logger.info("LLM MULTI-AGENT EXPERIMENTS - PAPER REPLICATION")
    logger.info("="*70)
    logger.info(f"Modules: {args.module if args.module else PAPER_CONFIG['modules']}")
    logger.info(f"Presentations: {args.presentation if args.presentation else PAPER_CONFIG['presentations']}")
    logger.info(f"Assessment: {PAPER_CONFIG['assessment']}")
    days_display = args.day if args.day is not None else f"0-{PAPER_CONFIG['max_days']}"
    logger.info(f"Days range: {days_display}")
    logger.info(f"Pilot mode: {args.pilot}")
    if args.n_students:
        logger.info(f"Max students per config: {args.n_students}")
    logger.info("="*70)
    
    # Load configuration
    config_path = args.llm_config or Path(__file__).parent.parent / 'config' / 'llm_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize LLM wrapper
    logger.info(f"Initializing LLM: {config['llm']['provider']}")
    llm_wrapper = create_llm_wrapper(
        provider=config['llm']['provider'],
        config=config['llm'][config['llm']['provider']]
    )
    
    # Initialize multi-agent system
    agent_config_path = args.agent_config or Path(__file__).parent.parent / 'config' / 'agent_config.yaml'
    multi_agent_system = MultiAgentSystem(
        llm_wrapper=llm_wrapper,
        config_path=agent_config_path
    )
    
    # Determine which configurations to run
    if args.module:
        modules = [args.module]
    else:
        modules = PAPER_CONFIG['modules']
    
    if args.presentation:
        presentations = [args.presentation]
    else:
        presentations = PAPER_CONFIG['presentations']
    
    # Always use single assessment from PAPER_CONFIG
    assessment = PAPER_CONFIG['assessment']
    
    if args.day is not None:
        days_range = [args.day]
    elif args.pilot:
        days_range = [0, 1, 2]  # Test first 3 days in pilot
    else:
        days_range = range(PAPER_CONFIG['max_days'] + 1)  # 0 to 11 (12 days total)
    
    # Run all combinations
    all_results = []
    total_configs = len(modules) * len(presentations) * len(days_range)
    
    logger.info(f"\nTotal configurations to run: {total_configs}")
    logger.info(f"Configuration: {len(modules)} modules × {len(presentations)} presentations × {len(days_range)} days")
    logger.info(f"Assessment: {assessment}")
    
    # Show concurrency info
    if args.n_workers:
        logger.info(f"🚀 Concurrent workers: {args.n_workers} (user-specified)")
    elif hasattr(llm_wrapper, 'server_urls'):
        n_gpus = len(llm_wrapper.server_urls)
        logger.info(f"🚀 Concurrent workers: {n_gpus} (auto-detected from {n_gpus} GPU servers)")
    elif hasattr(llm_wrapper, 'client') and hasattr(llm_wrapper.client, 'server_urls'):
        n_gpus = len(llm_wrapper.client.server_urls)
        logger.info(f"🚀 Concurrent workers: {n_gpus} (auto-detected from {n_gpus} GPU servers)")
    else:
        logger.info(f"Processing mode: Serial (1 worker)")
    
    logger.info(f"Estimated time per student: 30-60 seconds (serial)")
    if args.n_students:
        logger.info(f"Students per config: ~{args.n_students}")
        base_time = total_configs * args.n_students * 45 / 3600
        if args.n_workers and args.n_workers > 1:
            logger.info(f"Estimated total time: {base_time:.1f} hours (serial) → {base_time/args.n_workers:.1f} hours (with {args.n_workers} workers)")
        elif hasattr(llm_wrapper, 'server_urls') and len(llm_wrapper.server_urls) > 1:
            n_gpus = len(llm_wrapper.server_urls)
            logger.info(f"Estimated total time: {base_time:.1f} hours (serial) → {base_time/n_gpus:.1f} hours (with {n_gpus} workers)")
        elif hasattr(llm_wrapper, 'client') and hasattr(llm_wrapper.client, 'server_urls') and len(llm_wrapper.client.server_urls) > 1:
            n_gpus = len(llm_wrapper.client.server_urls)
            logger.info(f"Estimated total time: {base_time:.1f} hours (serial) → {base_time/n_gpus:.1f} hours (with {n_gpus} workers)")
        else:
            logger.info(f"Estimated total time: {base_time:.1f} hours")
    logger.info("")
    
    config_count = 0
    
    for module in modules:
        for presentation in presentations:
            for day in days_range:
                config_count += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"Configuration {config_count}/{total_configs}")
                logger.info(f"Module: {module}, Presentation: {presentation}, Day: {day}")
                logger.info(f"{'='*70}")
                
                result = run_single_configuration(
                    module=module,
                    presentation=presentation,
                    assessment=assessment,
                    days_to_cutoff=day,
                    multi_agent_system=multi_agent_system,
                    hdf5_path=args.hdf5_path,
                    features=PAPER_CONFIG['features'],
                    max_students=args.n_students,
                    output_dir=Path(args.output_dir),
                    save_incrementally=args.save_incrementally,
                    n_workers=args.n_workers
                )
                
                if result:
                    all_results.append(result)
    
    # Save consolidated results (no timestamp, fixed filename)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    output_file = output_dir / "paper_replication_full.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nFull results saved to: {output_file}")
    
    # Create summary table (like Table 4 in paper)
    summary_data = []
    for result in all_results:
        if 'metrics' in result:
            summary_data.append({
                'Module': result['module'],
                'Presentation': result['presentation'],
                'Day': result['days_to_cutoff'],
                'PR-AUC': result['metrics'].get('pr_auc', 0),
                'ROC-AUC': result['metrics'].get('roc_auc', 0),
                'F1': result['metrics'].get('f1', 0),
                'Precision': result['metrics'].get('precision', 0),
                'Recall': result['metrics'].get('recall', 0),
                'N_Students': result['n_students'],
                'Resumed': result.get('resumed', False)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "paper_replication_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary table saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("PAPER REPLICATION SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("="*70)
    
    # Print final statistics
    print(f"\nTotal configurations completed: {len(all_results)}")
    print(f"LLM calls made: {llm_wrapper.get_usage_stats()['total_calls']}")
    print(f"Total tokens used: {llm_wrapper.get_usage_stats()['total_tokens']:,}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run LLM multi-agent experiments on OULAD dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pilot test with 10 students
  python -m llm_experiments.experiments.run_paper_replication --pilot --n_students 10
  
  # Single configuration
  python -m llm_experiments.experiments.run_paper_replication \\
      --module BBB --presentation 2014J --assessment "TMA 1" --day 0
  
  # All BBB module experiments
  python -m llm_experiments.experiments.run_paper_replication --module BBB
  
  # Full paper replication (all configs)
  python -m llm_experiments.experiments.run_paper_replication
"""
    )
    
    # Experiment scope
    parser.add_argument('--module', type=str, default=None,
                       help='Single module to test (e.g., BBB). Default: all modules [BBB,DDD,EEE,FFF]')
    parser.add_argument('--presentation', type=str, default=None,
                       help='Single presentation to test (e.g., 2014J). Default: 2014J')
    parser.add_argument('--day', type=int, default=None,
                       help='Single day to test (e.g., 0). Default: all days 0-11')
    
    # Experiment modes
    parser.add_argument('--pilot', action='store_true', 
                       help='Run pilot mode: 1 module, 1 presentation, 1 assessment, day 0 only')
    parser.add_argument('--n_students', type=int, default=None, 
                       help='Maximum number of students per configuration (default: all)')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of concurrent workers for parallel processing (default: auto-detect from GPU count, 1 for non-multi-GPU)')
    
    # Data
    parser.add_argument('--hdf5_path', type=str, 
                       default='selflearner/data_load/data/oulad.h5',
                       help='Path to OULAD HDF5 file')
    
    # Configuration files
    parser.add_argument('--llm_config', type=str, default=None,
                       help='Path to LLM configuration file (default: config/llm_config.yaml)')
    parser.add_argument('--agent_config', type=str, default=None,
                       help='Path to agent configuration file (default: config/agent_config.yaml)')
    
    # Output
    parser.add_argument('--output_dir', type=str, 
                       default='./llm_experiments/results/paper_replication',
                       help='Output directory for results')
    parser.add_argument('--save_incrementally', action='store_true', default=True,
                       help='Save results incrementally (one student per line)')
    
    args = parser.parse_args()
    
    # Run experiments
    results = run_paper_replication_experiments(args)
    
    return results


if __name__ == '__main__':
    main()

