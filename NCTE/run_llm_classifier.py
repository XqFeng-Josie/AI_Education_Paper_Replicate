#!/usr/bin/env python3
"""
Simplified LLM Evaluator for NCTE Classroom Transcript Analysis

This module provides a clean evaluation interface that uses:
- PromptManager for prompt handling
- LLMInference for model inference
- Real-time logging and checkpointing
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pandas as pd

from prompt_manager import PromptManager
from llm_inference import LLMInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# OpenAI pricing (as of 2024, in USD per 1M tokens)
OPENAI_PRICING = {
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-4": {"prompt": 30.00, "completion": 60.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
}


def calculate_openai_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost for OpenAI API usage.
    
    Args:
        model_name: Name of the OpenAI model
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
    
    Returns:
        Estimated cost in USD
    """
    # Find matching pricing
    pricing = None
    for key, value in OPENAI_PRICING.items():
        if key in model_name.lower():
            pricing = value
            break
    
    if not pricing:
        # Default to gpt-4o-mini pricing if model not found
        pricing = OPENAI_PRICING["gpt-4o-mini"]
    
    # Calculate cost (pricing is per 1M tokens)
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    
    return prompt_cost + completion_cost


@dataclass
class EvaluationRecord:
    """Single evaluation record with all details."""

    timestamp: str
    model_name: str
    task_name: str
    mode: str  # zero_shot or few_shot
    n_shots: int
    example_idx: int
    example_id: str
    student_text: str
    teacher_text: str
    true_label: int

    # Prompt details
    prompt_type: str  # string or message
    prompt_content: str
    few_shot_examples: str

    # Model response details
    raw_response: str
    extracted_label: int
    inference_time: float

    # Token usage (for API models like OpenAI)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Evaluation
    correct: bool
    error_message: Optional[str]


@dataclass
class TaskSummary:
    """Summary statistics for a task."""

    task_name: str
    total_examples: int
    completed_examples: int
    correct_predictions: int
    accuracy: float
    start_time: str
    last_update: str
    avg_inference_time: float
    
    # Token usage statistics (for API models)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class NCTEEvaluator:
    """Simplified evaluator for NCTE tasks."""

    def __init__(
        self,
        model_name: str,
        output_dir: str = "outputs/evaluation",
        device: str = "auto",
        mode: str = "zero_shot",
    ):
        """Initialize the evaluator."""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.prompt_manager = PromptManager()
        self.llm_inference = LLMInference(model_name, device=device)
        self.mode = mode
        
        # File paths without timestamp - will be set per task
        self.current_task_name = None
        self.records_file = None
        self.summary_file = None

        # State tracking
        self.task_summaries = {}
        self.total_examples = 0
        self.completed_examples = 0
        self.start_time = datetime.now().isoformat()

        # Fixed few-shot examples for consistency
        self.fixed_few_shot_examples = {}
        
        # Completed example IDs (for resume capability)
        self.completed_example_ids = set()

        logger.info(f"Initialized NCTEEvaluator for {model_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Using message format: True")

    def prepare_fixed_few_shot_examples(
        self, task_name: str, train_data: pd.DataFrame, n_shots: int = 4
    ) -> str:
        """Prepare fixed few-shot examples for consistency."""
        if task_name in self.fixed_few_shot_examples:
            return self.fixed_few_shot_examples[task_name]

        logger.info(f"Preparing fixed {n_shots}-shot examples for {task_name}")

        few_shot_examples = self.prompt_manager.prepare_few_shot_examples(
            task_name, train_data, n_shots
        )

        # Store for reuse
        self.fixed_few_shot_examples[task_name] = few_shot_examples

        logger.info(f"Prepared few-shot examples for {task_name}")
        return few_shot_examples

    def _save_record(self, record: EvaluationRecord):
        """Save single evaluation record to JSONL file."""
        try:
            with open(self.records_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error saving record: {e}")

    def _update_summary(self, task_name: str):
        """Update and save summary statistics for a specific task."""
        try:
            if task_name not in self.task_summaries:
                return
            
            task_summary = self.task_summaries[task_name]
            summary = {
                "model_name": self.model_name,
                "mode": self.mode,
                "task_name": task_name,
                "start_time": self.start_time,
                "last_update": datetime.now().isoformat(),
                **asdict(task_summary)
            }

            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error updating summary: {e}")

    def _set_task_files(self, task_name: str):
        """Set file paths for a specific task."""
        self.current_task_name = task_name
        model_clean = self.model_name.replace('/', '_')
        
        self.records_file = (
            self.output_dir / f"results_{model_clean}_{task_name}_{self.mode}.jsonl"
        )
        self.summary_file = (
            self.output_dir / f"summary_{model_clean}_{task_name}_{self.mode}.json"
        )
        
        logger.info(f"Task files set:")
        logger.info(f"  Results: {self.records_file}")
        logger.info(f"  Summary: {self.summary_file}")
    
    def _load_completed_example_ids(self, task_name: str) -> set:
        """Load already completed example IDs from existing results file.
        
        Args:
            task_name: The task name to load completed IDs for
            
        Returns:
            Set of completed example IDs for this specific task
        """
        completed_ids = set()
        mismatched_tasks = 0
        
        if self.records_file and self.records_file.exists():
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            
                            # Verify task_name matches to ensure we're reading the right file
                            record_task = record.get('task_name', '')
                            if record_task != task_name:
                                mismatched_tasks += 1
                                continue
                            
                            # Only add example_id if task matches
                            if 'example_id' in record:
                                completed_ids.add(str(record['example_id']))
                                
                        except json.JSONDecodeError:
                            continue
                
                if mismatched_tasks > 0:
                    logger.warning(
                        f"Found {mismatched_tasks} records with mismatched task_name in {self.records_file.name}"
                    )
                
                logger.info(f"Found {len(completed_ids)} completed examples for task '{task_name}'")
            except Exception as e:
                logger.warning(f"Error loading completed IDs: {e}")
        else:
            logger.info(f"No existing results file found for task '{task_name}', starting fresh")
        
        return completed_ids
    
    def _load_existing_summary(self, task_name: str) -> Optional[TaskSummary]:
        """Load existing task summary if available.
        
        Args:
            task_name: The task name to load summary for
            
        Returns:
            TaskSummary object if found and valid, None otherwise
        """
        if self.summary_file and self.summary_file.exists():
            try:
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                # Verify task_name matches
                summary_task = summary_data.get('task_name', '')
                if summary_task != task_name:
                    logger.warning(
                        f"Summary task_name mismatch: expected '{task_name}', "
                        f"found '{summary_task}' in {self.summary_file.name}. "
                        f"Starting fresh for task '{task_name}'."
                    )
                    return None
                
                # Convert to TaskSummary
                task_summary = TaskSummary(
                    task_name=summary_data.get('task_name', task_name),
                    total_examples=summary_data.get('total_examples', 0),
                    completed_examples=summary_data.get('completed_examples', 0),
                    correct_predictions=summary_data.get('correct_predictions', 0),
                    accuracy=summary_data.get('accuracy', 0.0),
                    start_time=summary_data.get('start_time', datetime.now().isoformat()),
                    last_update=summary_data.get('last_update', datetime.now().isoformat()),
                    avg_inference_time=summary_data.get('avg_inference_time', 0.0),
                    total_prompt_tokens=summary_data.get('total_prompt_tokens', 0),
                    total_completion_tokens=summary_data.get('total_completion_tokens', 0),
                    total_tokens=summary_data.get('total_tokens', 0),
                    estimated_cost_usd=summary_data.get('estimated_cost_usd', 0.0),
                )
                
                logger.info(
                    f"Loaded existing summary for task '{task_name}': "
                    f"{task_summary.completed_examples} examples already completed"
                )
                return task_summary
                
            except Exception as e:
                logger.warning(f"Could not load existing summary: {e}")
        
        return None
    
    def _recalculate_summary_from_results(self, task_name: str) -> TaskSummary:
        """Recalculate summary statistics from the results file.
        
        This method reads all records from the results file and recalculates
        all statistics from scratch, ensuring the summary is always up-to-date.
        """
        logger.info(f"Recalculating summary for task '{task_name}' from results file")
        
        if not self.records_file.exists():
            logger.warning(f"Results file not found: {self.records_file}")
            return TaskSummary(
                task_name=task_name,
                total_examples=0,
                completed_examples=0,
                correct_predictions=0,
                accuracy=0.0,
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                avg_inference_time=0.0,
            )
        
        # Statistics to calculate
        total_examples = 0
        correct_predictions = 0
        total_inference_time = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        first_timestamp = None
        last_timestamp = None
        
        try:
            with open(self.records_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        
                        # Verify task_name matches
                        if record.get('task_name') != task_name:
                            continue
                        
                        total_examples += 1
                        
                        # Check if prediction is correct
                        if record.get('extracted_label') == record.get('true_label'):
                            correct_predictions += 1
                        
                        # Accumulate inference time
                        total_inference_time += record.get('inference_time', 0.0)
                        
                        # Accumulate token usage
                        total_prompt_tokens += record.get('prompt_tokens', 0)
                        total_completion_tokens += record.get('completion_tokens', 0)
                        total_tokens += record.get('total_tokens', 0)
                        
                        # Track timestamps
                        timestamp = record.get('timestamp', '')
                        if timestamp:
                            if first_timestamp is None:
                                first_timestamp = timestamp
                            last_timestamp = timestamp
                    
                    except json.JSONDecodeError:
                        continue
            
            # Calculate statistics
            accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
            avg_inference_time = total_inference_time / total_examples if total_examples > 0 else 0.0
            
            # Calculate cost for OpenAI models
            estimated_cost = 0.0
            if "gpt" in self.model_name.lower():
                estimated_cost = calculate_openai_cost(
                    self.model_name,
                    total_prompt_tokens,
                    total_completion_tokens
                )
            
            summary = TaskSummary(
                task_name=task_name,
                total_examples=total_examples,
                completed_examples=total_examples,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
                start_time=first_timestamp or datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                avg_inference_time=avg_inference_time,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost,
            )
            
            logger.info(
                f"Recalculated summary: {total_examples} examples, "
                f"accuracy={accuracy:.4f}, tokens={total_tokens:,}"
            )
            
            return summary
        
        except Exception as e:
            logger.error(f"Error recalculating summary from results: {e}")
            return TaskSummary(
                task_name=task_name,
                total_examples=0,
                completed_examples=0,
                correct_predictions=0,
                accuracy=0.0,
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                avg_inference_time=0.0,
            )

    def evaluate_single_example(
        self,
        task_name: str,
        example_data: Dict[str, Any],
        mode: str = "zero_shot",
        n_shots: int = 0,
        train_data: Optional[pd.DataFrame] = None,
    ) -> EvaluationRecord:
        """Evaluate a single example."""

        start_time = time.time()

        # Prepare few-shot examples if needed
        few_shot_examples = ""
        if mode == "few_shot" and train_data is not None and n_shots > 0:
            few_shot_examples = self.prepare_fixed_few_shot_examples(
                task_name, train_data, n_shots
            )

        # Get student and teacher text (handle both data formats)
        student_text = example_data.get("student_text", example_data.get("text", ""))
        teacher_text = example_data.get("teacher_text", "")

        # Format prompt using message format only
        prompt = self.prompt_manager.format_message_prompt(
            task_name, student_text, teacher_text, few_shot_examples
        )
        prompt_content = str(prompt)  # Convert to string for logging
        prompt_type = "message"

        # Get model response
        try:
            raw_response = self.llm_inference.generate_response(
                prompt, max_new_tokens=10
            )
            extracted_label = self.llm_inference._extract_classification(raw_response)
            error_message = None

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raw_response = ""
            extracted_label = 0  # Default
            error_message = str(e)

        inference_time = time.time() - start_time
        
        # Get token usage (for API models like OpenAI)
        token_usage = self.llm_inference.get_token_usage()

        # Get example ID (handle different formats)
        example_id = str(
            example_data.get(
                "exchange_idx", example_data.get("comb_idx", self.completed_examples)
            )
        )

        # Create evaluation record
        record = EvaluationRecord(
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            task_name=task_name,
            mode=mode,
            n_shots=n_shots,
            example_idx=self.completed_examples,
            example_id=example_id,
            student_text=student_text,
            teacher_text=teacher_text,
            true_label=int(example_data[task_name]),
            prompt_type=prompt_type,
            prompt_content=prompt_content,
            few_shot_examples=few_shot_examples,
            raw_response=raw_response,
            extracted_label=extracted_label,
            inference_time=inference_time,
            prompt_tokens=token_usage["prompt_tokens"],
            completion_tokens=token_usage["completion_tokens"],
            total_tokens=token_usage["total_tokens"],
            correct=(extracted_label == int(example_data[task_name])),
            error_message=error_message,
        )

        # Save record immediately
        self._save_record(record)
        self.completed_examples += 1

        return record

    def evaluate_task(
        self,
        data: pd.DataFrame,
        task_name: str,
        mode: str = "zero_shot",
        n_shots: int = 0,
        train_data: Optional[pd.DataFrame] = None,
        max_examples: Optional[int] = -1,
    ) -> TaskSummary:
        """Evaluate a complete task."""

        logger.info(f"Starting evaluation for task: {task_name}")
        logger.info(f"Mode: {mode}, N-shots: {n_shots}")
        
        # Set task-specific file paths
        self._set_task_files(task_name)
        
        # Load already completed example IDs
        completed_ids = self._load_completed_example_ids(task_name)

        # Filter data for this task
        task_data = data.dropna(subset=[task_name]).reset_index(drop=True)
        
        # Filter out already completed examples
        if completed_ids:
            # Get example_id column (could be exchange_idx or comb_idx)
            id_col = None
            if 'exchange_idx' in task_data.columns:
                id_col = 'exchange_idx'
            elif 'comb_idx' in task_data.columns:
                id_col = 'comb_idx'
            
            if id_col:
                original_size = len(task_data)
                task_data = task_data[~task_data[id_col].astype(str).isin(completed_ids)].reset_index(drop=True)
                skipped = original_size - len(task_data)
                if skipped > 0:
                    logger.info(f"Skipping {skipped} already completed examples")
        
        if max_examples != -1:
            task_data = task_data.head(max_examples)

        logger.info(f"Task data size: {len(task_data)} examples (to process)")
        
        # If no examples to process, recalculate summary from results file
        if len(task_data) == 0:
            logger.info(f"All {len(completed_ids)} examples already completed for task '{task_name}'")
            logger.info("Recalculating summary statistics from results file...")
            
            # Recalculate summary from results file
            task_summary = self._recalculate_summary_from_results(task_name)
            self.task_summaries[task_name] = task_summary
            
            # Save the recalculated summary
            self._update_summary(task_name)
            
            logger.info(
                f"Summary updated: {task_summary.completed_examples} examples, "
                f"accuracy={task_summary.accuracy:.4f}"
            )
            
            return task_summary

        # Prepare fixed few-shot examples if needed
        if mode == "few_shot" and train_data is not None and n_shots > 0:
            logger.info(f"Preparing fixed {n_shots}-shot examples for {task_name}")
            self.prepare_fixed_few_shot_examples(task_name, train_data, n_shots)

        # Load existing summary or create new one
        existing_summary = self._load_existing_summary(task_name)
        
        if existing_summary:
            # Continue from existing summary
            task_summary = existing_summary
            # Update total_examples to include new ones
            task_summary.total_examples += len(task_data)
            total_inference_time = task_summary.avg_inference_time * task_summary.completed_examples
        else:
            # Initialize new task summary
            task_summary = TaskSummary(
                task_name=task_name,
                total_examples=len(task_data),
                completed_examples=0,
                correct_predictions=0,
                accuracy=0.0,
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                avg_inference_time=0.0,
            )
            total_inference_time = 0.0

        self.task_summaries[task_name] = task_summary
        self.total_examples += len(task_data)

        # Process examples with progress bar
        completed_indices = []

        with tqdm(total=len(task_data), desc=f"Evaluating {task_name}") as pbar:
            for idx, (_, row) in enumerate(task_data.iterrows()):
                try:
                    # Evaluate example
                    record = self.evaluate_single_example(
                        task_name, row.to_dict(), mode, n_shots, train_data
                    )

                    # Update statistics
                    if record.correct:
                        task_summary.correct_predictions += 1

                    task_summary.completed_examples += 1
                    total_inference_time += record.inference_time
                    
                    # Update token usage statistics
                    task_summary.total_prompt_tokens += record.prompt_tokens
                    task_summary.total_completion_tokens += record.completion_tokens
                    task_summary.total_tokens += record.total_tokens
                    
                    # Calculate cost for OpenAI models
                    if "gpt" in self.model_name.lower():
                        task_summary.estimated_cost_usd = calculate_openai_cost(
                            self.model_name,
                            task_summary.total_prompt_tokens,
                            task_summary.total_completion_tokens
                        )
                    
                    completed_indices.append(idx)

                    # Update summary every 10 examples
                    if len(completed_indices) % 100 == 0 or len(
                        completed_indices
                    ) == len(task_data):
                        task_summary.accuracy = (
                            task_summary.correct_predictions
                            / task_summary.completed_examples
                        )
                        task_summary.avg_inference_time = (
                            total_inference_time / task_summary.completed_examples
                        )
                        task_summary.last_update = datetime.now().isoformat()

                        self._update_summary(task_name)

                        # Log progress with token usage for API models
                        if "gpt" in self.model_name.lower():
                            logger.info(
                                f"Progress: {len(completed_indices)}/{len(task_data)}, "
                                f"Accuracy: {task_summary.accuracy:.3f}, "
                                f"Tokens: {task_summary.total_tokens:,}, "
                                f"Cost: ${task_summary.estimated_cost_usd:.4f}"
                            )
                        else:
                            logger.info(
                                f"Progress: {len(completed_indices)}/{len(task_data)}, "
                                f"Accuracy: {task_summary.accuracy:.3f}"
                            )

                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error processing example {idx}: {e}")
                    continue

        # Final update
        task_summary.accuracy = task_summary.correct_predictions / max(
            1, task_summary.completed_examples
        )
        task_summary.avg_inference_time = total_inference_time / max(
            1, task_summary.completed_examples
        )
        task_summary.last_update = datetime.now().isoformat()

        self._update_summary(task_name)

        logger.info(f"Completed task {task_name}:")
        logger.info(
            f"  Examples: {task_summary.completed_examples}/{task_summary.total_examples}"
        )
        logger.info(f"  Accuracy: {task_summary.accuracy:.3f}")
        logger.info(f"  Avg inference time: {task_summary.avg_inference_time:.3f}s")
        
        # Log token usage and cost for API models
        if "gpt" in self.model_name.lower():
            logger.info(f"  Total tokens: {task_summary.total_tokens:,}")
            logger.info(f"    - Prompt tokens: {task_summary.total_prompt_tokens:,}")
            logger.info(f"    - Completion tokens: {task_summary.total_completion_tokens:,}")
            logger.info(f"  Estimated cost: ${task_summary.estimated_cost_usd:.4f} USD")

        return task_summary

    def get_results_summary(self) -> Dict[str, Any]:
        """Get comprehensive results summary."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "total_examples": self.total_examples,
            "completed_examples": self.completed_examples,
            "task_summaries": {
                name: asdict(summary) for name, summary in self.task_summaries.items()
            },
            "output_directory": str(self.output_dir),
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description="NCTE LLM Evaluation")

    parser.add_argument(
        "--model_name",
        type=str,
        default="llama-3.1-8b-instruct",
        help="Model name to evaluate",
    )
    parser.add_argument(
        "--task_name", type=str, default="all", help="Task to evaluate or 'all'"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--n_shots", type=int, default=5, help="Number of few-shot examples"
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Dataset file (defaults to paired_annotations.csv or student_reasoning.csv based on task)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=-1,
        help="Maximum examples per task (for testing), if -1, use all examples",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/evaluation", help="Output directory"
    )

    parser.add_argument("--device", type=str, default="auto", help="Device for model")

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()

    logger.info("Starting NCTE LLM evaluation")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Message format: True (greedy generation)")

    # Define task groups and their data files
    paired_tasks = [
        "student_on_task",
        "teacher_on_task",
        "high_uptake",
        "focusing_question",
    ]
    reasoning_tasks = ["student_reasoning"]
    tasks = [
        "student_on_task",
        "teacher_on_task",
        "high_uptake",
        "focusing_question",
        "student_reasoning",
    ]

    # Load appropriate data based on task_name
    data_dict = {}  # Store data for different task groups

    # Load both data files for all tasks
    logger.info("Loading data for all tasks...")
    try:
        paired_data = pd.read_csv("data/paired_annotations.csv")
        logger.info(f"Loaded {len(paired_data)} examples from paired_annotations.csv")
        data_dict["paired"] = paired_data
    except Exception as e:
        logger.error(f"Error loading paired_annotations.csv: {e}")
        return

    try:
        reasoning_data = pd.read_csv("data/student_reasoning.csv")
        logger.info(f"Loaded {len(reasoning_data)} examples from student_reasoning.csv")
        data_dict["reasoning"] = reasoning_data
    except Exception as e:
        logger.error(f"Error loading student_reasoning.csv: {e}")
        return
    # Initialize evaluator
    try:
        evaluator = NCTEEvaluator(
            args.model_name, args.output_dir, args.device, args.mode
        )
        logger.info("Evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing evaluator: {e}")
        return

    # Run evaluation

    if args.task_name != "all":
        tasks = [args.task_name]

    try:
        for task in tasks:
            # Determine which data to use for this task
            if task in reasoning_tasks:
                if "reasoning" not in data_dict:
                    logger.warning(
                        f"Task {task} requires student_reasoning.csv which is not loaded"
                    )
                    continue
                test_data = data_dict["reasoning"]
                train_data = data_dict["reasoning"]
            elif task in paired_tasks:
                if "paired" not in data_dict:
                    logger.warning(
                        f"Task {task} requires paired_annotations.csv which is not loaded"
                    )
                    continue
                test_data = data_dict["paired"]
                train_data = data_dict["paired"]
            else:
                logger.warning(f"Unknown task: {task}")
                continue

            if task in test_data.columns:
                logger.info(f"Starting task: {task}")
                logger.info(f"Using {len(test_data)} examples")

                task_summary = evaluator.evaluate_task(
                    test_data,
                    task,
                    args.mode,
                    args.n_shots,
                    train_data,
                    args.max_examples,
                )

                logger.info(
                    f"Task {task} completed with accuracy: {task_summary.accuracy:.3f}"
                )
            else:
                logger.warning(
                    f"Task {task} not found in data columns: {test_data.columns.tolist()}"
                )

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    finally:
        # Final summary
        results = evaluator.get_results_summary()
        logger.info("=== EVALUATION COMPLETED ===")
        logger.info(
            f"Total examples processed: {results['completed_examples']}/{results['total_examples']}"
        )

        # Calculate total tokens and cost across all tasks
        total_tokens_all_tasks = 0
        total_cost_all_tasks = 0.0
        
        for task_name, summary in results["task_summaries"].items():
            logger.info(
                f"{task_name}: {summary['accuracy']:.3f} accuracy "
                f"({summary['completed_examples']}/{summary['total_examples']} examples)"
            )
            
            # Add token usage info for API models
            if "gpt" in args.model_name.lower():
                total_tokens_all_tasks += summary.get('total_tokens', 0)
                total_cost_all_tasks += summary.get('estimated_cost_usd', 0.0)
                logger.info(
                    f"  └─ Tokens: {summary.get('total_tokens', 0):,}, "
                    f"Cost: ${summary.get('estimated_cost_usd', 0.0):.4f}"
                )
        
        # Log total cost for API models
        if "gpt" in args.model_name.lower() and total_tokens_all_tasks > 0:
            logger.info("=" * 50)
            logger.info("📊 TOTAL API USAGE:")
            logger.info(f"  Total tokens: {total_tokens_all_tasks:,}")
            logger.info(f"  Total estimated cost: ${total_cost_all_tasks:.4f} USD")
            logger.info("=" * 50)

        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Output directory: {results['output_directory']}")


if __name__ == "__main__":
    main()
