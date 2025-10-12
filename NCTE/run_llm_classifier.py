#!/usr/bin/env python3
"""
Simplified LLM Evaluator for NCTE Classroom Transcript Analysis

This module provides a clean evaluation interface that uses:
- PromptManager for prompt handling
- LLMInference for model inference
- Real-time logging and checkpointing
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pandas as pd
import numpy as np

from prompt_manager import PromptManager
from llm_inference import LLMInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
        # Setup file paths
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records_file = (
            self.output_dir
            / f"results_{model_name.replace('/', '_')}_{self.session_id}_{self.mode}.jsonl"
        )
        self.summary_file = (
            self.output_dir
            / f"summary_{model_name.replace('/', '_')}_{self.session_id}_{self.mode}.json"
        )
        self.checkpoint_file = (
            self.output_dir
            / f"checkpoint_{model_name.replace('/', '_')}_{self.session_id}_{self.mode}.json"
        )

        # State tracking
        self.task_summaries = {}
        self.total_examples = 0
        self.completed_examples = 0
        self.start_time = datetime.now().isoformat()

        # Fixed few-shot examples for consistency
        self.fixed_few_shot_examples = {}

        logger.info(f"Initialized NCTEEvaluator for {model_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Records file: {self.records_file}")
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

    def _update_summary(self):
        """Update and save summary statistics."""
        try:
            summary = {
                "model_name": self.model_name,
                "session_id": self.session_id,
                "start_time": self.start_time,
                "last_update": datetime.now().isoformat(),
                "total_examples": self.total_examples,
                "completed_examples": self.completed_examples,
                "progress_percentage": (
                    self.completed_examples / max(1, self.total_examples)
                )
                * 100,
                "task_summaries": {
                    name: asdict(summary)
                    for name, summary in self.task_summaries.items()
                },
            }

            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error updating summary: {e}")

    def _save_checkpoint(self, task_name: str, completed_indices: List[int]):
        """Save checkpoint for resume capability."""
        try:
            checkpoint = {
                "model_name": self.model_name,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "current_task": task_name,
                "completed_indices": completed_indices,
                "task_summaries": {
                    name: asdict(summary)
                    for name, summary in self.task_summaries.items()
                },
            }

            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

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

        # Filter data for this task
        task_data = data.dropna(subset=[task_name]).reset_index(drop=True)

        if max_examples != -1:
            task_data = task_data.head(max_examples)

        logger.info(f"Task data size: {len(task_data)} examples")

        # Prepare fixed few-shot examples if needed
        if mode == "few_shot" and train_data is not None and n_shots > 0:
            logger.info(f"Preparing fixed {n_shots}-shot examples for {task_name}")
            self.prepare_fixed_few_shot_examples(task_name, train_data, n_shots)

        self.total_examples += len(task_data)

        # Initialize task summary
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

        self.task_summaries[task_name] = task_summary

        # Process examples with progress bar
        completed_indices = []
        total_inference_time = 0.0

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

                        self._update_summary()
                        self._save_checkpoint(task_name, completed_indices)

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

        self._update_summary()
        self._save_checkpoint(task_name, completed_indices)

        logger.info(f"Completed task {task_name}:")
        logger.info(
            f"  Examples: {task_summary.completed_examples}/{task_summary.total_examples}"
        )
        logger.info(f"  Accuracy: {task_summary.accuracy:.3f}")
        logger.info(f"  Avg inference time: {task_summary.avg_inference_time:.3f}s")

        return task_summary

    def get_results_summary(self) -> Dict[str, Any]:
        """Get comprehensive results summary."""
        return {
            "model_name": self.model_name,
            "session_id": self.session_id,
            "total_examples": self.total_examples,
            "completed_examples": self.completed_examples,
            "task_summaries": {
                name: asdict(summary) for name, summary in self.task_summaries.items()
            },
            "files": {
                "detailed_results": str(self.records_file),
                "summary": str(self.summary_file),
                "checkpoint": str(self.checkpoint_file),
            },
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

        for task_name, summary in results["task_summaries"].items():
            logger.info(
                f"{task_name}: {summary['accuracy']:.3f} accuracy "
                f"({summary['completed_examples']}/{summary['total_examples']} examples)"
            )

        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Detailed records: {results['files']['detailed_results']}")
        logger.info(f"Summary: {results['files']['summary']}")


if __name__ == "__main__":
    main()
