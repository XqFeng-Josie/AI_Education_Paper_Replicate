"""
Step 2: Run zero-shot inference using a large language model.

This script:
1. Loads a 70B model with optional quantization
2. Processes zero-shot prompts exercise by exercise
3. Parses model outputs to extract probability predictions
4. Saves predictions in the required format
"""

import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm_zeroshot.utils import (
    load_model_with_quantization,
    parse_model_output,
    setup_gpu_environment,
    MODEL_MAPPING
)


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    device: str = "cuda"
) -> str:
    """
    Run inference on a single prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
    
    Returns:
        Generated text
    """
    # Format prompt using chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096)
    
    # FIXED: Always move inputs to the correct device
    # If using device_map="auto", the model handles device placement
    # But we still need to move inputs to the first device
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Get the first device from the device map
        first_device = next(iter(model.hf_device_map.values()))
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimized parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for faster generation
            # Attention implementation (use flash attention if available)
            attn_implementation="flash_attention_2" if hasattr(model.config, 'attn_implementation') else None,
        )
    
    # Decode only the generated part (excluding input)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def process_exercises(
    data_path: str,
    model,
    tokenizer,
    output_file: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    limit: int = None,
    resume: bool = True
):
    """
    Process all exercises and generate predictions.
    
    Args:
        data_path: Path to zero-shot prompts file
        model: The language model
        tokenizer: The tokenizer
        output_file: Path to save predictions
        max_new_tokens: Maximum tokens to generate per exercise
        temperature: Sampling temperature
        limit: Optional limit on number of exercises
        resume: Whether to resume from existing predictions
    """
    # Load existing predictions if resuming
    completed_exercises = set()
    if resume and os.path.exists(output_file):
        print(f"Resuming from existing predictions: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    instance_id = parts[0]
                    # Extract exercise_id from instance_id (first 10 chars)
                    exercise_id = instance_id[:10]
                    completed_exercises.add(exercise_id)
        print(f"  Found {len(completed_exercises)} completed exercises")
    
    # Load data
    print(f"Loading data from: {data_path}")
    exercises = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            exercise_data = json.loads(line.strip())
            
            # Skip if already completed
            if exercise_data['exercise_id'] not in completed_exercises:
                exercises.append(exercise_data)
    
    print(f"Processing {len(exercises)} exercises (skipped {len(completed_exercises)} completed)")
    
    # Prepare output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open output file in append mode
    mode = 'a' if resume else 'w'
    
    # Statistics
    total_tokens = 0
    successful_parses = 0
    failed_parses = 0
    
    import time
    start_time = time.time()
    
    with open(output_file, mode) as f:
        for idx, exercise_data in enumerate(tqdm(exercises, desc="Running inference")):
            exercise_id = exercise_data['exercise_id']
            prompt = exercise_data['prompt']
            instance_ids = exercise_data['instance_ids']
            num_tokens = exercise_data['num_tokens']
            
            # Run inference
            try:
                ex_start = time.time()
                generated_text = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                ex_time = time.time() - ex_start
                
                # Parse predictions
                predictions = parse_model_output(
                    output_text=generated_text,
                    expected_num_tokens=num_tokens,
                    default_probability=0.5
                )
                
                # Check if parsing was successful
                if len(predictions) == num_tokens:
                    # Count successful parse if we don't use all default values
                    if not all(p == 0.5 for p in predictions):
                        successful_parses += 1
                    else:
                        failed_parses += 1
                else:
                    failed_parses += 1
                
                # Write predictions to file
                for instance_id, pred in zip(instance_ids, predictions):
                    f.write(f"{instance_id} {pred:.6f}\n")
                
                total_tokens += num_tokens
                
                # Print progress every 10 exercises
                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (idx + 1)
                    remaining = avg_time * (len(exercises) - idx - 1)
                    print(f"\n[Progress] {idx+1}/{len(exercises)} exercises | "
                          f"Avg: {avg_time:.1f}s/ex | Last: {ex_time:.1f}s | "
                          f"ETA: {remaining/60:.1f}min")
                
            except Exception as e:
                print(f"\nError processing exercise {exercise_id}: {e}")
                # Write default predictions
                for instance_id in instance_ids:
                    f.write(f"{instance_id} 0.500000\n")
                failed_parses += 1
                total_tokens += num_tokens
            
            # Flush periodically
            if (successful_parses + failed_parses) % 10 == 0:
                f.flush()
    
    # Print statistics
    print(f"\n{'='*80}")
    print("INFERENCE STATISTICS")
    print(f"{'='*80}")
    print(f"Total exercises processed: {len(exercises)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Successful parses: {successful_parses} ({successful_parses/(successful_parses+failed_parses)*100:.1f}%)")
    print(f"Failed parses: {failed_parses} ({failed_parses/(successful_parses+failed_parses)*100:.1f}%)")
    print(f"Predictions saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run zero-shot inference using a large language model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama-3.3-70b-instruct',
        help=f'Model name (options: {", ".join(MODEL_MAPPING.keys())})'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to zero-shot prompts file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save predictions'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size (currently only 1 is supported)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate per exercise'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0.0 for greedy decoding)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        default='int8',
        choices=['int8', 'int4', 'none'],
        help='Quantization method'
    )
    parser.add_argument(
        '--gpu_ids',
        type=int,
        nargs='+',
        default=None,
        help='GPU IDs to use (e.g., --gpu_ids 0 1 2 3)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of exercises to process (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from existing predictions'
    )
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Do not resume, start fresh'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("STEP 2: ZERO-SHOT INFERENCE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_file}")
    print(f"Quantization: {args.quantization}")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    if args.limit:
        print(f"Limit: {args.limit} exercises")
    print(f"Resume: {args.resume}")
    print("="*80)
    print()
    
    # Setup GPU environment
    setup_gpu_environment(args.gpu_ids)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_with_quantization(
        model_name=args.model,
        quantization=args.quantization
    )
    
    print("\nStarting inference...")
    
    # Process exercises
    process_exercises(
        data_path=args.data_path,
        model=model,
        tokenizer=tokenizer,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
        resume=args.resume
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
