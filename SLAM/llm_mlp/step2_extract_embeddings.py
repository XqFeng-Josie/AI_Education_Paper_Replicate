"""
Step 2: Extract LLM embeddings at token level.

This script:
1. Loads prepared exercise-grouped JSONL data
2. Extracts embeddings for each token individually (no aggregation)
3. Each token keeps exercise context in its prompt
4. Saves token-level embeddings to .pt files
"""

import argparse
import json
import os
import sys
from typing import List, Union

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from accelerate import infer_auto_device_map, dispatch_model

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm_mlp.utils import MODEL_MAPPING


def extract_token_embeddings(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    exercises: List[dict],
    batch_size: int,
    max_length: int,
    device: Union[torch.device, str],
    checkpoint_path: str = None,
    resume_from: int = 0,
    show_progress: bool = True,
    multi_gpu: bool = False
) -> tuple:
    """
    Extract embeddings for each token individually (no aggregation).
    Each token keeps exercise context in its prompt but gets its own embedding.
    
    Args:
        model: Frozen LLM model
        tokenizer: Tokenizer for the model
        exercises: List of exercise dicts with 'prompts' field
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to use
        checkpoint_path: Path to save checkpoints for resume
        resume_from: Index to resume from (0 = start from beginning)
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (all_embeddings, all_labels, all_instance_ids)
        - all_embeddings: tensor [N_tokens, hidden_dim]
        - all_labels: list of token labels
        - all_instance_ids: list of instance IDs
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_instance_ids = []
    
    # Load checkpoint if resuming
    if resume_from > 0 and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        all_embeddings = checkpoint['embeddings_list']
        all_labels = checkpoint['labels_list']
        all_instance_ids = checkpoint['instance_ids_list']
        print(f"  Loaded {len(all_embeddings)} previously processed tokens")
        exercises = exercises[resume_from:]
        print(f"  Continuing from exercise {resume_from}...")
    
    progress_bar = tqdm(
        enumerate(exercises, start=resume_from),
        desc="Extracting token embeddings",
        disable=not show_progress,
        initial=resume_from,
        total=len(exercises) + resume_from
    )
    
    checkpoint_interval = 1000  # Save checkpoint every 1000 exercises
    
    with torch.no_grad():
        for ex_idx, exercise in progress_bar:
            token_prompts = exercise['prompts']
            token_labels = exercise['token_labels']
            instance_ids = exercise['instance_ids']
            
            # Process tokens in batches
            for i in range(0, len(token_prompts), batch_size):
                batch_prompts = token_prompts[i:i+batch_size]
                
                # Tokenize
                encoded = tokenizer(
                    batch_prompts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # For multi-GPU, send to the first device of the model
                if multi_gpu:
                    # Get the device of the first model parameter
                    first_device = next(model.parameters()).device
                    input_ids = encoded['input_ids'].to(first_device)
                    attention_mask = encoded['attention_mask'].to(first_device)
                else:
                    input_ids = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)
                
                # Extract embeddings
                use_amp = torch.cuda.is_available() and (not multi_gpu or str(device) != 'cpu')
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                    
                    # Mean pooling over sequence for each token
                    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
                    masked_hidden = hidden_states * mask
                    summed = masked_hidden.sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1.0)
                    pooled = summed / counts
                
                # Convert to float32 and store each token separately
                batch_embeddings = pooled.float().cpu()
                for j in range(len(batch_embeddings)):
                    all_embeddings.append(batch_embeddings[j].unsqueeze(0))
                    all_labels.append(token_labels[i + j])
                    all_instance_ids.append(instance_ids[i + j])
            
            # Save checkpoint periodically
            if checkpoint_path and (ex_idx + 1) % checkpoint_interval == 0:
                checkpoint = {
                    'embeddings_list': all_embeddings,
                    'labels_list': all_labels,
                    'instance_ids_list': all_instance_ids,
                    'last_index': ex_idx + 1
                }
                torch.save(checkpoint, checkpoint_path)
                progress_bar.set_postfix({'checkpoint': f'saved at {ex_idx + 1}', 'tokens': len(all_embeddings)})
    
    # Save final checkpoint
    if checkpoint_path:
        checkpoint = {
            'embeddings_list': all_embeddings,
            'labels_list': all_labels,
            'instance_ids_list': all_instance_ids,
            'last_index': len(exercises) + resume_from
        }
        torch.save(checkpoint, checkpoint_path)
    
    # Concatenate all token embeddings
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    return all_embeddings_tensor, all_labels, all_instance_ids


def main():
    parser = argparse.ArgumentParser(
        description='Extract token-level LLM embeddings for SLAM data'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_MAPPING.keys()),
        help='Model to use for embedding extraction'
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        help='Data split to process'
    )
    parser.add_argument(
        '--track',
        type=str,
        default='en_es',
        choices=['en_es', 'es_en', 'fr_en'],
        help='Dataset track'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='llm_mlp/data',
        help='Directory containing prepared JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='llm_mlp/embeddings',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for token embedding extraction'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--max_exercises',
        type=int,
        default=None,
        help='Maximum number of exercises to process (for testing)'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU ID to use (default: 0)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint if available'
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='Use multiple GPUs with device_map="auto" for large models (e.g., 70B). Ignores --gpu_id.'
    )
    parser.add_argument(
        '--use_int8',
        action='store_true',
        help='Use INT8 quantization for 2-3x speedup with <1%% accuracy loss'
    )
    parser.add_argument(
        '--use_int4',
        action='store_true',
        help='Use INT4 quantization for 4-6x speedup with 1-3%% accuracy loss'
    )
    parser.add_argument(
        '--use_flash_attn',
        action='store_true',
        help='Use Flash Attention 2 for 1.5-2x speedup with 0%% accuracy loss'
    )
    parser.add_argument(
        '--use_bfloat16',
        action='store_true',
        help='Use BFloat16 instead of Float16 (for A100/H100 GPUs)'
    )
    
    args = parser.parse_args()
    
    # Validate quantization args
    if args.use_int8 and args.use_int4:
        raise ValueError("Cannot use both INT8 and INT4 quantization. Choose one.")
    
    # Setup paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', args.data_dir)
    jsonl_path = os.path.join(data_dir, f'{args.track}_{args.split}_exercise.jsonl')

    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f'{args.track}_{args.split}_{args.model}_token_embeddings.pt'
    output_path = os.path.join(output_dir, output_filename)
    
    # Setup checkpoint path
    checkpoint_path = output_path.replace('.pt', '_checkpoint.pt')
    resume_from = 0
    
    # Check if we should resume
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        resume_from = checkpoint['last_index']
        print(f"Found checkpoint at exercise {resume_from}")
    
    # Check if final embeddings already exist
    if os.path.exists(output_path) and not args.resume:
        print(f"Embeddings already exist at: {output_path}")
        print("Skipping extraction. Delete the file to re-extract, or use --resume to continue.")
        return
    
    print("="*80)
    print("STEP 2: TOKEN-LEVEL LLM EMBEDDING EXTRACTION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Track: {args.track}")
    print(f"Input file: {jsonl_path}")
    print(f"Mode: Token-level (with exercise context in prompts)")
    
    # Load data
    print("\n[1/4] Loading prepared exercise data...")
    exercises = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            exercises.append(json.loads(line))
    
    print(f"  Loaded {len(exercises)} exercises")
    total_tokens = sum(ex['num_tokens'] for ex in exercises)
    print(f"  Total tokens: {total_tokens}")
    
    # Apply max_exercises if specified (for testing)
    if args.max_exercises is not None:
        print(f"  Limiting to {args.max_exercises} exercises for testing")
        exercises = exercises[:args.max_exercises]
        total_tokens = sum(ex['num_tokens'] for ex in exercises)
        print(f"  Tokens in subset: {total_tokens}")
    
    # Setup device and multi-GPU strategy
    if args.multi_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU mode requires CUDA, but no CUDA devices found.")
        
        num_gpus = torch.cuda.device_count()
        print(f"\n[INFO] Multi-GPU mode enabled: Using {num_gpus} GPUs")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        device = "auto"  # Will use device_map
    else:
        if torch.cuda.is_available():
            if args.gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {args.gpu_id} not available. Using GPU 0.")
                args.gpu_id = 0
            torch.cuda.set_device(args.gpu_id)
            device = torch.device(f'cuda:{args.gpu_id}')
        else:
            device = torch.device('cpu')
    
    print(f"\n[2/4] Loading model: {args.model}")
    if not args.multi_gpu:
        print(f"  Device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    
    # Load model and tokenizer
    model_path = MODEL_MAPPING[args.model]
    print(f"  Path: {model_path}")
    
    # Check if it's a local path
    is_local_path = os.path.exists(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype
    if args.use_bfloat16:
        torch_dtype = torch.bfloat16
        print("  Using BFloat16 precision")
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Prepare quantization config
    quantization_config = None
    if args.use_int8 or args.use_int4:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "Quantization requires 'bitsandbytes'. Install with: pip install bitsandbytes"
            )
        
        if args.use_int8:
            print("  Using INT8 quantization (2-3x speedup expected)")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        elif args.use_int4:
            print("  Using INT4 quantization (4-6x speedup expected)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    
    # Flash Attention config with availability check
    attn_implementation = None
    if args.use_flash_attn:
        # Check if Flash Attention is available
        flash_attn_available = False
        try:
            import flash_attn
            flash_attn_available = True
            print("  Using Flash Attention 2 (1.5-2x speedup expected)")
            attn_implementation = "flash_attention_2"
        except ImportError as e:
            print("  ⚠️  WARNING: Flash Attention not available")
            if "GLIBC" in str(e):
                print("     Reason: System GLIBC version too old (requires GLIBC 2.32+)")
                print("     Solution: Use quantization only (--use_int8 or --use_int4)")
            else:
                print(f"     Reason: {str(e)}")
                print("     Install with: pip install flash-attn --no-build-isolation")
            print("  Continuing WITHOUT Flash Attention...")
    
    # Load model with appropriate device strategy
    if args.multi_gpu:
        print("  Loading model with device_map='auto' for multi-GPU...")
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "local_files_only": is_local_path,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        
        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        
        print("  Model distributed across GPUs:")
        # Print device map
        if hasattr(model, 'hf_device_map'):
            device_summary = {}
            for name, dev in model.hf_device_map.items():
                dev_str = str(dev)
                if dev_str not in device_summary:
                    device_summary[dev_str] = 0
                device_summary[dev_str] += 1
            for dev, count in sorted(device_summary.items()):
                print(f"    {dev}: {count} modules")
    else:
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "local_files_only": is_local_path
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"  # Quantization requires device_map
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        
        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        
        # Only move to device if not using quantization
        if quantization_config is None:
            model.to(device)
    
    model.eval()
    
    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_size = model.config.hidden_size
    print(f"  Hidden size: {hidden_size}")
    
    # Extract embeddings (token-level, not aggregated)
    print(f"\n[3/4] Extracting token embeddings (batch_size={args.batch_size})...")
    if resume_from > 0:
        print(f"  Resuming from exercise {resume_from}")
    
    embeddings, labels, instance_ids = extract_token_embeddings(
        model=model,
        tokenizer=tokenizer,
        exercises=exercises,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
        show_progress=True,
        multi_gpu=args.multi_gpu
    )
    
    print(f"  Extracted embeddings shape: {embeddings.shape}")
    print(f"  Total tokens: {len(labels)}")
    
    # Save embeddings
    print(f"\n[4/4] Saving embeddings...")
    save_data = {
        'embeddings': embeddings,
        'labels': torch.tensor(labels, dtype=torch.long),
        'instance_ids': instance_ids,
        'model_name': args.model,
        'hidden_size': hidden_size,
        'track': args.track,
        'split': args.split,
        'mode': 'token-level',
    }
    
    torch.save(save_data, output_path)
    print(f"  Saved to: {output_path}")
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Removed checkpoint file")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Mode: Token-level (exercise context preserved in prompts)")
    print(f"Tokens processed: {len(embeddings)}")
    print(f"Embedding dimension: {hidden_size}")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print("="*80)


if __name__ == '__main__':
    main()
