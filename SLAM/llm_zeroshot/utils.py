"""
Utility functions for zero-shot LLM inference pipeline.
"""

import json
import re
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_MAPPING = {
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def create_zeroshot_prompt(
    exercise_data: Dict[str, Any],
    include_examples: bool = False
) -> str:
    """
    Create a zero-shot prompt for exercise-level token prediction.
    
    Args:
        exercise_data: Dictionary containing exercise information from prepared data
        include_examples: Whether to include few-shot examples (not implemented yet)
    
    Returns:
        Formatted prompt string for zero-shot inference
    """
    # Extract exercise information
    prompts = exercise_data['prompts']
    num_tokens = exercise_data['num_tokens']
    
    # Parse learner history and exercise info from the first token's prompt
    # The prompts are in format: "Learner's history: X days of practice, Y attempts, Z% correct..."
    first_prompt = prompts[0]
    
    # Extract learner history (everything before "Exercise (all tokens):")
    if "Exercise (all tokens):" in first_prompt:
        learner_history = first_prompt.split("Exercise (all tokens):")[0].strip()
        exercise_text_match = re.search(r'Exercise \(all tokens\): "(.*?)"', first_prompt)
        exercise_text = exercise_text_match.group(1) if exercise_text_match else "Unknown"
    else:
        # Fallback: just use the first part
        learner_history = first_prompt[:200] + "..."
        exercise_text = "Unknown"
    
    # Extract token information from each prompt
    token_infos = []
    for idx, prompt in enumerate(prompts):
        # Extract token details using regex
        # Pattern: Current token #X/Y: 'TOKEN' (POS: ..., Format: ..., Morphology: ..., DepLabel: ...)
        # Note: Morphology can contain commas, so we match until ", DepLabel:"
        match = re.search(
            r"Current token #\d+/\d+: '([^']+)' \(POS: ([^,]+), Format: ([^,]+), Morphology: (.+?), DepLabel: ([^)]+)\)",
            prompt
        )
        
        if match:
            token, pos, fmt, morph, dep = match.groups()
            token_infos.append({
                'token': token,
                'pos': pos,
                'format': fmt,
                'morphology': morph,
                'dep_label': dep
            })
        else:
            # Fallback: try to at least extract the token
            token_match = re.search(r"Current token #\d+/\d+: '([^']+)'", prompt)
            if token_match:
                token = token_match.group(1)
            else:
                token = f"token_{idx+1}"
            
            token_infos.append({
                'token': token,
                'pos': 'UNKNOWN',
                'format': 'unknown',
                'morphology': 'unknown',
                'dep_label': 'unknown'
            })
    
    # Build the zero-shot prompt
    prompt = f"""You are an AI assistant helping predict language learning outcomes. This is a second language acquisition task where learners (native Spanish speakers) are learning English through Duolingo exercises.

**Learner Profile:**
{learner_history}

**Exercise:**
The learner is presented with an English phrase: "{exercise_text}"

**Task:**
Predict the probability (0.0 to 1.0) that the learner will answer each token correctly, where:
- 0.0 = definitely incorrect
- 1.0 = definitely correct
- Values in between represent uncertainty

**Tokens to predict:**
"""
    
    for idx, info in enumerate(token_infos, 1):
        prompt += f"{idx}. '{info['token']}' - POS: {info['pos']}, Format: {info['format']}, Morphology: {info['morphology']}, Dependency: {info['dep_label']}\n"
    
    prompt += f"""
**Output Format:**
Respond with ONLY a valid JSON object in this exact format:
{{"predictions": [p1, p2, ..., p{num_tokens}]}}

Where each p_i is a probability value between 0.0 and 1.0 representing the likelihood that the learner will answer token i correctly.

Example output: {{"predictions": [0.85, 0.62, 0.91, 0.45]}}

Do not include any explanation or additional text, only the JSON object."""

    return prompt


def parse_model_output(
    output_text: str,
    expected_num_tokens: int,
    default_probability: float = 0.5
) -> List[float]:
    """
    Parse model output to extract probability predictions.
    
    Args:
        output_text: Raw text output from the model
        expected_num_tokens: Expected number of token predictions
        default_probability: Default value to use if parsing fails
    
    Returns:
        List of probability predictions (length = expected_num_tokens)
    """
    try:
        # Try to find JSON object in the output
        # Look for {"predictions": [...]}
        json_match = re.search(r'\{[^}]*"predictions"[^}]*\[[^\]]*\][^}]*\}', output_text)
        
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            predictions = data.get('predictions', [])
            
            # Validate predictions
            if len(predictions) == expected_num_tokens:
                # Ensure all values are floats between 0 and 1
                validated_preds = []
                for p in predictions:
                    try:
                        p_float = float(p)
                        # Clamp to [0, 1] range
                        p_float = max(0.0, min(1.0, p_float))
                        validated_preds.append(p_float)
                    except (ValueError, TypeError):
                        validated_preds.append(default_probability)
                
                return validated_preds
            else:
                print(f"Warning: Expected {expected_num_tokens} predictions, got {len(predictions)}")
        
        # If JSON parsing failed, try to extract numbers from the output
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', output_text)
        if len(numbers) >= expected_num_tokens:
            predictions = [float(n) for n in numbers[:expected_num_tokens]]
            return [max(0.0, min(1.0, p)) for p in predictions]
        
    except Exception as e:
        print(f"Error parsing model output: {e}")
        print(f"Output text: {output_text[:200]}...")
    
    # Fallback: return default probabilities
    print(f"Warning: Using default probabilities ({default_probability}) for all {expected_num_tokens} tokens")
    return [default_probability] * expected_num_tokens


def load_model_with_quantization(
    model_name: str,
    quantization: str = "int8",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16
) -> tuple:
    """
    Load a large language model with optional quantization.
    
    Args:
        model_name: Model name or path
        quantization: Quantization method ("int8", "int4", "none")
        device_map: Device mapping strategy
        torch_dtype: Torch data type for non-quantized model
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Get model path from mapping
    model_path = MODEL_MAPPING.get(model_name, model_name)
    
    print(f"Loading model: {model_path}")
    print(f"Quantization: {quantization}")
    
    # Configure quantization
    quantization_config = None
    if quantization == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    elif quantization == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    return model, tokenizer


def setup_gpu_environment(gpu_ids: Optional[List[int]] = None):
    """
    Setup GPU environment for multi-GPU inference.
    
    Args:
        gpu_ids: List of GPU IDs to use (None = use all available)
    """
    if gpu_ids is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        print(f"Using GPUs: {gpu_ids}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Warning: CUDA not available, using CPU")
