# Zero-Shot LLM Inference for SLAM

This directory contains the implementation of zero-shot inference for the SLAM (Second Language Acquisition Modeling) task using large language models (70B+).

## Overview

Unlike the `llm_mlp/` approach which uses LLM embeddings + MLP training, this zero-shot method:
- ✅ Directly uses the 70B model's generation capabilities
- ✅ No training required - pure zero-shot inference
- ✅ Outputs probability predictions via structured JSON
- ✅ Leverages full exercise context for each prediction

## Quick Start

### Full Pipeline (Recommended)

```bash
# Run complete pipeline on dev set
./llm_zeroshot/run_zeroshot_pipeline.sh llama-3.3-70b-instruct en_es dev

# Test on first 10 exercises
./llm_zeroshot/run_zeroshot_pipeline.sh llama-3.3-70b-instruct en_es dev 10
```

### Step-by-Step Execution

```bash
# Step 1: Prepare zero-shot prompts
python llm_zeroshot/step1_prepare_zeroshot_prompts.py \
    --track en_es \
    --split dev \
    --data_dir llm_mlp/data \
    --output_dir llm_zeroshot/data

# Step 2: Run inference
python llm_zeroshot/step2_zeroshot_inference.py \
    --model llama-3.3-70b-instruct \
    --data_path llm_zeroshot/data/en_es_dev_zeroshot.jsonl \
    --output_file llm_zeroshot/predictions/llama-3.3-70b-instruct_en_es_dev.pred \
    --quantization int8 \
    --resume

# Step 3: Evaluate
python llm_zeroshot/step3_evaluate.py \
    --pred llm_zeroshot/predictions/llama-3.3-70b-instruct_en_es_dev.pred \
    --key dataset/en_es.slam.20190204.dev.key \
    --output_dir llm_zeroshot/results
```

## Prompt Design

The prompt follows this structure:

```
You are an AI assistant helping predict language learning outcomes. 
This is a second language acquisition task where learners (native Spanish 
speakers) are learning English through Duolingo exercises.

**Learner Profile:**
Learner's history: 7.6 days of practice, 1015 attempts, 6.1% correct.
Performance by format: listen=6.0%, reverse_translate=9.7%, reverse_tap=5.2%.

**Exercise:**
The learner is presented with an English phrase: "They work in education"

**Task:**
Predict the probability (0.0 to 1.0) that the learner will answer each 
token correctly...

**Tokens to predict:**
1. 'They' - POS: PRON, Format: reverse_tap, Morphology: Case,Number,...
2. 'work' - POS: NOUN, Format: reverse_tap, Morphology: VerbForm,...
...

**Output Format:**
{"predictions": [0.85, 0.62, 0.91, 0.45]}
```

## Computational Requirements

### GPU Requirements
- **Minimum**: 1x A100 (40GB) with int8 quantization
- **Recommended**: 1x A100 (80GB) or 2x A100 (40GB) with model parallelism
- **Alternative**: int4 quantization for GPUs with 24GB+ VRAM

### Performance Expectations
- **Inference speed**: ~5-10 exercises/minute (depends on GPU)
- **Full dev set**: ~46k exercises ≈ 80-150 hours on single A100
- **Recommended**: Test on small subset first (`--limit 100`)

## Key Features

### Resume Capability
The inference script supports resuming from partial results:
```bash
# Will skip already processed exercises
python llm_zeroshot/step2_zeroshot_inference.py ... --resume
```

### Quantization Support
Reduce memory footprint with quantization:
- `--quantization int8`: ~40GB VRAM (recommended)
- `--quantization int4`: ~20GB VRAM (faster but may reduce quality)
- `--quantization none`: ~140GB VRAM (requires multi-GPU)

### Testing Mode
Process limited exercises for quick testing:
```bash
./llm_zeroshot/run_zeroshot_pipeline.sh llama-3.3-70b-instruct en_es dev 100
```

## Output Format

Predictions are saved in the same format as the baseline:
```
++1ogDCE0101 0.234567
++1ogDCE0102 0.876543
...
```

Compatible with existing evaluation scripts.

## Supported Models

- `llama-3.3-70b-instruct` (recommended)
- `llama-3.1-70b-instruct`
- `llama-3.1-8b` (for comparison/testing)

Add more models in `utils.py` → `MODEL_MAPPING`

## Troubleshooting

### Out of Memory
- Use `--quantization int4` instead of `int8`
- Reduce `--max_new_tokens` (default: 100)
- Ensure no other processes are using GPU

### Slow Inference
- Expected behavior for 70B models
- Use `--limit` to test on smaller subset first
- Consider multi-GPU setup if available

### Parse Failures
The script handles malformed JSON outputs gracefully:
- Attempts regex fallback to extract numbers
- Uses default probability (0.5) if parsing fails
- Reports parse success rate in statistics

## Comparison with LLM+MLP

| Aspect | Zero-Shot | LLM+MLP |
|--------|-----------|---------|
| Training | None | Required |
| Inference Speed | Slow (~10 ex/min) | Fast (batch processing) |
| Model Size | 70B+ | 7-8B + small MLP |
| Explainability | High (can add reasoning) | Low (black box) |
| Adaptability | Immediate | Requires retraining |
| Compute | High (GPU intensive) | Medium (one-time training) |

## Directory Structure

```
llm_zeroshot/
├── __init__.py
├── README.md                           # This file
├── utils.py                            # Shared utilities
├── step1_prepare_zeroshot_prompts.py   # Prompt preparation
├── step2_zeroshot_inference.py         # Inference engine
├── step3_evaluate.py                   # Evaluation
├── run_zeroshot_pipeline.sh            # Complete pipeline
├── data/                               # Generated prompts
├── predictions/                        # Model outputs
└── results/                            # Evaluation results
```
