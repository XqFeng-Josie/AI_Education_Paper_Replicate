#!/bin/bash
# Optimized zero-shot inference with reduced token generation

set -e

MODEL="${1:-llama-3.3-70b-instruct}"
TRACK="${2:-en_es}"  
SPLIT="${3:-dev}"
LIMIT="${4:-}"

echo "=========================================="
echo "OPTIMIZED Zero-Shot Inference"
echo "=========================================="
echo "Model: $MODEL"
echo "Track: $TRACK"
echo "Split: $SPLIT"
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT exercises"
fi
echo "Optimizations:"
echo "  - Reduced max_new_tokens to 50 (JSON output only)"
echo "  - Device placement fixed (CPU→GPU)"
echo "  - KV cache enabled"
echo "  - Progress reporting every 10 exercises"
echo "=========================================="
echo

# Ensure data is prepared
if [ ! -f "llm_zeroshot/data/${TRACK}_${SPLIT}_zeroshot.jsonl" ]; then
    echo "Preparing prompts first..."
    python llm_zeroshot/step1_prepare_zeroshot_prompts.py \
        --track $TRACK \
        --split $SPLIT \
        --data_dir llm_mlp/data \
        --output_dir llm_zeroshot/data \
        ${LIMIT:+--limit $LIMIT}
    echo
fi

# Run optimized inference
echo "Running optimized inference..."
python llm_zeroshot/step2_zeroshot_inference.py \
    --model $MODEL \
    --data_path llm_zeroshot/data/${TRACK}_${SPLIT}_zeroshot.jsonl \
    --output_file llm_zeroshot/predictions/${MODEL}_${TRACK}_${SPLIT}_optimized.pred \
    --quantization int8 \
    --batch_size 1 \
    --max_new_tokens 50 \
    --temperature 0.0 \
    --resume \
    ${LIMIT:+--limit $LIMIT}

echo
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
