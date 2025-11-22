#!/bin/bash
# Zero-shot inference pipeline using 70B LLM for SLAM task

set -e  # Exit on error

MODEL="${1:-llama-3.3-70b-instruct}"
TRACK="${2:-en_es}"
SPLIT="${3:-dev}"
LIMIT="${4:-}"  # Optional: limit number of exercises for testing

echo "=========================================="
echo "Zero-Shot LLM Inference Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Track: $TRACK"
echo "Split: $SPLIT"
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT exercises (testing mode)"
fi
echo "=========================================="
echo

# Step 1: Prepare zero-shot prompts
echo "Step 1: Preparing zero-shot prompts..."
python llm_zeroshot/step1_prepare_zeroshot_prompts.py \
    --track $TRACK \
    --split $SPLIT \
    --data_dir llm_mlp/data \
    --output_dir llm_zeroshot/data \
    ${LIMIT:+--limit $LIMIT}

# Step 2: Run zero-shot inference
echo
echo "Step 2: Running zero-shot inference with $MODEL..."
python llm_zeroshot/step2_zeroshot_inference.py \
    --model $MODEL \
    --data_path llm_zeroshot/data/${TRACK}_${SPLIT}_zeroshot.jsonl \
    --output_file llm_zeroshot/predictions/${MODEL}_${TRACK}_${SPLIT}.pred \
    --quantization int8 \
    --batch_size 1 \
    --temperature 0.0 \
    --resume \
    ${LIMIT:+--limit $LIMIT}

# Step 3: Evaluate predictions
echo
echo "Step 3: Evaluating predictions..."
python llm_zeroshot/step3_evaluate.py \
    --pred llm_zeroshot/predictions/${MODEL}_${TRACK}_${SPLIT}.pred \
    --key dataset/${TRACK}.slam.20190204.${SPLIT}.key \
    --output_dir llm_zeroshot/results

echo
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Results saved to:"
echo "  Predictions: llm_zeroshot/predictions/${MODEL}_${TRACK}_${SPLIT}.pred"
echo "  Metrics: llm_zeroshot/results/evaluation_metrics.txt"
echo "=========================================="
