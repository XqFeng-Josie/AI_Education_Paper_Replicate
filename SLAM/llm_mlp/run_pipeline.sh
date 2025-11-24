#!/bin/bash
# End-to-end pipeline for LLM+MLP training on SLAM task (token-level with exercise context)

set -e  # Exit on error

MODEL="${1:-llama-3.1-8b}"
TRACK="${2:-en_es}"

echo "=========================================="
echo "LLM+MLP Pipeline (Token-Level)"
echo "=========================================="
echo "Model: $MODEL"
echo "Track: $TRACK"
echo "Mode: Token-level with exercise context"
echo "=========================================="
echo

# Step 1: Prepare exercise-level data (with exercise context in prompts)
echo "Step 1: Preparing data with exercise context..."
python llm_mlp/step1_prepare_data.py \
    --track $TRACK \
    --output_dir llm_mlp/data

# Step 2: Extract embeddings for dev (token-level, no aggregation)
echo
echo "Step 2a: Extracting dev token embeddings..."
python llm_mlp/step2_extract_embeddings.py \
    --model $MODEL \
    --split dev \
    --track $TRACK \
    --batch_size 32 \
    --data_dir llm_mlp/data \
    --resume

# Step 2b: Extract embeddings for test (token-level, no aggregation)
echo
echo "Step 2b: Extracting test token embeddings..."
python llm_mlp/step2_extract_embeddings.py \
    --model $MODEL \
    --split test \
    --track $TRACK \
    --batch_size 32 \
    --data_dir llm_mlp/data \
    --resume

# Step 3: Train MLP on token embeddings
echo
echo "Step 3: Training MLP on token embeddings..."
python llm_mlp/step3_train_mlp.py \
    --embeddings_path llm_mlp/embeddings/${TRACK}_dev_${MODEL}_token_embeddings.pt \
    --model_name $MODEL \
    --track $TRACK \
    --num_epochs 50 \
    --batch_size 128 \
    --learning_rate 5e-4

# Step 4: Inference on token embeddings
echo
echo "Step 4: Running inference..."
python llm_mlp/step4_inference.py \
    --model_dir llm_mlp/models/${MODEL}_${TRACK} \
    --embeddings_path llm_mlp/embeddings/${TRACK}_test_${MODEL}_token_embeddings.pt \
    --output_file llm_mlp/predictions/${MODEL}_${TRACK}_test.pred

# Step 5: Evaluation
echo
echo "Step 5: Evaluating..."
python llm_mlp/step5_evaluate.py \
    --pred llm_mlp/predictions/${MODEL}_${TRACK}_test.pred \
    --key dataset/${TRACK}.slam.20190204.test.key

echo
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
