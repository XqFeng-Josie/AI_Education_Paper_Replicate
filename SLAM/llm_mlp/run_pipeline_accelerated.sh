#!/bin/bash
# Accelerated pipeline for 70B models using quantization and Flash Attention

set -e  # Exit on error

MODEL="${1:-llama3-70b}"
TRACK="${2:-en_es}"
ACCELERATION="${3:-int8_flash}"  # Options: int8, int4, flash, int8_flash, int4_flash

echo "==========================================="
echo "LLM+MLP Pipeline (ACCELERATED)"
echo "==========================================="
echo "Model: $MODEL"
echo "Track: $TRACK"
echo "Acceleration: $ACCELERATION"
echo "==========================================="
echo

# Determine acceleration flags
ACCEL_FLAGS="--multi_gpu"
case $ACCELERATION in
    int8)
        ACCEL_FLAGS="$ACCEL_FLAGS --use_int8"
        BATCH_SIZE=16
        echo "Using INT8 quantization (2-3x speedup)"
        ;;
    int4)
        ACCEL_FLAGS="$ACCEL_FLAGS --use_int4"
        BATCH_SIZE=32
        echo "Using INT4 quantization (4-6x speedup)"
        ;;
    flash)
        ACCEL_FLAGS="$ACCEL_FLAGS --use_flash_attn"
        BATCH_SIZE=8
        echo "Using Flash Attention 2 (1.5-2x speedup)"
        ;;
    int8_flash)
        ACCEL_FLAGS="$ACCEL_FLAGS --use_int8 --use_flash_attn"
        BATCH_SIZE=24
        echo "Using INT8 + Flash Attention 2 (3-5x speedup)"
        ;;
    int4_flash)
        ACCEL_FLAGS="$ACCEL_FLAGS --use_int4 --use_flash_attn"
        BATCH_SIZE=48
        echo "Using INT4 + Flash Attention 2 (6-10x speedup)"
        ;;
    *)
        BATCH_SIZE=8
        echo "No acceleration (baseline)"
        ;;
esac

echo "Batch size: $BATCH_SIZE"
echo "==========================================="
echo

# Step 1: Prepare data
echo "Step 1: Preparing data with exercise context..."
python llm_mlp/step1_prepare_data.py \
    --track $TRACK \
    --output_dir llm_mlp/data

# Step 2a: Extract dev embeddings (ACCELERATED)
echo
echo "Step 2a: Extracting dev token embeddings (ACCELERATED)..."
python llm_mlp/step2_extract_embeddings.py \
    --model $MODEL \
    --split dev \
    --track $TRACK \
    --batch_size $BATCH_SIZE \
    --data_dir llm_mlp/data \
    $ACCEL_FLAGS \
    --resume

# Step 2b: Extract test embeddings (ACCELERATED)
echo
echo "Step 2b: Extracting test token embeddings (ACCELERATED)..."
python llm_mlp/step2_extract_embeddings.py \
    --model $MODEL \
    --split test \
    --track $TRACK \
    --batch_size $BATCH_SIZE \
    --data_dir llm_mlp/data \
    $ACCEL_FLAGS \
    --resume

# Step 3: Train MLP
echo
echo "Step 3: Training MLP on token embeddings..."
python llm_mlp/step3_train_mlp.py \
    --embeddings_path llm_mlp/embeddings/${TRACK}_dev_${MODEL}_token_embeddings.pt \
    --model_name $MODEL \
    --track $TRACK \
    --num_epochs 50 \
    --batch_size 128 \
    --learning_rate 5e-4

# Step 4: Inference
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
echo "==========================================="
echo "Pipeline completed successfully!"
echo "==========================================="
