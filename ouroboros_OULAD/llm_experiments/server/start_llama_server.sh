#!/bin/bash

# Script to start Llama inference server
# Usage: bash start_llama_server.sh

MODEL_PATH="/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PORT=8001
HOST="0.0.0.0"
DEVICE="auto"
DTYPE="bfloat16"

echo "=========================================="
echo "Starting Llama Inference Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Server: $HOST:$PORT"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "=========================================="

# Activate conda environment if needed
# conda activate your_env

# Start server
python llama_server.py \
    --model_path "$MODEL_PATH" \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

