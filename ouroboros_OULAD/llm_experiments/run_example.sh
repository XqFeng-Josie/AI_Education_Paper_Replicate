#!/bin/bash

# Example script to run unified agent experiments
# Supports both OpenRouter and Local Llama

echo "=========================================="
echo "Unified Agent Experiment Runner"
echo "=========================================="

# Clean cache to ensure fresh data generation
CACHE_DIR="../selflearner/data_load/data"
echo "Cleaning HDF5 cache..."
if [ -f "$CACHE_DIR/selflearner.h5" ]; then
    rm -f "$CACHE_DIR/selflearner.h5"
    echo "  ✓ Removed selflearner.h5"
fi
if [ -f "$CACHE_DIR/oulad.h5" ]; then
    rm -f "$CACHE_DIR/oulad.h5"
    echo "  ✓ Removed oulad.h5"
fi
echo "Cache cleaned. Data will be regenerated."
echo "=========================================="
# generate data
cd /projects/bdns/xfeng4/1/AI_Education_Paper_Replicate/ouroboros_OULAD/
python convert_csv_to_h5.py

# Configuration
MODE="few_shot"
NUM_FEW_SHOT=5
MODULE="EEE"
PRESENTATION="2014J"
ASSESSMENT_NAME="TMA 1"
DAYS_TO_CUTOFF=3
MAX_STUDENTS=100  # Set to empty for all students
SAVE_INTERVAL=1
NUM_WORKERS=8
# Parse command line arguments
PROVIDER=${1:-multi_local}  # Default to multi_local (multi-GPU)

echo "Provider: $PROVIDER"
echo "Mode: $MODE"
echo "Few-shot examples: $NUM_FEW_SHOT"
echo "Task: $MODULE / $PRESENTATION / $ASSESSMENT_NAME / days=$DAYS_TO_CUTOFF"
echo "Max students: ${MAX_STUDENTS:-all}"
echo "=========================================="

cd llm_experiments/
# Build base command
CMD="python unified_agent_main.py \
    --provider $PROVIDER \
    --mode $MODE \
    --num_few_shot $NUM_FEW_SHOT \
    --module $MODULE \
    --presentation $PRESENTATION \
    --assessment_name \"$ASSESSMENT_NAME\" \
    --days_to_cutoff $DAYS_TO_CUTOFF \
    --save_interval $SAVE_INTERVAL \
    --num_workers $NUM_WORKERS "

# Add max_students if specified
if [ ! -z "$MAX_STUDENTS" ]; then
    CMD="$CMD --max_students $MAX_STUDENTS"
fi

# Provider-specific settings
if [ "$PROVIDER" = "openrouter" ]; then
    echo "Using OpenRouter API"
    echo "Model: meta-llama/llama-3.1-70b-instruct"
    echo "Please ensure OPENROUTER_API_KEY is set"
    
    CMD="$CMD \
        --model meta-llama/llama-3.1-70b-instruct \
        --max_retries 3 \
        --retry_delay 60"
    
elif [ "$PROVIDER" = "local" ]; then
    echo "Using Local Llama Server (Single GPU)"
    echo "Server URL: http://localhost:8001"
    echo "Please ensure local server is running:"
    echo "  cd server && bash start_llama_server.sh"
    
    CMD="$CMD \
        --server_url http://localhost:8001 \
        --max_retries 3 \
        --retry_delay 30 \
        --num_workers 1"

elif [ "$PROVIDER" = "multi_local" ]; then
    echo "Using Multi-GPU Local Llama Servers"
    echo "Servers: http://localhost:8000-8007 (8 GPUs)"
    echo "Please ensure all servers are running:"
    echo "  cd server && bash start_multi_gpu_servers.sh"
    echo ""
    echo "Auto-detecting available servers..."
    
    CMD="$CMD \
        --base_port 8000 \
        --num_servers 8 \
        --load_balance_strategy round_robin \
        --max_retries 3 \
        --retry_delay 30"
else
    echo "Error: Unknown provider '$PROVIDER'"
    echo "Usage: bash run_example.sh [openrouter|local|multi_local]"
    exit 1
fi

echo "=========================================="
echo "Running command:"
echo "$CMD"
echo "=========================================="

# Execute
eval $CMD

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment completed successfully"
else
    echo "❌ Experiment failed with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

