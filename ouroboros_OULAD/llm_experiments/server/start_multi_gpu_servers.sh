#!/bin/bash

# Multi-GPU Llama Server Starter
# 在多个GPU上启动多个Llama推理服务实例
#
# Usage: bash start_multi_gpu_servers.sh

MODEL_PATH="/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
HOST="0.0.0.0"
DTYPE="bfloat16"
BASE_PORT=8000

# GPU配置：指定要使用的GPU编号
# 例如：GPUS=(0 1 2 3) 表示使用GPU 0,1,2,3
GPUS=(0 1 2 3 0 1 2 3)  # 根据你的GPU数量调整

echo "=========================================="
echo "Starting Multi-GPU Llama Servers"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "GPUs: ${GPUS[@]}"
echo "Base Port: $BASE_PORT"
echo "=========================================="

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 存储进程ID
PIDS=()

# 在每个GPU上启动一个服务器实例
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    PORT=$((BASE_PORT + i))
    LOG_FILE="$LOG_DIR/llama_server_gpu${GPU}_port${PORT}.log"
    
    echo "Starting server on GPU $GPU, Port $PORT..."
    
    # 使用CUDA_VISIBLE_DEVICES指定GPU
    CUDA_VISIBLE_DEVICES=$GPU nohup python llama_server.py \
        --model_path "$MODEL_PATH" \
        --port $PORT \
        --host "$HOST" \
        --device "cuda:0" \
        --dtype "$DTYPE" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    
    echo "  ✓ Server started on GPU $GPU (Port $PORT, PID $PID)"
    echo "    Log: $LOG_FILE"
    
    # 等待几秒，避免同时启动导致资源竞争
    sleep 5
done

echo "=========================================="
echo "All servers started!"
echo "=========================================="
echo "Server URLs:"
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    PORT=$((BASE_PORT + i))
    echo "  GPU $GPU: http://localhost:$PORT"
done
echo ""
echo "Process IDs: ${PIDS[@]}"
echo "=========================================="
echo ""
echo "To stop all servers, run:"
echo "  bash stop_multi_gpu_servers.sh"
echo ""
echo "To view logs:"
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    PORT=$((BASE_PORT + i))
    echo "  tail -f $LOG_DIR/llama_server_gpu${GPU}_port${PORT}.log"
done
echo "=========================================="

# 保存PID到文件，便于后续停止
echo "${PIDS[@]}" > $LOG_DIR/server_pids.txt

