#!/bin/bash

# Stop all Llama servers
# Usage: bash stop_multi_gpu_servers.sh

LOG_DIR="./logs"
PID_FILE="$LOG_DIR/server_pids.txt"

echo "=========================================="
echo "Stopping Multi-GPU Llama Servers"
echo "=========================================="

if [ -f "$PID_FILE" ]; then
    PIDS=$(cat $PID_FILE)
    echo "Found PIDs: $PIDS"
    
    for PID in $PIDS; do
        if kill -0 $PID 2>/dev/null; then
            echo "Stopping process $PID..."
            kill $PID
        else
            echo "Process $PID not running"
        fi
    done
    
    # 等待进程结束
    sleep 2
    
    # 强制杀死仍在运行的进程
    for PID in $PIDS; do
        if kill -0 $PID 2>/dev/null; then
            echo "Force killing process $PID..."
            kill -9 $PID
        fi
    done
    
    rm $PID_FILE
    echo "✓ All servers stopped"
else
    echo "No PID file found. Searching for running servers..."
    
    # 查找并杀死所有llama_server进程
    pkill -f "llama_server.py"
    
    if [ $? -eq 0 ]; then
        echo "✓ Stopped running llama_server processes"
    else
        echo "No running servers found"
    fi
fi

echo "=========================================="

