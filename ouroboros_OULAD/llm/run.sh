#!/bin/bash

python -m llm.run_paper_replication \
    --llm_config config/llm_config_multi_gpu.yaml \
    --n_workers 1 \
    --output_dir outputs/llm_config_multi_gpu_1_worker

# echo "Running paper replication with multi-GPU configuration..."
# python -m llm.run_paper_replication \
#     --llm_config config/llm_config_multi_gpu.yaml \
#     --n_workers 2

# echo "Running paper replication with multi-GPU configuration..."
# python -m llm.run_paper_replication \
#     --llm_config config/llm_config_multi_gpu.yaml \
#     --n_workers 3