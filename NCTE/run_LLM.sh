#!/bin/bash

### Use LLM to classify the data
source venv/bin/activate

echo "Running llama-3.1-8b-instruct with few_shot..."
python run_llm_classifier.py \
--model_name llama-3.1-8b-instruct \
--task_name all \
--mode few_shot \
--n_shots 5

echo "Running llama-3.1-8b-instruct with zero_shot..."
python run_llm_classifier.py \
--model_name llama-3.1-8b-instruct \
--task_name all \
--mode zero_shot

echo "Running mistral-7b-instruct-v0.3 with zero_shot..."
python run_llm_classifier.py \
--model_name mistral-7b-instruct-v0.3 \
--task_name all \
--mode zero_shot

echo "Running mistral-7b-instruct-v0.3 with few_shot..."
python run_llm_classifier.py \
--model_name mistral-7b-instruct-v0.3 \
--task_name all \
--mode few_shot \
--n_shots 5

echo "Running qwen2.5-7b-instruct with zero_shot..."
python run_llm_classifier.py \
--model_name qwen2.5-7b-instruct \
--task_name all \
--mode zero_shot

echo "Running qwen2.5-7b-instruct with few_shot..."
python run_llm_classifier.py \
--model_name qwen2.5-7b-instruct \
--task_name all \
--mode few_shot \
--n_shots 5


# echo "Running llama-3.3-70B-instruct with zero_shot..."
# python run_llm_classifier.py \
# --model_name llama-3.3-70B-instruct \
# --task_name all \
# --mode zero_shot