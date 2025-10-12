#!/bin/bash

### Use LLM to classify the data
source venv/bin/activate

# echo "Running llama-3.1-8b-instruct with few_shot..."
# python run_llm_classifier.py \
# --model_name llama-3.1-8b-instruct \
# --task_name all \
# --mode few_shot \
# --n_shots 5

# echo "Running llama-3.1-8b-instruct with zero_shot..."
# python run_llm_classifier.py \
# --model_name llama-3.1-8b-instruct \
# --task_name all \
# --mode zero_shot

# echo "Running mistral-7b-instruct-v0.3 with zero_shot..."
# python run_llm_classifier.py \
# --model_name mistral-7b-instruct-v0.3 \
# --task_name all \
# --mode zero_shot

# echo "Running mistral-7b-instruct-v0.3 with few_shot..."
# python run_llm_classifier.py \
# --model_name mistral-7b-instruct-v0.3 \
# --task_name all \
# --mode few_shot \
# --n_shots 5

# ===============================================================================
# Student Reasoning Task Evaluation
# ===============================================================================

echo "Running llama-3.1-8b-instruct on student_reasoning with zero_shot..."
python run_llm_classifier.py \
--model_name llama-3.1-8b-instruct \
--task_name student_reasoning \
--mode zero_shot

echo "Running llama-3.1-8b-instruct on student_reasoning with few_shot..."
python run_llm_classifier.py \
--model_name llama-3.1-8b-instruct \
--task_name student_reasoning \
--mode few_shot \
--n_shots 5

echo "Running mistral-7b-instruct-v0.3 on student_reasoning with zero_shot..."
python run_llm_classifier.py \
--model_name mistral-7b-instruct-v0.3 \
--task_name student_reasoning \
--mode zero_shot

echo "Running mistral-7b-instruct-v0.3 on student_reasoning with few_shot..."
python run_llm_classifier.py \
--model_name mistral-7b-instruct-v0.3 \
--task_name student_reasoning \
--mode few_shot \
--n_shots 5
