# LLM Experiments

This directory contains experiments using Large Language Models (LLMs) for early identification of at-risk students, extending the traditional ML approach from the original paper.

## Experimental Design

### Methodology

The LLM experiments use a **Unified Single-Agent System** that integrates multiple analytical perspectives in a single LLM call:

- **Academic Performance Analyst**: Analyzes historical assignment submissions, VLE activity levels, and academic progress
- **Behavioral Pattern Analyst**: Examines login frequency, learning session patterns, and engagement consistency
- **Peer Comparison Analyst**: Compares student behavior with cohort statistics and peer groups
- **Temporal Analyst**: Identifies temporal patterns and trends over time

### Data Splitting

The experiments follow the **same time-window based split rules** as the traditional ML methods, ensuring consistency with the paper's methodology:

- Training set: All students whose cutoff time is before the prediction point
- Test set: All students whose cutoff time is at the prediction point (day 0-11)

### Sampling Strategy

For current evaluation, we sampled **800 test samples** per day according to the label distribution (balanced ~50/50). This approach ensures a robust evaluation of the model's ability to distinguish between at-risk and non-at-risk students across all four modules (BBB, DDD, EEE, FFF).

**Note**: The baseline traditional ML methods use the full day-off test dataset, which ranges from 1,945 samples (day 0) to 6,014 samples (day 11).

### Experiment Modes

1. **Few-shot mode**: Uses balanced sampling to select representative examples from the training set to guide predictions
2. **Zero-shot mode**: Direct prediction without example guidance

## Usage

### Prerequisites

1. Ensure the OULAD dataset is prepared (see main README)
2. Set up LLM provider:
   - **OpenRouter**: Set `OPENROUTER_API_KEY` environment variable
   - **Local Llama**: Start local server(s) (see `server/` directory)

### Running Experiments

Example script: `run_example.sh`

```bash
cd llm_experiments/
bash run_example.sh [provider]  # provider: openrouter, local, or multi_local
```

### Main Script: `unified_agent_main.py`

Key parameters:
- `--provider`: LLM provider (`openrouter`, `local`, `multi_local`)
- `--mode`: Experiment mode (`zero_shot`, `few_shot`)
- `--module`: Module code (e.g., `BBB`, `DDD`, `EEE`, `FFF`)
- `--presentation`: Course presentation (e.g., `2014J`)
- `--assessment_name`: Assessment name (e.g., `TMA 1`)
- `--days_to_cutoff`: Days to cutoff (0-11)
- `--max_students`: Maximum number of students to test (default: None = all)
- `--num_few_shot`: Number of few-shot examples (for few-shot mode)

Example:
```bash
python unified_agent_main.py \
    --provider multi_local \
    --mode few_shot \
    --num_few_shot 5 \
    --module BBB \
    --presentation 2014J \
    --assessment_name "TMA 1" \
    --days_to_cutoff 0 \
    --max_students 100
```

## Results

Results are saved in JSON format in the `results/` directory.

### Current Performance (800 samples, balanced)

| Day | N | Pos% | PR-AUC | Baseline N |
| --- | --- | --- | --- | --- |
| 0 | 800 | 50.0 | 0.6567 | 1,945 |
| 1 | 800 | 50.0 | 0.7249 | 3,450 |
| 2 | 800 | 50.0 | 0.7202 | 4,218 |
| 3 | 800 | 50.0 | 0.7153 | 4,649 |
| 4 | 800 | 50.0 | 0.7333 | 4,906 |
| 5 | 800 | 50.1 | 0.7717 | 5,067 |
| 6 | 800 | 50.0 | 0.7703 | 5,193 |
| 7 | 800 | 50.0 | 0.7576 | 5,345 |
| 8 | 800 | 50.0 | 0.7275 | 5,614 |
| 9 | 800 | 50.0 | 0.7551 | 5,852 |
| 10 | 800 | 50.0 | 0.6895 | 5,947 |
| 11 | 800 | 50.0 | 0.6383 | 6,014 |

The results contain:
- Student predictions and true labels
- PR-AUC, ROC-AUC, and accuracy metrics
- Few-shot examples used (if applicable)
- Individual student results with prompts and responses

## Files

- `unified_agent_main.py`: Main experiment script
- `unified_agent.py`: Unified agent implementation
- `behavior_to_text.py`: Converts numerical features to natural language descriptions
- `llm_client.py`: Unified LLM client supporting multiple providers
- `calculate_metrics.py`: Metric calculation utilities
- `server/`: Local Llama server setup scripts

