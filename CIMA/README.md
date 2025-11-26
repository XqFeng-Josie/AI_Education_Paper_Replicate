# CIMA Student Action Classification

Replication of the CIMA experiment: Classifying student actions in Italian preposition tutoring dialogues using large language models.

## Overview

- **Paper**: [Can Large Language Models Transform Computational Social Science?](https://www.educationaldatamining.org/edm2024/proceedings/2024.EDM-posters.95/2024.EDM-posters.95.pdf) (EDM 2024)
- **Task**: Classify student utterances into 4 categories: Guess, Question, Affirmation, Other
- **Evaluation Metric**: Macro-averaged F1 score
- **Dataset**: [CIMA Corpus](https://github.com/kstats/CIMA)

## Dataset

- **Total Samples**: 1,044
- **Split**: Train 730 / Dev 157 / Test 157 (70% / 15% / 15%)
- **Label Distribution**: Guess 45.5%, Question 46.3%, Affirmation 8.0%, Other 0.2%

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

If using API models, set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

### 3. Run Experiments

```bash
# Data preparation
python step1_prepare_data.py

# Zero-shot inference
python step2_zero_shot_inference.py

# Few-shot learning
python step4_few_shot.py

# Full conversation context
python step5_full_context.py

# Evaluate results
python step3_evaluate.py
```

## Project Structure

```
CIMA/
├── config.py                    # Configuration file
├── llm_inference.py            # LLM inference module
├── utils.py                     # Utility functions
├── step1_prepare_data.py        # Data preprocessing
├── step2_zero_shot_inference.py # Zero-shot inference
├── step3_evaluate.py            # Result evaluation
├── step4_few_shot.py            # Few-shot learning
├── step5_full_context.py        # Full conversation context
├── data/                        # Data files
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
└── results/                     # Experimental results
    ├── metrics.json
    └── *_predictions.csv
```

## Supported Models

- **GPT-4**: OpenAI API
- **Llama-3.1-405B**: OpenRouter API
- **Mistral-7B**: Local deployment (HuggingFace)

## Experimental Results

Results are saved in `results/metrics.json` and corresponding CSV files.

### Paper Results (from EDM 2024)

| Model | Approach | Macro F1 |
|-------|----------|----------|
| GPT-4 | Zero-shot | 0.49 |
| GPT-4 | 5-shot | 0.45 |
| Mistral-7B | Zero-shot | 0.11 |
| Mistral-7B | 5-shot | 0.11 |
| Mistral-7B | 20-shot | 0.20 |

### Implementation Results

| Model | Approach | Macro F1 |
|-------|----------|----------|
| Llama-3.1-405B | Zero-shot | 0.524 |
| Llama-3.1-405B | 5-shot | 0.400 |
| Llama-3.1-405B | Full Context | 0.738 |
| Mistral-7B | Zero-shot | 0.233 |


## Configuration

Main configuration is in `config.py`:

- **API Keys**: Set via environment variables or modify `config.py` directly
- **Model Configuration**: Configure model parameters in the `MODEL_CONFIGS` dictionary
- **Mistral Local Deployment**: Set `device` and `load_in_8bit` options

## Prompt Templates

### GPT-4 / Llama Style

**System**: 
```
You're observing a student learning Italian prepositions.
Classify their response into one out of 4 categories: [Guess, Question, Affirmation, Other].
Only return the label corresponding to one of the four categories.
```

**User**: 
```
Utterance: {utterance}
```

### Mistral Style

Test each label separately and select the one with highest probability:
```
Scenario: You're observing a student learning Italian prepositions.
Student Utterance: {utterance}
Student Action: {label}
```

## Notes

1. **Class Imbalance**: The "Other" class has only 2 samples, which should be considered during evaluation
2. **Mistral Local Deployment**: First run will download the model from HuggingFace (~13GB)
3. **API Calls**: Be mindful of costs when using API models
