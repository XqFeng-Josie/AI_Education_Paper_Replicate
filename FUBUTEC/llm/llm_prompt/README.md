# Few-shot Prompt Experiments

Few-shot prompting for student performance prediction using LLM reasoning (no model training required). Prompts and outputs are English-only.

## How it works
1. Convert each student row to a structured natural-language description.
2. Pick few-shot examples from the training set (balanced for classification, random/diverse for regression).
3. Build prompts with Chain-of-Thought guidance and optional JSON output format.
4. Let the LLM predict pass/fail (classification) or G3 (regression).

## Quick start
```bash
cd llm/llm_prompt
export OPENROUTER_API_KEY=your-api-key
python main.py \
  --model meta-llama/llama-3.3-70b-instruct \
  --n_examples 5 \
  --random_state 42 \
  --temperature 0.3
```

Key flags:
- `--data_path`: student CSV (default: `../data/student-por.csv`)
- `--model`: OpenRouter model (default: `meta-llama/llama-3.3-70b-instruct`)
- `--n_examples`: number of few-shot examples (default: 5)
- `--random_state`: seed (default: 42)
- `--temperature`: sampling temperature (default: 0.3)
- `--resume`: resume from checkpoint (skip completed tasks)
- `--checkpoint`: checkpoint path (default: `../results/checkpoint_prompt.json`)
- `--use_feature_selection`: enable feature-importance filtering (default: True)
- `--feature_selection_model`: `rf` or `xgb` (default: `rf`)
- `--n_top_features`: top-N features for prompts (default: 10)
- `--use_self_consistency` / `--n_consistency_samples` / `--consistency_temperature`: self-consistency settings
- `--output_cot`: emit JSON with reasoning + prediction

## Data split
- Matches the baseline first split: `KFold(n_splits=10, shuffle=True, random_state=42)` first fold is test.
- Single split only: train (~90%) for few-shot examples, test (~10%) for evaluation.

## Tasks executed
1. Setup A - classification
2. Setup A - regression
3. Setup C - classification
4. Setup C - regression

## Resume behavior
- A checkpoint (`../results/checkpoint_prompt.json` by default) is updated after every task.
- Use `--resume` to continue; the checkpoint is deleted after all tasks finish.

## Outputs (`../results`)
- `results_prompt_YYYYMMDD_HHMMSS.json`: summary of completed tasks.
- `results_prompt_summary_YYYYMMDD_HHMMSS.csv`: tabular summary.
- `results_prompt_detailed_YYYYMMDD_HHMMSS.json`: per-sample prompts, responses, and predictions.
- `checkpoint_prompt.json`: auto-managed when resume is enabled.

## Notes
- API cost scales with number of test samples; consider smaller runs first.
- Responses are parsed with fallbacks; enable `--output_cot` for JSON reasoning if needed.
- Rate limiting is handled in the client, but long runs may still benefit from `--resume`.

