"""
Step 4: Few-Shot Prompting
Implement 5-shot and 20-shot prompting experiments
"""
import pandas as pd
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
import config
from step2_zero_shot_inference import (
    run_llm_inference_real,
    save_predictions,
)


def select_few_shot_examples(train_df: pd.DataFrame, n_shots: int, stratified: bool = True) -> List[Dict]:
    """
    Select few-shot examples from training set
    
    Args:
        train_df: Training DataFrame
        n_shots: Number of examples per class (total = n_shots * 4)
        stratified: If True, select n_shots per class; else select n_shots total
    
    Returns:
        List of example dicts with 'utterance' and 'label'
    """
    examples = []
    
    if stratified:
        # Select n_shots per class
        for label in config.LABELS:
            label_df = train_df[train_df["label"] == label]
            n_available = min(n_shots, len(label_df))
            
            if n_available == 0:
                print(f"Warning: No examples available for label '{label}'")
                continue
            
            sampled = label_df.sample(n=n_available, random_state=config.RANDOM_SEED)
            
            for _, row in sampled.iterrows():
                examples.append({
                    "utterance": row["utterance"],
                    "label": row["label"]
                })
    else:
        # Select n_shots total (random)
        sampled = train_df.sample(n=n_shots, random_state=config.RANDOM_SEED)
        for _, row in sampled.iterrows():
            examples.append({
                "utterance": row["utterance"],
                "label": row["label"]
            })
    
    return examples


def format_few_shot_prompt(utterance: str, examples: List[Dict]) -> List[Dict[str, str]]:
    """
    Format few-shot prompt with examples
    
    Args:
        utterance: Test utterance to classify
        examples: List of example dicts
    
    Returns:
        List of message dicts
    """
    messages = [
        {"role": "system", "content": config.GPT4_SYSTEM_PROMPT}
    ]
    
    # Add few-shot examples
    for ex in examples:
        messages.append({
            "role": "user",
            "content": f"Utterance: {ex['utterance']}"
        })
        messages.append({
            "role": "assistant",
            "content": ex["label"]
        })
    
    # Add test utterance
    messages.append({
        "role": "user",
        "content": f"Utterance: {utterance}"
    })
    
    return messages


def predict_few_shot(utterances: List[str], n_shots: int) -> List[str]:
    """
    Run few-shot prediction
    
    Args:
        utterances: List of test utterances
        n_shots: Number of shots per class
    
    Returns:
        List of predicted labels
    """
    print(f"Running {n_shots}-shot inference...")
    
    # Load training data
    train_df = pd.read_csv(config.TRAIN_PATH)
    
    # Select few-shot examples
    examples = select_few_shot_examples(train_df, n_shots=n_shots, stratified=True)
    print(f"Selected {len(examples)} few-shot examples")
    
    # Count examples per class
    example_counts = Counter([ex["label"] for ex in examples])
    print("Examples per class:")
    for label in config.LABELS:
        print(f"  {label}: {example_counts.get(label, 0)}")
    
    predictions = []
    
    for utterance in tqdm(utterances, desc=f"{n_shots}-shot inference"):
        messages = format_few_shot_prompt(utterance, examples)
        prediction = run_llm_inference_real(messages, "llama")
        
        # Ensure prediction is valid
        if prediction not in config.LABELS:
            prediction = "Guess"
        
        predictions.append(prediction)
    
    return predictions


def main():
    """Main function for few-shot experiments"""
    print("="*60)
    print("CIMA Few-Shot Prompting Experiments")
    print("="*60)
    
    # Load test data
    test_df = pd.read_csv(config.TEST_PATH)
    utterances = test_df["utterance"].tolist()
    true_labels = test_df["label"].tolist()
    
    # Run 5-shot experiment
    print("\n" + "-"*60)
    predictions_5shot = predict_few_shot(utterances, n_shots=5)
    save_predictions(
        predictions_5shot,
        true_labels,
        utterances,
        config.RESULTS_DIR + "/llama_5shot_predictions.csv"
    )
    
    # Run 20-shot experiment
    # print("\n" + "-"*60)
    # predictions_20shot = predict_few_shot(utterances, n_shots=20)
    # save_predictions(
    #     predictions_20shot,
    #     true_labels,
    #     utterances,
    #     config.RESULTS_DIR + "/llama_20shot_predictions.csv"
    # )
    
    print("\n" + "="*60)
    print("Few-shot experiments completed!")
    print("="*60)
    print("\nTip: Run step3_evaluate.py to evaluate these predictions")


if __name__ == "__main__":
    main()
