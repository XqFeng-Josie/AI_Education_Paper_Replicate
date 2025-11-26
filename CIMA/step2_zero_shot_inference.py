"""
Step 2: Zero-Shot Inference
Run zero-shot prompting experiments with LLMs
"""
import pandas as pd
import json
from typing import List, Dict
from tqdm import tqdm
import config


def format_gpt4_prompt(utterance: str) -> List[Dict[str, str]]:
    """
    Format prompt in GPT-4 style (system + user messages)
    
    Args:
        utterance: Student utterance
    
    Returns:
        List of message dicts
    """
    return [
        {"role": "system", "content": config.GPT4_SYSTEM_PROMPT},
        {"role": "user", "content": config.GPT4_USER_PROMPT.format(utterance=utterance)}
    ]


def format_mistral_prompt(utterance: str, label: str) -> str:
    """
    Format prompt in Mistral style (single prompt with label)
    
    Args:
        utterance: Student utterance
        label: Candidate label
    
    Returns:
        Formatted prompt string
    """
    return config.MISTRAL_PROMPT.format(utterance=utterance, label=label)


def run_llm_inference_real(messages: List[Dict], model_name: str = "llama") -> str:
    """
    Real LLM inference function using API or local models
    
    Args:
        messages: List of message dicts
        model_name: Model identifier ("llama", "gpt", "mistral")
    
    Returns:
        Model prediction (single label)
    """
    from llm_inference import run_llm_inference
    
    prediction = run_llm_inference(messages, model_name)
    return prediction


def run_mistral_label_probability_real(prompt: str, label: str) -> float:
    """
    Real function to get label probability from Mistral
    
    Args:
        prompt: Formatted prompt
        label: Candidate label
    
    Returns:
        Probability score for the label
    """
    from llm_inference import run_mistral_label_probability
    
    prob = run_mistral_label_probability(prompt, label)
    return prob


def predict_llama_zero_shot(utterances: List[str]) -> List[str]:
    """
    Run zero-shot prediction with Llama-3.1-405B (GPT-4 style)
    
    Args:
        utterances: List of student utterances
    
    Returns:
        List of predicted labels
    """
    print("Running Llama-3.1-405B zero-shot inference...")
    predictions = []
    
    for utterance in tqdm(utterances, desc="Llama inference"):
        messages = format_gpt4_prompt(utterance)
        prediction = run_llm_inference_real(messages, "llama")
        
        # Ensure prediction is valid
        if prediction not in config.LABELS:
            # Default to most common label if invalid
            prediction = "Guess"
        
        predictions.append(prediction)
    
    return predictions


def predict_mistral_zero_shot(utterances: List[str]) -> List[str]:
    """
    Run zero-shot prediction with Mistral-7B (label probability style)
    
    For each utterance, test all labels and select the one with highest probability
    
    Args:
        utterances: List of student utterances
    
    Returns:
        List of predicted labels
    """
    print("Running Mistral-7B zero-shot inference...")
    predictions = []
    
    for utterance in tqdm(utterances, desc="Mistral inference"):
        label_probs = {}
        
        # Test each label
        for label in config.LABELS:
            prompt = format_mistral_prompt(utterance, label)
            prob = run_mistral_label_probability_real(prompt, label)
            label_probs[label] = prob
        
        # Select label with highest probability
        prediction = max(label_probs, key=label_probs.get)
        predictions.append(prediction)
    
    return predictions


def save_predictions(predictions: List[str], true_labels: List[str], 
                     utterances: List[str], output_path: str):
    """
    Save predictions to file
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        utterances: Original utterances
        output_path: Path to save predictions
    """
    results = pd.DataFrame({
        "utterance": utterances,
        "true_label": true_labels,
        "predicted_label": predictions
    })
    
    results.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved predictions to {output_path}")


def main():
    """Main function for zero-shot inference"""
    print("="*60)
    print("CIMA Zero-Shot Inference")
    print("="*60)
    
    # Load test data
    print(f"\nLoading test data from {config.TEST_PATH}...")
    test_df = pd.read_csv(config.TEST_PATH)
    
    print(f"Test set size: {len(test_df)}")
    print(f"Label distribution:")
    print(test_df["label"].value_counts())
    
    utterances = test_df["utterance"].tolist()
    true_labels = test_df["label"].tolist()
    
    # Run Llama-3.1-405B inference
    print("\n" + "-"*60)
    llama_predictions = predict_llama_zero_shot(utterances)
    save_predictions(
        llama_predictions, 
        true_labels, 
        utterances,
        config.RESULTS_DIR + "/llama_predictions.csv"
    )
    
    # # Run Mistral-7B inference
    print("\n" + "-"*60)
    mistral_predictions = predict_mistral_zero_shot(utterances)
    save_predictions(
        mistral_predictions,
        true_labels,
        utterances,
        config.RESULTS_DIR + "/mistral_predictions.csv"
    )
    
    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)


if __name__ == "__main__":
    main()
