"""
Step 5: Full Conversation Context
Use complete dialogue history for improved classification
"""
import pandas as pd
import json
from typing import List, Dict
from tqdm import tqdm
import config
from step2_zero_shot_inference import run_llm_inference_real, save_predictions


def load_full_dataset(dataset_path: str) -> Dict:
    """Load the full dataset with conversation history"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prepDataset"]


def get_conversation_history(entry: Dict) -> str:
    """
    Extract and format full conversation history
    
    Args:
        entry: Dataset entry with past_convo field
    
    Returns:
        Formatted conversation string
    """
    if "past_convo" not in entry or len(entry["past_convo"]) == 0:
        return ""
    
    # Format as a multi-turn dialogue
    conversation = "\n".join([
        f"Turn {i+1}: {utterance}"
        for i, utterance in enumerate(entry["past_convo"])
    ])
    
    return conversation


def format_context_prompt(conversation: str) -> List[Dict[str, str]]:
    """
    Format prompt with full conversation context
    
    Args:
        conversation: Full conversation history
    
    Returns:
        List of message dicts
    """
    system_prompt = """You're observing a student learning Italian prepositions through a dialogue with a tutor.
Based on the full conversation history, classify the student's LATEST response into one out of 4 categories:
[Guess, Question, Affirmation, Other].
Only return the label corresponding to one of the four categories."""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Conversation:\n{conversation}\n\nClassify the student's latest utterance:"}
    ]


def predict_with_context(test_df: pd.DataFrame, full_dataset: Dict) -> List[str]:
    """
    Run prediction with full conversation context
    
    Args:
        test_df: Test DataFrame with utterances
        full_dataset: Full dataset dict with conversation histories
    
    Returns:
        List of predicted labels
    """
    print("Running zero-shot inference with full conversation context...")
    
    predictions = []
    
    # Create a mapping from utterance to entry
    utterance_to_entry = {}
    for entry_id, entry in full_dataset.items():
        if "past_convo" in entry and len(entry["past_convo"]) > 0:
            last_utterance = entry["past_convo"][-1]
            utterance_to_entry[last_utterance] = entry
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Context inference"):
        utterance = row["utterance"]
        
        # Find conversation history
        if utterance in utterance_to_entry:
            entry = utterance_to_entry[utterance]
            conversation = get_conversation_history(entry)
        else:
            # Fallback to just the utterance
            conversation = utterance
        
        messages = format_context_prompt(conversation)
        prediction = run_llm_inference_real(messages, "llama")
        
        # Ensure prediction is valid
        if prediction not in config.LABELS:
            prediction = "Guess"
        
        predictions.append(prediction)
    
    return predictions


def main():
    """Main function for full context experiment"""
    print("="*60)
    print("CIMA Full Conversation Context Experiment")
    print("="*60)
    
    # Load test data
    print(f"\nLoading test data from {config.TEST_PATH}...")
    test_df = pd.read_csv(config.TEST_PATH)
    
    # Load full dataset
    print(f"Loading full dataset from {config.DATASET_PATH}...")
    full_dataset = load_full_dataset(config.DATASET_PATH)
    
    print(f"Test set size: {len(test_df)}")
    
    # Run context-aware prediction
    print("\n" + "-"*60)
    predictions = predict_with_context(test_df, full_dataset)
    
    # Save predictions
    save_predictions(
        predictions,
        test_df["label"].tolist(),
        test_df["utterance"].tolist(),
        config.RESULTS_DIR + "/llama_context_predictions.csv"
    )
    
    print("\n" + "="*60)
    print("Full context experiment completed!")
    print("="*60)
    print("\nTip: Run step3_evaluate.py to evaluate these predictions")


if __name__ == "__main__":
    main()
