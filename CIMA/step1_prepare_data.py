"""
Step 1: Data Preparation
Parse dataset.json and create train/dev/test splits
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import config


def load_and_parse_dataset(dataset_path):
    """
    Load dataset.json and extract student utterances with labels
    
    Returns:
        DataFrame with columns: utterance, studentActions_raw, label
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prep_dataset = data["prepDataset"]
    
    processed_data = []
    skipped = 0
    
    for entry_id, entry in prep_dataset.items():
        # Get the last utterance from past_convo
        if "past_convo" not in entry or len(entry["past_convo"]) == 0:
            skipped += 1
            continue
        
        utterance = entry["past_convo"][-1]
        
        # Get studentActions
        if "studentActions" not in entry:
            skipped += 1
            continue
        
        student_actions = entry["studentActions"]
        
        # Convert string booleans to actual booleans
        try:
            bool_actions = [s.lower() == "true" if isinstance(s, str) else s for s in student_actions]
        except:
            skipped += 1
            continue
        
        # Check if exactly one True value
        if sum(bool_actions) != 1:
            skipped += 1
            continue
        
        # Find which label is True
        true_index = bool_actions.index(True)
        
        # Map to label name using order: [Guess, Question, Affirmation, Other]
        if true_index >= len(config.LABELS):
            skipped += 1
            continue
        
        label = config.LABELS[true_index]
        
        processed_data.append({
            "utterance": utterance,
            "studentActions_raw": str(student_actions),
            "label": label
        })
    
    print(f"Processed {len(processed_data)} valid entries, skipped {skipped} entries")
    
    df = pd.DataFrame(processed_data)
    
    # Print label distribution
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nTotal samples: {len(df)}")
    
    return df


def create_stratified_split(df, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create stratified train/dev/test splits
    
    Args:
        df: DataFrame with utterances and labels
        train_ratio: Proportion for training set
        dev_ratio: Proportion for dev set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, dev_df, test_df
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Check minimum class counts for stratification
    label_counts = df["label"].value_counts()
    min_count = label_counts.min()
    
    # Use stratified split if possible
    use_stratify = min_count >= 2
    
    if not use_stratify:
        print(f"\nWarning: Some classes have fewer than 2 samples (min={min_count}).")
        print("Using non-stratified split instead.")
    
    # First split: train vs (dev + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["label"] if use_stratify else None,
        random_state=random_state
    )
    
    # Second split: dev vs test
    dev_test_ratio = dev_ratio / (dev_ratio + test_ratio)
    
    # Check if temp_df has enough samples for stratification
    temp_label_counts = temp_df["label"].value_counts()
    use_stratify_temp = temp_label_counts.min() >= 2 if use_stratify else False
    
    dev_df, test_df = train_test_split(
        temp_df,
        train_size=dev_test_ratio,
        stratify=temp_df["label"] if use_stratify_temp else None,
        random_state=random_state
    )
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Dev:   {len(dev_df)} ({len(dev_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print label distribution for each split
    for split_name, split_df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        print(f"\n{split_name} label distribution:")
        counts = split_df["label"].value_counts()
        for label in config.LABELS:
            count = counts.get(label, 0)
            pct = count / len(split_df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    return train_df, dev_df, test_df


def save_splits(train_df, dev_df, test_df):
    """Save splits to CSV files"""
    train_df.to_csv(config.TRAIN_PATH, index=False, encoding="utf-8")
    dev_df.to_csv(config.DEV_PATH, index=False, encoding="utf-8")
    test_df.to_csv(config.TEST_PATH, index=False, encoding="utf-8")
    
    print(f"\nSaved splits to:")
    print(f"  Train: {config.TRAIN_PATH}")
    print(f"  Dev:   {config.DEV_PATH}")
    print(f"  Test:  {config.TEST_PATH}")


def main():
    """Main function to prepare data"""
    # Load and parse dataset
    df = load_and_parse_dataset(config.DATASET_PATH)
    
    # Create stratified splits
    train_df, dev_df, test_df = create_stratified_split(
        df,
        train_ratio=config.TRAIN_RATIO,
        dev_ratio=config.DEV_RATIO,
        test_ratio=config.TEST_RATIO,
        random_state=config.RANDOM_SEED
    )
    
    # Save splits
    save_splits(train_df, dev_df, test_df)
    
    print("\n" + "="*60)
    print("Data preparation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
