"""
Step 4: Generate exercise-level predictions and convert to token-level format.

This script:
1. Loads trained MLP model
2. Loads test exercise embeddings
3. Generates exercise-level predictions
4. Replicates predictions to all tokens in each exercise
5. Outputs in token-level format for eval.py
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MLPClassifier(nn.Module):
    """MLP classifier head."""
    
    def __init__(self, input_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions with trained MLP on exercise embeddings'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--embeddings_path',
        type=str,
        required=True,
        help='Path to test token embeddings'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output file for predictions (token-level format)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for inference'
    )
    
    args = parser.parse_args()
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', args.model_dir)
    
    print("="*80)
    print("STEP 4: INFERENCE")
    print("="*80)
    print(f"Model directory: {model_dir}")
    
    # Load config
    print("\n[1/5] Loading model config...")
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"  Model: {config['model_name']}")
    print(f"  Input dim: {config['input_dim']}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    
    # Load model
    print("\n[2/5] Loading trained MLP...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    state_dict = torch.load(
        os.path.join(model_dir, 'mlp_classifier.pt'),
        map_location='cpu',
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  Loaded model to {device}")
    
    # Load embeddings
    print("\n[3/5] Loading test embeddings...")
    embeddings_path = os.path.join(os.path.dirname(__file__), '..', args.embeddings_path)
    data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    
    embeddings = data['embeddings']
    instance_ids = data['instance_ids']
    
    print(f"  Loaded {len(embeddings)} token embeddings")
    
    # Generate predictions
    print("\n[4/5] Generating token-level predictions...")
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), args.batch_size), desc="Predicting"):
            batch_emb = embeddings[i:i+args.batch_size].to(device)
            batch_probs = model.predict_proba(batch_emb)
            all_probs.append(batch_probs.cpu())
    
    all_probs = torch.cat(all_probs)
    
    # Save predictions
    print("\n[5/5] Saving predictions...")
    output_file = os.path.join(os.path.dirname(__file__), '..', args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for instance_id, prob in zip(instance_ids, all_probs):
            f.write(f"{instance_id} {prob.item():.6f}\n")
    
    print(f"  Saved predictions to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tokens processed: {len(embeddings)}")
    print(f"Token-level predictions generated: {len(instance_ids)}")
    print(f"Output file: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
