"""
Step 3: Train MLP classifier on frozen LLM embeddings.

This script:
1. Loads cached embeddings from step 2
2. Trains MLP head on dev embeddings
3. Saves trained model checkpoint
"""

import argparse
import json
import math
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MLPClassifier(nn.Module):
    """
    MLP classifier head following the specified architecture:
    Dense(input_dim → hidden_dim)
    ReLU
    Dropout(0.1)
    Dense(hidden_dim → 1)
    Sigmoid (applied during inference, not in forward for training with BCEWithLogitsLoss)
    """
    
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
        """Forward pass returns logits (for BCEWithLogitsLoss)."""
        return self.network(x).squeeze(-1)
    
    def predict_proba(self, x):
        """Predict probabilities with sigmoid activation."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def compute_metrics(probs, labels, threshold=0.5):
    """Compute classification metrics."""
    probs_list = probs.cpu().numpy().tolist()
    labels_list = labels.cpu().numpy().tolist()
    
    # Compute AUC
    try:
        auc = roc_auc_score(labels_list, probs_list)
    except ValueError:
        auc = float('nan')
    
    # Compute F1
    preds = [1 if p >= threshold else 0 for p in probs_list]
    try:
        f1 = f1_score(labels_list, preds)
    except ValueError:
        f1 = float('nan')
    
    # Compute accuracy
    accuracy = sum(p == l for p, l in zip(preds, labels_list)) / len(labels_list)
    
    return {
        'auc': float(auc),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_x, batch_y in progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x).float()
        loss = criterion(logits, batch_y.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x).float()
            loss = criterion(logits, batch_y.float())
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(batch_y.cpu())
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    metrics = compute_metrics(all_probs, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train MLP classifier on frozen LLM embeddings'
    )
    parser.add_argument(
        '--embeddings_path',
        type=str,
        required=True,
        help='Path to embeddings .pt file from step 2'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (for saving config)'
    )
    parser.add_argument(
        '--track',
        type=str,
        default='en_es',
        help='Dataset track'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='llm_mlp/models',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split ratio from training data'
    )
    
    args = parser.parse_args()
    
    embeddings_path = os.path.join(os.path.dirname(__file__), '..', args.embeddings_path)
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    print("="*80)
    print("STEP 3: MLP TRAINING")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Track: {args.track}")
    print(f"Embeddings: {embeddings_path}")
    
    # Load embeddings
    print("\n[1/5] Loading embeddings...")
    data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    labels = data['labels']  # Token-level labels
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Compute class imbalance
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    pos_weight = num_neg / max(1, num_pos)
    print(f"  Class distribution: {num_pos} positive, {num_neg} negative")
    print(f"  Pos weight for BCE loss: {pos_weight:.4f}")
    
    # Split into train/val
    print(f"\n[2/5] Splitting data (val_split={args.val_split})...")
    num_samples = len(embeddings)
    num_val = int(num_samples * args.val_split)
    num_train = num_samples - num_val
    
    # Random split
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_embeddings = embeddings[train_indices]
    train_labels = labels[train_indices]
    val_embeddings = embeddings[val_indices]
    val_labels = labels[val_indices]
    
    print(f"  Train: {len(train_embeddings)} samples")
    print(f"  Val: {len(val_embeddings)} samples")
    
    # Create dataloaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3/5] Setting up model (device={device})...")
    
    # Create model
    input_dim = embeddings.shape[1]
    model = MLPClassifier(input_dim=input_dim, dropout=args.dropout)
    model.to(device)
    
    print(f"  Model architecture:")
    print(f"    Input dim: {input_dim}")
    print(f"    Hidden dim: {input_dim // 2}")
    print(f"    Dropout: {args.dropout}")
    
    # Setup training
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print(f"\n[4/5] Training for {args.num_epochs} epochs...")
    best_auc = -float('inf')
    best_state_dict = None
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Log
        record = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1'],
            'val_accuracy': val_metrics['accuracy']
        }
        history.append(record)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if not math.isnan(val_metrics['auc']) and val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Save model
    print(f"\n[5/5] Saving model...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    model_dir = os.path.join(output_dir, f'{args.model_name}_{args.track}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save best model
    if best_state_dict is not None:
        torch.save(best_state_dict, os.path.join(model_dir, 'mlp_classifier.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(model_dir, 'mlp_classifier.pt'))
        best_epoch = args.num_epochs
    
    # Save config
    config = {
        'model_name': args.model_name,
        'track': args.track,
        'input_dim': input_dim,
        'hidden_dim': input_dim // 2,
        'dropout': args.dropout,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'best_epoch': best_epoch,
        'best_val_auc': float(best_auc) if not math.isnan(best_auc) else None,
        'pos_weight': float(pos_weight)
    }
    
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save training history
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"  Saved to: {model_dir}")
    print(f"  Best epoch: {best_epoch} (AUC: {best_auc:.4f})")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Track: {args.track}")
    print(f"Training samples: {len(train_embeddings)}")
    print(f"Validation samples: {len(val_embeddings)}")
    print(f"Best epoch: {best_epoch}/{args.num_epochs}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Output directory: {model_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
