"""
Recover final embeddings from checkpoint file.
"""
import torch
import os
import sys

checkpoint_path = sys.argv[1]
output_path = checkpoint_path.replace('_checkpoint.pt', '.pt')

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Embeddings list length: {len(checkpoint['embeddings_list'])}")
print(f"Labels list length: {len(checkpoint['labels_list'])}")

# Concatenate embeddings
print("Concatenating embeddings...")
embeddings_tensor = torch.cat(checkpoint['embeddings_list'], dim=0)
print(f"Final embeddings shape: {embeddings_tensor.shape}")

# Create final save data
print("Creating final save data...")
save_data = {
    'embeddings': embeddings_tensor,
    'labels': torch.tensor(checkpoint['labels_list'], dtype=torch.long),
    'instance_ids': checkpoint['instance_ids_list'],
    'model_name': 'llama-3.1-8b',  # Update if needed
    'hidden_size': embeddings_tensor.shape[1],
    'track': 'en_es',  # Update if needed
    'split': 'dev' if 'dev' in checkpoint_path else 'test',
    'mode': 'token-level',
}

print(f"Saving to: {output_path}")
torch.save(save_data, output_path)
print(f"Done! File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
