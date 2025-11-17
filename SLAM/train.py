"""
Frozen-backbone LLM training script for SLAM.

Instead of fine-tuning the full LLM, we freeze the base model weights and train
only a lightweight classification head on top of mean-pooled hidden states.
This keeps the experiment comparable across models and drastically reduces
training cost.
"""

import argparse
import json
import logging
import math
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import roc_auc_score, f1_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from data_preprocessing import load_data, InstanceData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_MAPPING = {
    "llama-3.1-8b": "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "llama-3.3-70b-instruct": "meta-llama/Meta-Llama-3.3-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_training_data(
    data: List[InstanceData], labels: Dict[str, float], ratio: float, seed: int
) -> Tuple[List[InstanceData], Dict[str, float]]:
    if ratio >= 0.999:
        return data, labels

    num_samples = max(1, int(len(data) * ratio))
    logger.info("Sampling %.1f%% of training data (%d -> %d instances)", ratio * 100, len(data), num_samples)
    rng = random.Random(seed)
    sampled_instances = rng.sample(data, num_samples)
    sampled_ids = {inst.instance_id for inst in sampled_instances}
    filtered_labels = {iid: labels[iid] for iid in sampled_ids}
    return sampled_instances, filtered_labels


def split_validation(
    data: List[InstanceData],
    labels: Dict[str, float],
    val_ratio: float,
    seed: int,
) -> Tuple[List[InstanceData], List[InstanceData], Dict[str, float], Dict[str, float]]:
    if val_ratio <= 0.0:
        return data, [], labels, {}

    num_val = max(1, int(len(data) * val_ratio))
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    val_indices = set(indices[:num_val])

    train_subset: List[InstanceData] = []
    val_subset: List[InstanceData] = []
    for idx, instance in enumerate(data):
        (val_subset if idx in val_indices else train_subset).append(instance)

    train_labels = {inst.instance_id: labels[inst.instance_id] for inst in train_subset}
    val_labels = {inst.instance_id: labels[inst.instance_id] for inst in val_subset}

    logger.info(
        "Train/validation split: %d train / %d val (%.1f%% holdout)",
        len(train_subset),
        len(val_subset),
        val_ratio * 100,
    )
    return train_subset, val_subset, train_labels, val_labels


def prepare_dataset(
    instances: List[InstanceData],
    labels: Dict[str, float],
    tokenizer: AutoTokenizer,
    max_length: int,
    description: str,
) -> Optional[Dataset]:
    if not instances:
        logger.warning("No instances provided for %s dataset.", description)
        return None

    texts: List[str] = []
    y: List[int] = []
    for inst in instances:
        if inst.instance_id not in labels:
            continue
        texts.append(inst.to_llm_input_text())
        y.append(int(labels[inst.instance_id]))

    if not texts:
        logger.warning("No labelled instances available for %s dataset.", description)
        return None

    dataset = Dataset.from_dict({"text": texts, "labels": y})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info("Prepared %s dataset with %d instances.", description, len(tokenized))
    return tokenized


class FrozenBackboneClassifier(nn.Module):
    """Mean-pooled encoder + trainable binary classification head."""

    def __init__(self, base_model: AutoModel, dropout: float = 0.1):
        super().__init__()
        self.encoder = base_model
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        encoder_dtype = next(self.encoder.parameters()).dtype
        self.classifier = self.classifier.to(dtype=encoder_dtype)

    def _pooled(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = self._pooled(hidden_states, attention_mask)

        pooled = self.dropout(pooled)
        pooled = pooled.to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def train(self, mode: bool = True):
        self.classifier.train(mode)
        self.encoder.eval()
        return self


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_classification_metrics(probs: List[float], labels: List[int]) -> Dict[str, float]:
    if not probs:
        return {"auc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}

    binary_preds = [1 if p >= 0.5 else 0 for p in probs]
    accuracy = sum(int(p == l) for p, l in zip(binary_preds, labels)) / len(labels)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    try:
        f1 = f1_score(labels, binary_preds)
    except ValueError:
        f1 = float("nan")

    return {"auc": float(auc), "f1": float(f1), "accuracy": float(accuracy)}


def evaluate(model: FrozenBackboneClassifier, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    losses: List[float] = []
    probs: List[float] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels_list.extend(batch["labels"].tolist())

    metrics = compute_classification_metrics(probs, labels_list)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return metrics


def train_one_epoch(
    model: FrozenBackboneClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int,
) -> float:
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * grad_accum_steps

    # Flush any remaining gradients
    if len(dataloader) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return epoch_loss / max(1, len(dataloader))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-backbone LLM classifier for SLAM")
    parser.add_argument("--data_dir", type=str, default="data_en_es", help="Directory containing SLAM split files")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAPPING.keys()), help="Backbone model key")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save classifier head and config")
    parser.add_argument("--num_epochs", type=int, default=2, help="Training epochs for classification head")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for classifier head")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for classifier head")
    parser.add_argument("--batch_size", type=int, default=64, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Eval batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=256, help="Sequence length for tokenizer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for classification head")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Fraction of training set to use (for quick tests)")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of training set reserved for validation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling and splitting")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    train_file = os.path.join(data_dir, "en_es.slam.20190204.train")
    if not os.path.isfile(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    logger.info("Loading training data from %s", train_file)
    train_instances, train_labels = load_data(train_file)
    logger.info("Loaded %d labelled instances", len(train_instances))

    if args.train_ratio <= 0 or args.train_ratio > 1.0:
        raise ValueError("--train_ratio must be in (0, 1].")

    train_instances, train_labels = sample_training_data(train_instances, train_labels, args.train_ratio, args.random_seed)
    train_subset, val_subset, train_labels, val_labels = split_validation(
        train_instances, train_labels, args.val_ratio, args.random_seed
    )

    model_path = MODEL_MAPPING[args.model]
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading frozen backbone from %s", model_path)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    classifier_model = FrozenBackboneClassifier(base_model, dropout=args.dropout).to(device)

    train_dataset = prepare_dataset(train_subset, train_labels, tokenizer, args.max_length, "train")
    if train_dataset is None:
        raise RuntimeError("Training dataset is empty after preprocessing.")

    val_dataset = prepare_dataset(val_subset, val_labels, tokenizer, args.max_length, "validation") if val_subset else None

    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = (
        create_dataloader(val_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if val_dataset is not None
        else None
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        classifier_model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    os.makedirs(args.output_dir, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_state_dict = None
    best_metric = -float("inf")
    best_epoch = -1

    for epoch in range(1, args.num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.num_epochs)
        train_loss = train_one_epoch(
            classifier_model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.grad_accum_steps,
        )

        record = {"epoch": epoch, "train_loss": float(train_loss)}

        if val_loader is not None:
            val_metrics = evaluate(classifier_model, val_loader, device)
            record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_auc": val_metrics["auc"],
                    "val_f1": val_metrics["f1"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )
            metric_value = val_metrics.get("auc", float("nan"))
            if not math.isnan(metric_value) and metric_value > best_metric:
                best_metric = metric_value
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu() for k, v in classifier_model.classifier.state_dict().items()}
        else:
            best_state_dict = {k: v.detach().cpu() for k, v in classifier_model.classifier.state_dict().items()}
            best_epoch = epoch

        history.append(record)
        logger.info("Epoch %d summary: %s", epoch, record)

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu() for k, v in classifier_model.classifier.state_dict().items()}
        best_epoch = args.num_epochs

    classifier_path = os.path.join(args.output_dir, "classifier.pt")
    torch.save(best_state_dict, classifier_path)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "model_key": args.model,
        "base_model_path": model_path,
        "hidden_size": classifier_model.encoder.config.hidden_size,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_epochs": args.num_epochs,
    }
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    display_metric = best_metric if math.isfinite(best_metric) else float("nan")
    logger.info(
        "Saved classifier head to %s (best epoch: %d, val AUC: %.4f)",
        classifier_path,
        best_epoch,
        display_metric,
    )


if __name__ == "__main__":
    main()