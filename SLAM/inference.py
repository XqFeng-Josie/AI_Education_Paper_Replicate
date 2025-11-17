"""
Inference script for the frozen-backbone SLAM classifiers.
Loads the frozen encoder, attaches the trained classification head, and produces
probabilities for each instance in the specified split.
"""

import argparse
import json
import logging
import os
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from data_preprocessing import load_data
from train import FrozenBackboneClassifier, MODEL_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_rank_world_size():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size

def load_config(model_dir: str) -> Dict:
    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing model_config.json in {model_dir}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_tokenizer(model_dir: str, base_model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_frozen_model(model_dir: str, device: torch.device) -> tuple:
    config = load_config(model_dir)
    model_key = config.get("model_key")
    base_model_path = config.get("base_model_path") or MODEL_MAPPING.get(model_key)
    if base_model_path is None:
        raise ValueError("Base model path missing in config and MODEL_MAPPING.")

    tokenizer = load_tokenizer(model_dir, base_model_path)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModel.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    )

    base_model.to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)
        p.data = p.data.to(device)

    classifier = FrozenBackboneClassifier(base_model, dropout=config.get("dropout", 0.1))

    state_path = os.path.join(model_dir, "classifier.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Classifier weights not found: {state_path}")

    state_dict = torch.load(state_path, map_location="cpu")
    classifier.classifier.load_state_dict(state_dict)

    classifier = classifier.to(device)
    classifier.eval()

    max_length = config.get("max_length", 256)
    return classifier, tokenizer, max_length


def predict_instances(
    model: FrozenBackboneClassifier,
    tokenizer,
    instances: List,
    device: torch.device,
    max_length: int,
    batch_size: int,
    threshold: float = 0.5,
) -> Dict[str, float]:
    predictions: Dict[str, float] = {}
    logger.info("Predicting %d instances with batch size %d (threshold=%.3f)", len(instances), batch_size, threshold)

    for start in tqdm(range(0, len(instances), batch_size), desc="Inference", leave=False):
        batch = instances[start : start + batch_size]
        texts = [inst.to_llm_input_text() for inst in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
                probs = torch.sigmoid(logits).cpu().tolist()

        for inst, prob in zip(batch, probs):
            prob_value = float(max(0.0, min(1.0, prob)))
            # Apply threshold: if prob >= threshold, predict as class 1, else class 0
            # pred_class = 1.0 if prob_value >= threshold else 0.0
            predictions[inst.instance_id] = prob_value

    return predictions


def write_predictions(predictions: Dict[str, float], instances: List, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        for inst in instances:
            pred = predictions.get(inst.instance_id, 0.0)
            # f.write(f"{inst.instance_id} {pred}\n")
            f.write(f"{inst.instance_id},{pred}\n")
    logger.info("Predictions written to %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for frozen-backbone SLAM classifiers")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with trained classifier head")
    parser.add_argument("--data_dir", type=str, default="data_en_es", help="Directory containing SLAM splits")
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="test", help="Split to predict")
    parser.add_argument("--output_file", type=str, default=None, help="Path for .pred file (baseline format)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=None, help="Override sequence length (defaults to training)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification (prob >= threshold -> class 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    model, tokenizer, config_max_length = load_frozen_model(args.model_dir, device)
    # max_length = args.max_length or config_max_length
    max_length = args.max_length if args.max_length is not None else config_max_length

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    data_file = os.path.join(data_dir, f"en_es.slam.20190204.{args.split}")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Split file not found: {data_file}")

    logger.info("Loading %s split from %s", args.split, data_file)
    instances, _ = load_data(data_file)
    logger.info("Loaded %d instances", len(instances))
    rank, world_size = get_rank_world_size()
    logger.info("Inference with world_size=%d, rank=%d", world_size, rank)

    # 按 rank 做简单切片：每个进程负责 1/world_size 的样本
    local_instances = instances[rank::world_size]
    logger.info("Rank %d will process %d instances", rank, len(local_instances))

    predictions = predict_instances(
        model,
        tokenizer,
        local_instances,
        device,
        max_length=max_length,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    output_path = (
        args.output_file
        if args.output_file
        else os.path.join(data_dir, f"llm_{os.path.basename(args.model_dir)}_{args.split}.pred")
    )
    write_predictions(predictions, local_instances, output_path)
    logger.info("Inference complete. Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()

