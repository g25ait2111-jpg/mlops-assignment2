"""
train.py — Model loading, W&B tracking, Trainer setup, training loop
MLOps Assignment 2 | IIT Jodhpur

Usage:
    python train.py --data_path /path/to/goodreads_reviews_dedup.json.gz
"""

import argparse
import os
import wandb
import torch
from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data import get_datasets
from utils import compute_metrics, id2label, label2id, NUM_LABELS

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-cased"
OUTPUT_DIR    = "./results"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN     = "distilbert-run-1"

HYPERPARAMS = {
    "model"        : MODEL_NAME,
    "epochs"       : 3,
    "batch_size"   : 16,
    "learning_rate": 3e-5,
    "max_length"   : 128,
    "dataset"      : "UCSD Goodreads",
    "warmup_steps" : 100,
    "weight_decay" : 0.01,
}


def get_device():
    if torch.cuda.is_available():
        print("[train.py] Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("[train.py] No GPU found — using CPU (consider reducing SAMPLES_PER_GENRE to 200)")
    return torch.device("cpu")


def load_model(device):
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    print(f"[train.py] Model loaded: {MODEL_NAME} | Labels: {NUM_LABELS}")
    return model


def build_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Core hyperparameters
        num_train_epochs=HYPERPARAMS["epochs"],
        per_device_train_batch_size=HYPERPARAMS["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=HYPERPARAMS["learning_rate"],
        warmup_steps=HYPERPARAMS["warmup_steps"],
        weight_decay=HYPERPARAMS["weight_decay"],

        # Logging & checkpointing
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        # W&B integration — single line does all logging
        report_to="wandb",
        run_name=WANDB_RUN,
    )


def train(data_path: str):
    # 1. Initialise W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        config=HYPERPARAMS,
    )

    # 2. Load data
    train_dataset, test_dataset, tokenizer = get_datasets(data_path)

    # 3. Load model
    device = get_device()
    model  = load_model(device)

    # 4. Build Trainer
    training_args = build_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("[train.py] Starting training...")
    trainer.train()
    print("[train.py] Training complete.")

    # 6. Save model & tokenizer locally (eval.py will reload them)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[train.py] Model saved to: {OUTPUT_DIR}")

    wandb.finish()
    return trainer, tokenizer, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("GOODREADS_DATA_PATH", "goodreads_reviews_dedup.json.gz"),
        help="Path to the Goodreads dataset file",
    )
    args = parser.parse_args()
    train(args.data_path)
