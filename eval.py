"""
eval.py — Evaluation, metrics logging, W&B Artifact upload, Hugging Face Hub push
MLOps Assignment 2 | IIT Jodhpur

Usage:
    python eval.py --data_path /path/to/goodreads_reviews_dedup.json.gz
                   --hf_username your-hf-username
                   --hf_token    your-hf-token        (or set HF_TOKEN env var)
"""

import argparse
import json
import os

import wandb
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report
from huggingface_hub import login

from data import get_datasets
from utils import compute_metrics, id2label, NUM_LABELS

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR    = "./results"
REPORT_PATH   = "eval_report.json"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN     = "distilbert-eval"
HF_REPO_NAME  = "distilbert-goodreads-genres"


def load_trained_model(output_dir: str = OUTPUT_DIR):
    """Reload the best checkpoint saved by train.py."""
    print(f"[eval.py] Loading model from: {output_dir}")
    model     = DistilBertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(output_dir)
    return model, tokenizer


def run_evaluation(model, test_dataset):
    """Run Trainer.evaluate() on test set — returns dict of metrics."""
    # Minimal TrainingArguments just for evaluation
    eval_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=32,
        report_to="none",   # already logged during training
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    eval_results = trainer.evaluate()
    print("[eval.py] Evaluation results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return trainer, eval_results


def save_classification_report(trainer, test_dataset):
    """Generate and save a per-class classification report as JSON."""
    predictions = trainer.predict(test_dataset)
    preds  = predictions.predictions.argmax(-1)
    labels = [item["labels"].item() for item in test_dataset]

    report = classification_report(
        labels,
        preds,
        target_names=list(id2label.values()),
        output_dict=True,
    )
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval.py] Classification report saved to: {REPORT_PATH}")
    return report


def log_to_wandb(eval_results: dict, hf_url: str = ""):
    """Log final metrics and upload classification report as W&B Artifact."""
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN)

    # Log scalar metrics
    wandb.log({
        "final/loss"    : eval_results.get("eval_loss", 0),
        "final/accuracy": eval_results.get("eval_accuracy", 0),
        "final/f1"      : eval_results.get("eval_f1", 0),
    })

    # Upload classification report as a versioned Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(REPORT_PATH)
    wandb.log_artifact(artifact)
    print("[eval.py] Evaluation report uploaded to W&B as Artifact.")

    # Log HF model URL in run summary
    if hf_url:
        wandb.run.summary["huggingface_model"] = hf_url
        print(f"[eval.py] HF URL logged to W&B summary: {hf_url}")

    wandb.finish()


def push_to_hub(model, tokenizer, hf_username: str, hf_token: str):
    """Push model + tokenizer to Hugging Face Hub."""
    login(token=hf_token)
    repo_id = f"{hf_username}/{HF_REPO_NAME}"
    print(f"[eval.py] Pushing to Hugging Face Hub: {repo_id}")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    hf_url = f"https://huggingface.co/{repo_id}"
    print(f"[eval.py] Model published at: {hf_url}")
    return hf_url


def main(data_path: str, hf_username: str, hf_token: str):
    # 1. Reload trained model
    model, tokenizer = load_trained_model()

    # 2. Load test data (same tokenizer, same split)
    _, test_dataset, _ = get_datasets(data_path)

    # 3. Evaluate
    trainer, eval_results = run_evaluation(model, test_dataset)

    # 4. Save classification report
    save_classification_report(trainer, test_dataset)

    # 5. Push to Hugging Face Hub (if credentials provided)
    hf_url = ""
    if hf_username and hf_token:
        hf_url = push_to_hub(model, tokenizer, hf_username, hf_token)
    else:
        print("[eval.py] Skipping HF push — provide --hf_username and --hf_token to enable.")

    # 6. Log everything to W&B
    log_to_wandb(eval_results, hf_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("GOODREADS_DATA_PATH", "goodreads_reviews_dedup.json.gz"),
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default=os.environ.get("HF_USERNAME", ""),
        help="Your Hugging Face username",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Your Hugging Face write token",
    )
    args = parser.parse_args()
    main(args.data_path, args.hf_username, args.hf_token)
