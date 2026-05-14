"""
utils.py — Shared helpers: label maps, dataset class, compute_metrics
MLOps Assignment 2 | IIT Jodhpur
"""

import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score


# ── Label mappings ──────────────────────────────────────────────────────────
# These genre labels match the UCSD Goodreads dataset used in the notebook.
# Adjust if your notebook uses different labels.
id2label = {
    0: "children",
    1: "comics_graphic",
    2: "fantasy_paranormal",
    3: "mystery_thriller_crime",
    4: "poetry",
    5: "romance",
    6: "young_adult",
}

label2id = {v: k for k, v in id2label.items()}

NUM_LABELS = len(id2label)


# ── PyTorch Dataset wrapper ──────────────────────────────────────────────────
class BookGenreDataset(Dataset):
    """
    Wraps tokenized encodings + label list into a PyTorch Dataset
    that the Hugging Face Trainer can consume.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Metric computation ───────────────────────────────────────────────────────
def compute_metrics(pred):
    """
    Called by the Hugging Face Trainer after each evaluation step.
    Returns accuracy and weighted F1 score.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
