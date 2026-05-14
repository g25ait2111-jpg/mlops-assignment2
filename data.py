"""
data.py — Data loading, sampling, train/test split, tokenisation
MLOps Assignment 2 | IIT Jodhpur
"""

import os
import json
import random
import gzip
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from utils import label2id, id2label, BookGenreDataset

# ── Config ───────────────────────────────────────────────────────────────────
SAMPLES_PER_GENRE = 500        # Reduce to 200 if running on CPU
TEST_SIZE         = 0.2
RANDOM_SEED       = 42
MAX_LENGTH        = 128        # Shorter = faster; 512 is max for DistilBERT
MODEL_NAME        = "distilbert-base-cased"

# Path to the downloaded Goodreads UCSD dataset file
# Download from: https://mengtingwan.github.io/data/goodreads.html
# File: goodreads_reviews_dedup.json.gz  (or the split genre files)
DATA_PATH = os.environ.get("GOODREADS_DATA_PATH", "goodreads_reviews_dedup.json.gz")


# ── Load & sample data ───────────────────────────────────────────────────────
def load_samples(data_path: str, samples_per_genre: int = SAMPLES_PER_GENRE):
    """
    Reads the gzipped JSON-lines Goodreads file and samples up to
    `samples_per_genre` reviews for each genre in label2id.
    Returns (texts, labels) as flat lists.
    """
    genre_buckets = {genre: [] for genre in label2id}

    opener = gzip.open if data_path.endswith(".gz") else open
    with opener(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            genre = record.get("genre", "")
            review = record.get("review_text", "").strip()

            if genre in genre_buckets and review:
                if len(genre_buckets[genre]) < samples_per_genre:
                    genre_buckets[genre].append(review)

            # Stop early if all buckets are full
            if all(len(v) >= samples_per_genre for v in genre_buckets.values()):
                break

    texts, labels = [], []
    for genre, reviews in genre_buckets.items():
        texts.extend(reviews)
        labels.extend([label2id[genre]] * len(reviews))

    # Shuffle together
    combined = list(zip(texts, labels))
    random.seed(RANDOM_SEED)
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ── Train / test split ───────────────────────────────────────────────────────
def split_data(texts, labels, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED):
    return train_test_split(texts, labels, test_size=test_size, random_state=seed)


# ── Tokenise ─────────────────────────────────────────────────────────────────
def tokenise(texts, tokenizer, max_length: int = MAX_LENGTH):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────
def get_datasets(data_path: str = DATA_PATH):
    """
    Full data pipeline: load → split → tokenise → wrap in Dataset objects.
    Returns (train_dataset, test_dataset, tokenizer).
    """
    print(f"[data.py] Loading data from: {data_path}")
    texts, labels = load_samples(data_path)
    print(f"[data.py] Total samples loaded: {len(texts)}")
    print(f"[data.py] Label distribution: { {id2label[i]: labels.count(i) for i in id2label} }")

    train_texts, test_texts, train_labels, test_labels = split_data(texts, labels)
    print(f"[data.py] Train: {len(train_texts)} | Test: {len(test_texts)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_enc = tokenise(train_texts, tokenizer)
    test_enc  = tokenise(test_texts,  tokenizer)

    train_dataset = BookGenreDataset(train_enc, train_labels)
    test_dataset  = BookGenreDataset(test_enc,  test_labels)

    return train_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    train_ds, test_ds, tok = get_datasets()
    print(f"Train dataset size : {len(train_ds)}")
    print(f"Test  dataset size : {len(test_ds)}")
    print(f"Sample item keys   : {list(train_ds[0].keys())}")
