# MLOps Assignment 2 — Hugging Face Fine-Tuning, Experiment Tracking & Model Deployment

**IIT Jodhpur | PGD AI Programme | MLOps**

A complete MLOps pipeline: fine-tune DistilBERT for book genre classification, track experiments with Weights & Biases, and publish the model to Hugging Face Hub.

---

## Project Description

This project implements a multi-class text classifier that predicts the genre of a book given a review excerpt. The model is based on **DistilBERT** (a lighter, faster variant of BERT) fine-tuned on the UCSD Goodreads dataset across 7 genre categories. The emphasis is on the MLOps workflow — experiment tracking, artifact versioning, and model publishing — rather than achieving maximum accuracy.

---

## Repository Structure

```
├── data.py           # Data loading, sampling, train/test split, tokenisation
├── train.py          # Model loading, W&B init, Trainer setup, training loop
├── eval.py           # Evaluation, metrics logging, W&B Artifact upload, HF Hub push
├── utils.py          # Shared helpers: label maps, Dataset class, compute_metrics
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export GOODREADS_DATA_PATH=/path/to/goodreads_reviews_dedup.json.gz
export HF_TOKEN=your_huggingface_write_token
export HF_USERNAME=your_huggingface_username
export WANDB_API_KEY=your_wandb_api_key
```

### 4. Run training
```bash
python train.py --data_path $GOODREADS_DATA_PATH
```

### 5. Run evaluation + push to HF Hub
```bash
python eval.py \
  --data_path   $GOODREADS_DATA_PATH \
  --hf_username $HF_USERNAME \
  --hf_token    $HF_TOKEN
```

> **GPU tip:** Run on Google Colab (free T4 GPU). On CPU, set `SAMPLES_PER_GENRE = 200` in `data.py` to keep training time manageable.

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.XX   |
| F1 Score  | 0.XX   |
| Eval Loss | 0.XX   |

*(Fill in your actual numbers after training)*

---

## Links

- **Hugging Face model:** https://huggingface.co/YOUR_USERNAME/distilbert-goodreads-genres
- **W&B dashboard:** https://wandb.ai/YOUR_USERNAME/mlops-assignment2

---

## Model Selection Rationale

DistilBERT was chosen because it is 40% smaller and 60% faster than full BERT while retaining ~97% of its performance on downstream tasks. For a constrained training environment (Colab free tier), this makes fine-tuning feasible within a single session. The `distilbert-base-cased` variant preserves capitalisation, which is meaningful for book review text where proper nouns and titles appear frequently.
