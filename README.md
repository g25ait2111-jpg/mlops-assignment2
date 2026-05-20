# MLOps Assignment 2 — Hugging Face Fine-Tuning, Experiment Tracking & Model Deployment

**IIT Jodhpur | PGD AI Programme | MLOps**

## Project Description

This project implements a complete MLOps pipeline that fine-tunes DistilBERT for book genre classification, tracks experiments with Weights & Biases, trains the model on Kaggle GPU, and publishes it to Hugging Face Hub. The model predicts the genre of a book given a review excerpt. The emphasis is on the MLOps workflow — experiment tracking, artifact versioning, and model publishing — rather than achieving maximum accuracy.

## Setup Instructions

1. Clone the repository: git clone https://github.com/g25ait2111-jpg/mlops-assignment2.git
2. Install dependencies: pip install -r requirements.txt
3. Training was done on Kaggle Notebooks using free GPU T4 x2. Kaggle Secrets were used to store WANDB_API_KEY and HF_TOKEN securely.

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.61   |
| F1 Score  | 0.60   |
| Eval Loss | 2.289  |

## Links

- Kaggle Notebook: https://www.kaggle.com/code/soumyaag25ait2111/notebookcdfa10d7d5
- Hugging Face Model: https://huggingface.co/Soumyaachanna/distilbert-goodreads-genres
- W&B Dashboard: https://wandb.ai/g25ait2111-iit-jodhpur/mlops-assignment2

## Model Selection Rationale

DistilBERT was chosen because it is 40% smaller and 60% faster than full BERT while retaining approximately 97% of its performance. The distilbert-base-cased variant preserves capitalisation, which is meaningful for book review text where proper nouns and titles appear frequently.
