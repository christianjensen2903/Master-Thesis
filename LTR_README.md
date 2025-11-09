# Learning-to-Rank (LTR) Reranker

This directory contains an implementation of a Learning-to-Rank (LTR) reranker for legal citation retrieval. The LTR model reranks initial retrieval results using rich metadata features from EU case law judgments.

## Overview

The LTR system consists of three main components:

1. **Feature Extractor** (`ltr_feature_extractor.py`): Extracts metadata-based features from judgment pairs
2. **LTR Retriever** (`retrievers/ltr_retriever.py`): Reranks initial retrieval results using a trained LightGBM model
3. **Training Script** (`train_ltr.py`): Generates training data and trains the LTR model

## Features

The LTR model uses the following features for reranking:

### Text Similarity
- Dense similarity score from base retriever

### Temporal Features
- Time difference (days and years)
- Log-transformed time difference

### Metadata Features
- **Authentic Language**: Same language, number of shared languages, one-hot encoding for major languages (FR, DE, EN, IT, NL, ES)
- **Advocate General**: Same advocate general, has advocate general
- **Rapporteur**: Same rapporteur, has rapporteur
- **Parties**: Same applicant, same defendant, has parties
- **Procedure Type**: Same procedure type, one-hot encoding for common types (preliminary ruling, annulment, appeal, infringement)
- **Subject Matter**: Number of shared subjects, Jaccard similarity
- **Case Law**: Number of shared case law references, Jaccard similarity, candidate cites query

### Document Features
- Relative paragraph position (paragraph number / total paragraphs)
- Absolute paragraph numbers
- Paragraph number difference
- Document lengths

## Installation

Ensure you have the required dependencies:

```bash
pip install lightgbm numpy pandas tqdm numba
```

## Usage

### 1. Train the LTR Model

First, train the LTR model using the training script:

```bash
python train_ltr.py \
    --base-retriever dense \
    --model-name checkpoints/simcse_citation_model \
    --output-path checkpoints/ltr/ltr_model.txt \
    --embeddings-path artifacts/ltr_train_embeddings.npy \
    --num-negatives 5 \
    --train-cutoff-year 2018
```

**Arguments:**
- `--base-retriever`: Base retriever for initial ranking (dense, tfidf, or bow)
- `--model-name`: Model name for dense retriever
- `--output-path`: Output path for trained LightGBM model
- `--embeddings-path`: Path to save/load embeddings (optional, for speed)
- `--num-negatives`: Number of negative samples per positive (default: 5)
- `--max-queries`: Maximum training queries for debugging (optional)

The training script will:
1. Load citation data and metadata
2. Generate embeddings using the base retriever
3. For each training query, retrieve candidates and extract features
4. Train a LightGBM ranker with lambdarank objective
5. Save the trained model

### 2. Evaluate the LTR Model

Evaluate the trained model using the example script:

```bash
python example_ltr_usage.py
```

Or integrate into your own code:

```python
from retrievers import DenseRetriever, LTRRetriever
from evaluator import Evaluator

# Initialize base retriever
base_retriever = DenseRetriever(
    model_name="checkpoints/simcse_citation_model",
    max_seq_length=256,
)

# Initialize LTR retriever
ltr_retriever = LTRRetriever(
    base_retriever=base_retriever,
    model_path="checkpoints/ltr/ltr_model.txt",
    judgments_path="data/judgments_cleaned.json",
    rerank_top_k=1000,  # Only rerank top 1000 candidates
)

# Initialize and run evaluator
evaluator = Evaluator(
    retriever=ltr_retriever,
    mode="citation_pairs",
    csv_path="data/par-to-par-cleaned.csv",
    metadata_path="data/par-to-par.json",
    judgments_path="data/judgments_cleaned.json",
    train_cutoff_year=2018,
    top_k=10000,
    save_embeddings_path="artifacts/ltr_embeddings.npy",
)

score = evaluator.run()
```

## Architecture

### Training Pipeline

```
Training Data → Base Retriever (Initial Ranking) → Feature Extraction → LightGBM Training
                                                                            ↓
                                                                    Trained Model
```

### Inference Pipeline

```
Query → Base Retriever → Top K Candidates → Feature Extraction → LTR Reranking → Final Results
```

### Two-Stage Retrieval

The LTR retriever uses a two-stage approach:

1. **First Stage**: Base retriever (e.g., dense retriever) quickly retrieves top K candidates (e.g., K=1000)
2. **Second Stage**: LTR model reranks these candidates using rich metadata features

This approach balances efficiency and effectiveness:
- Fast first-stage retrieval narrows down the search space
- Slower but more accurate LTR reranking on a smaller set

## Model Training Details

### LightGBM Configuration

- **Objective**: LambdaRank (learning-to-rank)
- **Metric**: NDCG@5,10,20
- **Learning Rate**: 0.05
- **Max Depth**: 6
- **Num Leaves**: 31
- **Feature Fraction**: 0.8 (bagging)
- **Bagging Fraction**: 0.8
- **Early Stopping**: 50 rounds

### Training Data Generation

For each training query (citing paragraph):
1. Retrieve top candidates using base retriever
2. Label positives (actual citations) and negatives
3. Sample hard negatives (from top ranks) and random negatives
4. Extract features for all query-candidate pairs

**Negative Sampling Strategy:**
- 70% hard negatives (from top-ranked non-relevant documents)
- 30% random negatives (from lower ranks)
- Default: 5 negatives per positive

## Expected Performance

The LTR model typically improves over the base retriever by incorporating:
- Temporal signals (cases cite older cases)
- Metadata similarities (same language, procedure type, etc.)
- Citation relationships (candidate cites query's cited cases)

Expected improvements:
- +2-5% MAP@10 over dense retriever alone
- Better precision at top ranks (more relevant results in top-10)
- More interpretable ranking (feature importance analysis)

## Feature Importance

After training, the model reports feature importance. Typically important features include:
- `dense_similarity`: Base retriever score
- `time_diff_*`: Temporal features
- `case_law_jaccard`: Shared case law references
- `subject_jaccard`: Shared subject matter
- `same_auth_lang`: Same authentic language

## Tips

1. **Pre-compute embeddings**: Use `--embeddings-path` to save/load embeddings and speed up training
2. **Tune negatives**: Adjust `--num-negatives` to balance training data (more negatives = more training time but better hard negative mining)
3. **Debug with fewer queries**: Use `--max-queries 1000` for quick debugging
4. **Experiment with base retrievers**: Try different base retrievers (dense, tfidf) to see which works best
5. **Rerank top-K**: Adjust `rerank_top_k` in LTRRetriever to balance speed vs accuracy

## Files

- `ltr_feature_extractor.py`: Feature extraction from judgments metadata
- `retrievers/ltr_retriever.py`: LTR retriever implementation
- `train_ltr.py`: Training script for LTR model
- `example_ltr_usage.py`: Example usage with evaluator
- `LTR_README.md`: This file

## Troubleshooting

### Model not loading
- Ensure the model path is correct
- Check that LightGBM is installed: `pip install lightgbm`

### Missing metadata
- Some paragraphs may not have all metadata fields
- Feature extractor handles missing values gracefully (defaults to 0)

### Slow training
- Use pre-computed embeddings (`--embeddings-path`)
- Reduce `--max-queries` for debugging
- Reduce `rerank_top_k` (fewer candidates to rerank)

### Poor performance
- Check base retriever performance first
- Ensure training data quality (positive samples exist)
- Tune LightGBM hyperparameters in `train_ltr.py`

