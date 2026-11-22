# Recommender Systems: Offline Evaluation Pipeline (Parts 1 & 2)

Offline evaluation pipeline for the Recommender Systems course project on the **MovieLens 1M** dataset (1,000,209 explicit ratings on a 1–5 scale across 6,040 users and 3,706 movies; 4.47% matrix density / 95.53% sparsity).

## Team (Group 18)
* **Suraj Nagdevani** (`1547889743`) — Temporal split pipeline, dataset profiling, and metric modules (`Precision@10`, `Recall@10`, `NDCG@10`, `Coverage@10`)
* **Raj Pawar** (`154816580`) — Matrix transformations, similarity/neighborhood logic, and GitHub repository deployment
* **Himanshu Mahajan** (`154812678`) — Baselines, evaluation logger (`results.csv`), and report/method documentation

**Declared Part 3 Track:** Track A — Diversity and Popularity Bias

## Evaluation Protocol
* **Split:** Global temporal `80% / 10% / 10%` train / validation / test split (no random split to prevent temporal leakage)
* **Relevance Threshold:** Explicit ratings `r >= 4` treated as positive ground truth
* **Ranking & Filtering:** Full-catalog ranking over unseen items without user/item filtering thresholds

## What Was Implemented

### Part 1 — Data, Protocol & Baselines
* Dataset characterization and long-tail distribution analysis on MovieLens 1M
* Popularity baselines (`Most Popular` — all-time frequency and positive-only)
* Neighborhood/baseline evaluation and standardized result logging (`RecSys_Part_1.py`)

### Part 2 — Models, Disaggregated Slices & Sparsity Ablation
* **Models (`RecSys_Part-2.py`):**
  * **BPR-MF (Self-Implemented):** Bayesian Personalized Ranking matrix factorization (`d=64`, `lr=0.015`, `reg=0.01`)
  * **User-KNN:** User-based collaborative filtering (`k=50`, cosine similarity)
  * **Content-Based Profile:** Item content profile using movie genres and release year (cosine similarity)
* **Disaggregated Analysis:** User-activity and item-popularity/long-tail slice evaluation (highlighting cold-start / long-tail sparsity effects)
* **Sparsity Ablation:** Training interaction subsampling across `100%`, `50%`, and `20%` regimes comparing `BPR-MF` and `User-KNN`

## Repository Contents
* `RecSys_Part_1.py` — Part 1 data split, baselines, and evaluation script
* `RecSys_Part-2.py` — Part 2 models (`BPR-MF`, `User-KNN`, `Content-Based Profile`), slice evaluation, and sparsity ablation
* `results.csv` — 10-column evaluation results log (`group_id, part, model_name, variant_notes, k, similarity, precision_at_10, recall_at_10, ndcg_at_10, coverage`)
* `Recommender Systems - Part 2 Cumulative Report.pdf` — Cumulative report for Parts 1 & 2 with Part 3 Track A declared
* `requirements.txt` — Python dependencies

## Running the Pipeline
Execute from the repository root passing the MovieLens 1M directory path (no hardcoded local paths):

```bash
python RecSys_Part-2.py --data-dir /path/to/ml-1m
```
