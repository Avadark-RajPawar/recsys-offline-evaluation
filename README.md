# Recommender Systems: Offline Evaluation Pipeline (Part 1)

This repository contains an automated offline evaluation pipeline for recommender systems built on the **MovieLens 1M** dataset. It establishes non-personalized baseline models and compares their performance against a tuned personalized Collaborative Filtering approach.

## Models Implemented
* **Most Popular (Variant 1 - All-Time):** Non-personalized baseline recommending the most frequently interacted-with movies.
* **Most Popular (Variant 2 - Positive Only):** Non-personalized baseline recommending movies with the highest volume of positive ratings ($\ge 4$ stars).
* **Item-Item KNN:** Personalized collaborative filtering utilizing Cosine Similarity, featuring an automated hyperparameter tuning loop to select the optimal neighborhood size ($k \in \{10, 20, 50\}$) based on validation NDCG.

## Evaluation Methodology
The pipeline applies a strict chronological temporal split (80% Train, 10% Validation, 10% Test) to prevent data leakage. Models generate Top-10 recommendations for each user in the test set, evaluated on:
* **Precision@10**
* **Recall@10**
* **NDCG@10**
* **Catalog Coverage**

## Setup & Execution
1. Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/).
2. Extract the data and place the `ml-1m` folder directly in the project directory (ensure `ratings.dat`, `movies.dat`, and `users.dat` are accessible).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt