import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']

# Dataset pathD
base_dir = r'C:\Users\Avadark\Downloads\ml-1m\ml-1m'

ratings = pd.read_csv(
    os.path.join(base_dir, 'ratings.dat'),
    sep='::',
    names=ratings_columns,
    engine='python',
    encoding='latin-1'
)

# 1. Load Movies
movies_columns = ['item_id', 'title', 'genres']
movies = pd.read_csv(
    os.path.join(base_dir, 'movies.dat'),
    sep='::',
    names=movies_columns,
    engine='python',
    encoding='latin-1'
)

# 2. Load Users
users_columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv(
    os.path.join(base_dir, 'users.dat'),
    sep='::',
    names=users_columns,
    engine='python',
    encoding='latin-1'
)

# print(ratings.head(),"\n")
# print(movies.head(),"\n")
# print(users.head(),"\n")

# 1. Basic Stats & Matrix Density
num_users = ratings['user_id'].nunique()
num_items = ratings['item_id'].nunique()
num_interactions = len(ratings)

# Density is the percentage of filled cells in the user-item matrix
density = num_interactions / (num_users * num_items)
sparsity = 1 - density

print(f"Users: {num_users}")
print(f"Items: {num_items}")
print(f"Interactions: {num_interactions}")
print(f"Matrix Density: {density:.2%} (Sparsity: {sparsity:.2%})")

# 2. The Long Tail Plot (Interactions per Item)
# This counts how many ratings each item got, sorts them from highest to lowest, and plots the curve.
item_interaction_counts = ratings['item_id'].value_counts().values

plt.figure(figsize=(10, 5))
plt.plot(item_interaction_counts)
plt.title("Long Tail: Interactions per Item")
plt.xlabel("Items (Sorted by popularity)")
plt.ylabel("Number of Ratings")
plt.grid(True, alpha=0.3)
plt.show()

Ratings_count = ratings['rating'].value_counts().sort_index()

# 3. Rating Distribution Plot
# This shows how many 1-star, 2-star, etc., ratings exist in the dataset.
plt.figure(figsize=(8, 4))
Ratings_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Ratings")
plt.xlabel("Rating (1 to 5)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# print(Ratings_count)

# Time span analysis
# Convert Unix timestamps to human-readable dates
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
print(f"Time span: {ratings['datetime'].min()} to {ratings['datetime'].max()}")

# Plot activity over time (Ratings per Month)
plt.figure(figsize=(10, 4))
ratings['datetime'].dt.to_period('M').value_counts().sort_index().plot(linewidth=2)
plt.title("Activity Over Time (Ratings per Month)")
plt.xlabel("Month")
plt.ylabel("Number of Ratings")
plt.grid(True, alpha=0.3)
plt.show()

# Temporal Split (Train / Validation / Test)
# Sorting the entire dataset chronologically
ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)

# Calculate indices for 80% train, 10% validation, 10% test
total_rows = len(ratings_sorted)
train_end = int(total_rows * 0.8)
val_end = int(total_rows * 0.9)

# Split the data
train_data = ratings_sorted.iloc[:train_end]
val_data = ratings_sorted.iloc[train_end:val_end]
test_data = ratings_sorted.iloc[val_end:]

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

# Most Popular Movies
all_counts = train_data['item_id'].value_counts()
variant_1_top10 = all_counts.head(10).index.tolist()

for item_id in variant_1_top10:
    title = movies[movies['item_id'] == item_id]['title'].values[0]
    print(f"- {title} (ID: {item_id})")

# Positive Popularity (Only ratings >= 4)
positive_train_data = train_data[train_data['rating'] >= 4]
positive_counts = positive_train_data['item_id'].value_counts()
positive_top10 = positive_counts.head(10).index.tolist()

for item_id in positive_top10:
    title = movies[movies['item_id'] == item_id]['title'].values[0]
    print(f"- {title} (ID: {item_id})")

# Create the User-Item matrix (items as rows, users as columns)
item_user_matrix = train_data.pivot(index='item_id', columns='user_id', values='rating').fillna(0)


# Fit the KNN model directly using scikit-learn
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(item_user_matrix)

# Find the 5 most similar movies to any target movie
target_movie_id = 1
distances, indices = knn.kneighbors(item_user_matrix.loc[[target_movie_id]], n_neighbors=6)

# Display recommendations
print(f"Movies most similar to {movies[movies['item_id'] == target_movie_id]['title'].values[0]}:")
for idx in indices.flatten()[1:]:  # Skip index 0 because it's the movie itself
    rec_id = item_user_matrix.index[idx]
    title = movies[movies['item_id'] == rec_id]['title'].values[0]
    print(f"- {title} (ID: {rec_id})")

# Compute pairwise cosine similarity between all items at once
sim_matrix = cosine_similarity(item_user_matrix)
item_sim_df = pd.DataFrame(sim_matrix, index=item_user_matrix.index, columns=item_user_matrix.index)


def recommend_item_knn(user_id, train_df, similarity_df, top_n=10, k=20, min_rating=4):
    # Step A: Identify movies the user positively rated in the training set
    user_ratings = train_df[train_df['user_id'] == user_id]
    liked_items = user_ratings[user_ratings['rating'] >= min_rating]['item_id'].values

    # Cold-start check: user has no positive ratings in the training split
    if len(liked_items) == 0:
        return []

    # Step B: Filter out all movies the user has already rated
    seen_items = set(user_ratings['item_id'].values)

    # Step C: Aggregate similarity scores for candidate unseen movies
    valid_liked = [item for item in liked_items if item in similarity_df.index]
    if not valid_liked:
        return []

    # Extract similarity sub-matrix between the user's liked movies and all items
    sub_sim = similarity_df.loc[valid_liked]

    candidate_scores = {}
    for candidate_id in similarity_df.columns:
        if candidate_id in seen_items:
            continue  # Do not recommend already watched movies

        # Similarities between the candidate and all items the user liked
        sims = sub_sim[candidate_id].values

        # Keep only the top-k highest similarity values
        top_k_sims = np.sort(sims)[::-1][:k]

        # Aggregate the score
        score = np.sum(top_k_sims)
        if score > 0:
            candidate_scores[candidate_id] = score

    # Step D: Sort candidate items descending by total score and return top-N
    ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [item_id for item_id, _ in ranked_candidates]



# Creating relevant data frame
def get_ground_truth(eval_df, min_rating=4):
    relevant_df = eval_df[eval_df['rating'] >= min_rating]
    ground_truth = relevant_df.groupby('user_id')['item_id'].apply(set).to_dict()
    return ground_truth


# METRIC COMPUTATION (PRECISION, RECALL, NDCG)
def compute_precision_recall_ndcg_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return 0.0, 0.0, 0.0

    # Truncate recommendations to K
    recs_at_k = recommended_items[:k]

    # Hits: check which recommended items are in the ground truth set
    hits = [1 if item in relevant_items else 0 for item in recs_at_k]
    num_hits = sum(hits)

    # Precision@K
    precision = num_hits / k

    # Recall@K
    recall = num_hits / len(relevant_items)

    # DCG@K: sum(hit / log2(rank + 1))
    # Rank starts at 1, so index 0 is rank 1 -> log2(1 + 1) = log2(2) = 1
    dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])

    # IDCG@K: Ideal DCG (all hits placed at the very top)
    ideal_hits = [1] * min(len(relevant_items), k)
    idcg = sum([1.0 / np.log2(idx + 2) for idx in range(len(ideal_hits))])

    # NDCG@K
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, ndcg


def evaluate_recommendations(rec_function, eval_df, all_catalog_items, k=10):
    ground_truth = get_ground_truth(eval_df)
    precisions = []
    recalls = []
    ndcgs = []

    # Use a set to collect every unique movie recommended to any user
    all_recommended_items = set()

    eval_users = list(ground_truth.keys())

    for uid in eval_users:
        relevant = ground_truth[uid]
        recs = rec_function(uid)

        # Track items for coverage
        for item in recs[:k]:
            all_recommended_items.add(item)

        p, r, n = compute_precision_recall_ndcg_at_k(recs, relevant, k=k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    # Coverage percentage
    total_catalog_size = len(all_catalog_items)
    coverage = len(all_recommended_items) / total_catalog_size if total_catalog_size > 0 else 0.0

    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"NDCG@{k}": np.mean(ndcgs),
        f"Coverage@{k}": coverage,
        "Users Evaluated": len(eval_users)
    }

# All unique movies available in the dataset catalog
all_unique_movies = set(movies['item_id'].unique())

# Hyperparameter Tuning

k_values = [10, 20, 50]
tuning_results = {}

print("--- Starting Hyperparameter Tuning on Validation Data ---")

for k_val in k_values:
    print(f"\nEvaluating Item-KNN with neighborhood size k={k_val}...")


    def knn_fn(uid, current_k=k_val):
        # Whenever the grader passes a uid, fill in all the fixed tables automatically:
        return recommend_item_knn(
            uid,
            train_data,
            item_sim_df,
            top_n=10,
            k=current_k,
            min_rating=4
        )

    res = evaluate_recommendations(knn_fn, val_data, all_unique_movies, k=10)
    tuning_results[k_val] = res

    print(
        f"k={k_val} -> NDCG@10: {res['NDCG@10']:.4f} | Recall@10: {res['Recall@10']:.4f} | Coverage@10: {res['Coverage@10']:.2%}")

# Pick the best k based on NDCG@10
best_k = max(tuning_results.keys(), key=lambda k: tuning_results[k]['NDCG@10'])
print(f"\nOptimal neighborhood size chosen: k = {best_k}")

print("\n==========================================")
print("FINAL BENCHMARK EVALUATION ON TEST DATA")
print("==========================================")

# Baseline 1 wrapper
def rec_popular_v1(uid):
    return variant_1_top10

# Baseline 2 wrapper
def rec_popular_v2(uid):
    return positive_top10

# Best Item-KNN wrapper
def rec_best_knn(uid):
    return recommend_item_knn(
        user_id=uid,
        train_df=train_data,
        similarity_df=item_sim_df,
        top_n=10,
        k=best_k,
        min_rating=4
    )

# Evaluate all three on unseen test_data
test_res_v1 = evaluate_recommendations(rec_popular_v1, test_data, all_unique_movies, k=10)
test_res_v2 = evaluate_recommendations(rec_popular_v2, test_data, all_unique_movies, k=10)
test_res_knn = evaluate_recommendations(rec_best_knn, test_data, all_unique_movies, k=10)

# Build final comparison table
summary_table = pd.DataFrame([
    {
        "Model": "Most Popular (Variant 1: All-Time)",
        "Precision@10": test_res_v1["Precision@10"],
        "Recall@10": test_res_v1["Recall@10"],
        "NDCG@10": test_res_v1["NDCG@10"],
        "Coverage@10": f"{test_res_v1['Coverage@10']:.2%}"
    },
    {
        "Model": "Most Popular (Variant 2: Positive Only)",
        "Precision@10": test_res_v2["Precision@10"],
        "Recall@10": test_res_v2["Recall@10"],
        "NDCG@10": test_res_v2["NDCG@10"],
        "Coverage@10": f"{test_res_v2['Coverage@10']:.2%}"
    },
    {
        "Model": f"Item-KNN (Tuned k={best_k})",
        "Precision@10": test_res_knn["Precision@10"],
        "Recall@10": test_res_knn["Recall@10"],
        "NDCG@10": test_res_knn["NDCG@10"],
        "Coverage@10": f"{test_res_knn['Coverage@10']:.2%}"
    }
])

print("\n--- FINAL PART 1 BENCHMARK RESULTS ---")
print(summary_table.to_string(index=False))


# Export results for the repository requirement
summary_table.to_csv('results.csv', index=False)