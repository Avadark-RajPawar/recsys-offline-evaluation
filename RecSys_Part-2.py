import argparse
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse

GROUP_ID = 18
PART_LABEL = "Part 2"
SEEDS = [42, 43, 44]


def load_data(base_dir: str):
    ratings_columns = ["user_id", "item_id", "rating", "timestamp"]
    movies_columns = ["item_id", "title", "genres"]
    users_columns = ["user_id", "gender", "age", "occupation", "zip_code"]

    ratings = pd.read_csv(
        os.path.join(base_dir, "ratings.dat"),
        sep="::",
        names=ratings_columns,
        engine="python",
        encoding="latin-1",
    )
    movies = pd.read_csv(
        os.path.join(base_dir, "movies.dat"),
        sep="::",
        names=movies_columns,
        engine="python",
        encoding="latin-1",
    )
    users = pd.read_csv(
        os.path.join(base_dir, "users.dat"),
        sep="::",
        names=users_columns,
        engine="python",
        encoding="latin-1",
    )
    return ratings, movies, users


def temporal_split_80_10_10(ratings: pd.DataFrame):
    ratings_sorted = ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    total_rows = len(ratings_sorted)
    train_end = int(total_rows * 0.8)
    val_end = int(total_rows * 0.9)

    train_data = ratings_sorted.iloc[:train_end].copy()
    val_data = ratings_sorted.iloc[train_end:val_end].copy()
    test_data = ratings_sorted.iloc[val_end:].copy()
    return train_data, val_data, test_data


def get_ground_truth(eval_df: pd.DataFrame, min_rating: float = 4.0) -> dict:
    relevant_df = eval_df[eval_df["rating"] >= min_rating]
    return relevant_df.groupby("user_id")["item_id"].apply(set).to_dict()


def compute_precision_recall_ndcg_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return 0.0, 0.0, 0.0

    recs_at_k = recommended_items[:k]
    hits = [1 if item in relevant_items else 0 for item in recs_at_k]
    num_hits = sum(hits)

    precision = num_hits / k
    recall = num_hits / len(relevant_items)

    dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
    ideal_hits = [1] * min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(len(ideal_hits)))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return precision, recall, ndcg


def evaluate_recommendations(
    rec_function,
    eval_df: pd.DataFrame,
    all_catalog_items: set,
    k: int = 10,
    min_rating: float = 4.0,
    user_filter: set = None,
    item_filter: set = None,
):
    ground_truth = get_ground_truth(eval_df, min_rating=min_rating)
    precisions, recalls, ndcgs = [], [], []
    all_recommended_items = set()

    for uid, relevant in ground_truth.items():
        if user_filter is not None and uid not in user_filter:
            continue
        target_rel = {it for it in relevant if it in item_filter} if item_filter is not None else relevant
        if not target_rel:
            continue

        recs = rec_function(uid)[:k]
        for item in recs:
            all_recommended_items.add(item)

        p, r, n = compute_precision_recall_ndcg_at_k(recs, target_rel, k=k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    coverage = len(all_recommended_items) / len(all_catalog_items) if all_catalog_items else 0.0
    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": float(coverage),
    }


# ---------------------------------------------------------
# 1. Self-Implemented BPR-MF (From Scratch, Ranking-Loss CF)
# ---------------------------------------------------------
class SelfImplementedBPRMF:
    def __init__(self, n_factors=32, lr=0.02, reg=0.01, epochs=6, seed=42, min_rating=4.0):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed
        self.min_rating = min_rating
        self.rng = np.random.default_rng(seed)
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = []
        self.U = None
        self.V = None
        self.item_bias = None
        self.user_pos = {}
        self.user_seen = {}

    def fit(self, train_df: pd.DataFrame, catalog_items):
        users = np.sort(train_df["user_id"].unique())
        items = np.sort(np.array(list(catalog_items)))

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = items

        n_users = len(users)
        n_items = len(items)
        self.U = self.rng.normal(0.0, 0.05, size=(n_users, self.n_factors))
        self.V = self.rng.normal(0.0, 0.05, size=(n_items, self.n_factors))
        self.item_bias = np.zeros(n_items, dtype=np.float64)

        pos_map = defaultdict(list)
        seen_map = defaultdict(set)
        for row in train_df.itertuples(index=False):
            if row.user_id not in self.user2idx or row.item_id not in self.item2idx:
                continue
            uidx = self.user2idx[row.user_id]
            iidx = self.item2idx[row.item_id]
            seen_map[uidx].add(iidx)
            if row.rating >= self.min_rating:
                pos_map[uidx].append(iidx)

        self.user_pos = {u: np.array(v, dtype=np.int64) for u, v in pos_map.items() if len(v) > 0}
        self.user_seen = seen_map

        active_users = list(self.user_pos.keys())
        if not active_users or n_items == 0:
            return

        n_samples = max(len(train_df), len(active_users) * 10)
        for _ in range(self.epochs):
            sampled_users = self.rng.choice(active_users, size=n_samples, replace=True)
            for u in sampled_users:
                pos_arr = self.user_pos[u]
                i = int(pos_arr[int(self.rng.integers(0, len(pos_arr)))])

                seen_u = self.user_seen[u]
                while True:
                    j = int(self.rng.integers(0, n_items))
                    if j not in seen_u:
                        break

                score_diff = (self.item_bias[i] + np.dot(self.U[u], self.V[i])) - (
                    self.item_bias[j] + np.dot(self.U[u], self.V[j])
                )
                # Derivative of -log(sigmoid(x)) is -sigmoid(-x)
                if score_diff >= 0:
                    exp_neg = np.exp(-score_diff)
                    grad = exp_neg / (1.0 + exp_neg)
                else:
                    exp_pos = np.exp(score_diff)
                    grad = 1.0 / (1.0 + exp_pos)

                u_old = self.U[u].copy()
                v_i_old = self.V[i].copy()
                v_j_old = self.V[j].copy()

                self.U[u] += self.lr * (grad * (v_i_old - v_j_old) - self.reg * u_old)
                self.V[i] += self.lr * (grad * u_old - self.reg * v_i_old)
                self.V[j] += self.lr * (-grad * u_old - self.reg * v_j_old)
                self.item_bias[i] += self.lr * (grad - self.reg * self.item_bias[i])
                self.item_bias[j] += self.lr * (-grad - self.reg * self.item_bias[j])

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user2idx or self.U is None:
            return []
        uidx = self.user2idx[user_id]
        scores = self.item_bias + self.V @ self.U[uidx]
        seen = self.user_seen.get(uidx, set())
        if seen:
            scores[list(seen)] = -np.inf
        top_idx = np.argpartition(-scores, range(min(top_n, len(scores))))[:top_n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(self.idx2item[i]) for i in top_idx if np.isfinite(scores[i])]


# ---------------------------------------------------------
# 2. User-Based kNN (Neighbourhood CF)
# ---------------------------------------------------------
class UserKNNModel:
    def __init__(self, k=30):
        self.k = k
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = []
        self.R = None
        self.sim = None
        self.user_seen = {}

    def fit(self, train_df: pd.DataFrame, catalog_items):
        users = np.sort(train_df["user_id"].unique())
        items = np.sort(np.array(list(catalog_items)))
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = items

        rows, cols, vals = [], [], []
        seen_map = defaultdict(set)
        for row in train_df.itertuples(index=False):
            if row.user_id in self.user2idx and row.item_id in self.item2idx:
                u = self.user2idx[row.user_id]
                it = self.item2idx[row.item_id]
                rows.append(u)
                cols.append(it)
                vals.append(float(row.rating))
                seen_map[u].add(it)

        self.user_seen = seen_map
        self.R = sparse.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
        norms = sparse.linalg.norm(self.R, axis=1)
        norms[norms == 0] = 1.0
        R_norm = self.R.multiply(1.0 / norms[:, None]).tocsr()
        sim_mat = (R_norm @ R_norm.T).toarray()
        np.fill_diagonal(sim_mat, 0.0)
        self.sim = sim_mat

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user2idx or self.sim is None:
            return []
        u = self.user2idx[user_id]
        sims = self.sim[u]
        if self.k < len(sims):
            top_k_idx = np.argpartition(-sims, self.k)[: self.k]
            weights = np.zeros_like(sims)
            weights[top_k_idx] = np.maximum(sims[top_k_idx], 0.0)
        else:
            weights = np.maximum(sims, 0.0)

        denom = np.sum(weights)
        if denom <= 0:
            return []
        scores = (weights @ self.R.toarray()) / denom
        seen = self.user_seen.get(u, set())
        if seen:
            scores[list(seen)] = -np.inf
        top_idx = np.argpartition(-scores, range(min(top_n, len(scores))))[:top_n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(self.idx2item[i]) for i in top_idx if np.isfinite(scores[i])]


# ---------------------------------------------------------
# 3. Content-Based Profile Matching (Genres + Year)
# ---------------------------------------------------------
class ContentBasedModel:
    def __init__(self, movies_df: pd.DataFrame, min_rating: float = 4.0):
        self.movies_df = movies_df
        self.min_rating = min_rating
        self.item2idx = {}
        self.idx2item = []
        self.item_feats = None
        self.user_profiles = {}
        self.user_seen = defaultdict(set)

    @staticmethod
    def _extract_year(title: str) -> int:
        match = re.search(r"\((\d{4})\)\s*$", str(title))
        return int(match.group(1)) if match else 1995

    def fit(self, train_df: pd.DataFrame, catalog_items):
        items = np.sort(np.array(list(catalog_items)))
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = items

        genres_set = sorted(
            {g.strip() for gs in self.movies_df["genres"].dropna() for g in str(gs).split("|") if g.strip()}
        )
        genre_to_col = {g: i for i, g in enumerate(genres_set)}

        years = [self._extract_year(t) for t in self.movies_df["title"]]
        min_y, max_y = (min(years), max(years)) if years else (1920, 2000)
        span = max(max_y - min_y, 1)

        feats = np.zeros((len(items), len(genres_set) + 1), dtype=np.float64)
        for row in self.movies_df.itertuples(index=False):
            if row.item_id not in self.item2idx:
                continue
            idx = self.item2idx[row.item_id]
            for g in str(row.genres).split("|"):
                g = g.strip()
                if g in genre_to_col:
                    feats[idx, genre_to_col[g]] = 1.0
            feats[idx, -1] = (self._extract_year(row.title) - min_y) / span

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_feats = feats / norms

        user_pos_indices = defaultdict(list)
        for row in train_df.itertuples(index=False):
            if row.item_id in self.item2idx:
                iidx = self.item2idx[row.item_id]
                self.user_seen[row.user_id].add(iidx)
                if row.rating >= self.min_rating:
                    user_pos_indices[row.user_id].append(iidx)

        for uid, pos_list in user_pos_indices.items():
            prof = self.item_feats[pos_list].mean(axis=0)
            p_norm = np.linalg.norm(prof)
            if p_norm > 0:
                self.user_profiles[uid] = prof / p_norm

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_profiles or self.item_feats is None:
            return []
        scores = self.item_feats @ self.user_profiles[user_id]
        seen = self.user_seen.get(user_id, set())
        if seen:
            scores[list(seen)] = -np.inf
        top_idx = np.argpartition(-scores, range(min(top_n, len(scores))))[:top_n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(self.idx2item[i]) for i in top_idx if np.isfinite(scores[i])]


def make_result_row(model_name, variant_notes, k, similarity, metrics):
    return {
        "group_id": GROUP_ID,
        "part": PART_LABEL,
        "model_name": model_name,
        "variant_notes": variant_notes,
        "k": k,
        "similarity": similarity,
        "precision_at_10": round(metrics["precision"], 4),
        "recall_at_10": round(metrics["recall"], 4),
        "ndcg_at_10": round(metrics["ndcg"], 4),
        "coverage": round(metrics["coverage"], 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Group 18 Part 2 Recommender Evaluation")
    parser.add_argument("--data-dir", required=True, help="Directory containing ratings.dat, movies.dat, users.dat")
    parser.add_argument("--output-csv", default="results.csv", help="Path to output CSV file")
    args = parser.parse_args()

    ratings, movies, _ = load_data(args.data_dir)
    train_data, val_data, test_data = temporal_split_80_10_10(ratings)
    all_catalog_items = set(movies["item_id"].unique())

    # 1. Validation-only tuning with equal search budget (3 configs per tunable model)
    bpr_grid = [(16, 0.02, 0.02), (32, 0.02, 0.01), (64, 0.015, 0.01)]
    best_bpr_cfg = bpr_grid[0]
    best_bpr_val_ndcg = -1.0
    for factors, lr, reg in bpr_grid:
        m = SelfImplementedBPRMF(n_factors=factors, lr=lr, reg=reg, epochs=5, seed=SEEDS[0])
        m.fit(train_data, all_catalog_items)
        val_res = evaluate_recommendations(lambda u, mod=m: mod.recommend(u, 10), val_data, all_catalog_items, k=10)
        if val_res["ndcg"] > best_bpr_val_ndcg:
            best_bpr_val_ndcg = val_res["ndcg"]
            best_bpr_cfg = (factors, lr, reg)

    uknn_grid = [10, 30, 50]
    best_uknn_k = uknn_grid[0]
    best_uknn_val_ndcg = -1.0
    for k_cand in uknn_grid:
        m = UserKNNModel(k=k_cand)
        m.fit(train_data, all_catalog_items)
        val_res = evaluate_recommendations(lambda u, mod=m: mod.recommend(u, 10), val_data, all_catalog_items, k=10)
        if val_res["ndcg"] > best_uknn_val_ndcg:
            best_uknn_val_ndcg = val_res["ndcg"]
            best_uknn_k = k_cand

    rows = []

    # 2. Multi-seed evaluation on full training set -> test set (with 95% CI in console)
    bpr_seed_ndcgs = []
    for s in SEEDS:
        bpr_m = SelfImplementedBPRMF(
            n_factors=best_bpr_cfg[0], lr=best_bpr_cfg[1], reg=best_bpr_cfg[2], epochs=6, seed=s
        )
        bpr_m.fit(train_data, all_catalog_items)
        res = evaluate_recommendations(lambda u, mod=bpr_m: mod.recommend(u, 10), test_data, all_catalog_items, k=10)
        bpr_seed_ndcgs.append(res["ndcg"])
        if s == SEEDS[0]:
            bpr_primary = bpr_m
            bpr_primary_res = res

    ci_margin = 1.96 * (np.std(bpr_seed_ndcgs, ddof=1) / np.sqrt(len(bpr_seed_ndcgs)))
    print(
        f"BPR-MF (Self-Implemented) Test NDCG@10 across seeds {SEEDS}: "
        f"{np.mean(bpr_seed_ndcgs):.4f} +/- {ci_margin:.4f} (95% CI)"
    )

    uknn_m = UserKNNModel(k=best_uknn_k)
    uknn_m.fit(train_data, all_catalog_items)
    uknn_res = evaluate_recommendations(lambda u: uknn_m.recommend(u, 10), test_data, all_catalog_items, k=10)

    cb_m = ContentBasedModel(movies)
    cb_m.fit(train_data, all_catalog_items)
    cb_res = evaluate_recommendations(lambda u: cb_m.recommend(u, 10), test_data, all_catalog_items, k=10)

    rows.append(
        make_result_row(
            "BPR-MF (Self-Implemented)",
            f"Full train; d={best_bpr_cfg[0]},lr={best_bpr_cfg[1]},reg={best_bpr_cfg[2]}",
            best_bpr_cfg[0],
            "BPR-DotProduct",
            bpr_primary_res,
        )
    )
    rows.append(
        make_result_row(
            "User-KNN",
            f"Full train; val-tuned k={best_uknn_k}",
            best_uknn_k,
            "Cosine",
            uknn_res,
        )
    )
    rows.append(
        make_result_row(
            "Content-Based Profile",
            "Full train; genres + release year",
            "N/A",
            "Cosine",
            cb_res,
        )
    )

    # 3. Disaggregation: (a) Low vs High activity users, (b) Popular vs Long-tail items
    user_counts = train_data["user_id"].value_counts()
    u_med = user_counts.median()
    low_act_users = set(user_counts[user_counts <= u_med].index)
    high_act_users = set(user_counts[user_counts > u_med].index)

    for label, u_set in [("Low-activity users", low_act_users), ("High-activity users", high_act_users)]:
        s_res = evaluate_recommendations(
            lambda u: bpr_primary.recommend(u, 10), test_data, all_catalog_items, k=10, user_filter=u_set
        )
        rows.append(
            make_result_row(
                "BPR-MF (Self-Implemented)",
                f"Slice: {label}",
                best_bpr_cfg[0],
                "BPR-DotProduct",
                s_res,
            )
        )

    item_counts = train_data["item_id"].value_counts()
    i_med = item_counts.median()
    pop_items = set(item_counts[item_counts > i_med].index)
    tail_items = set(item_counts[item_counts <= i_med].index)

    for label, i_set in [("Popular items", pop_items), ("Long-tail items", tail_items)]:
        s_res = evaluate_recommendations(
            lambda u: bpr_primary.recommend(u, 10), test_data, all_catalog_items, k=10, item_filter=i_set
        )
        rows.append(
            make_result_row(
                "BPR-MF (Self-Implemented)",
                f"Slice: {label}",
                best_bpr_cfg[0],
                "BPR-DotProduct",
                s_res,
            )
        )

    # 4. Sparsity ablation: 50% and 20% of training interactions
    for frac in [0.50, 0.20]:
        sub_train = train_data.sample(frac=frac, random_state=SEEDS[0])

        bpr_sub = SelfImplementedBPRMF(
            n_factors=best_bpr_cfg[0], lr=best_bpr_cfg[1], reg=best_bpr_cfg[2], epochs=6, seed=SEEDS[0]
        )
        bpr_sub.fit(sub_train, all_catalog_items)
        bpr_sub_res = evaluate_recommendations(
            lambda u, mod=bpr_sub: mod.recommend(u, 10), test_data, all_catalog_items, k=10
        )
        rows.append(
            make_result_row(
                "BPR-MF (Self-Implemented)",
                f"Sparsity {int(frac * 100)}% train",
                best_bpr_cfg[0],
                "BPR-DotProduct",
                bpr_sub_res,
            )
        )

        uknn_sub = UserKNNModel(k=best_uknn_k)
        uknn_sub.fit(sub_train, all_catalog_items)
        uknn_sub_res = evaluate_recommendations(
            lambda u, mod=uknn_sub: mod.recommend(u, 10), test_data, all_catalog_items, k=10
        )
        rows.append(
            make_result_row(
                "User-KNN",
                f"Sparsity {int(frac * 100)}% train",
                best_uknn_k,
                "Cosine",
                uknn_sub_res,
            )
        )

        # Include Part 1 baselines in the standard 10-column schema
        all_counts = train_data["item_id"].value_counts()
        pop_all_top10 = all_counts.head(10).index.tolist()

        pos_train = train_data[train_data["rating"] >= 4]
        pop_pos_top10 = pos_train["item_id"].value_counts().head(10).index.tolist()

        pop_all_res = evaluate_recommendations(lambda _: pop_all_top10, test_data, all_catalog_items, k=10)
        pop_pos_res = evaluate_recommendations(lambda _: pop_pos_top10, test_data, all_catalog_items, k=10)

        part1_rows = [
            {
                "group_id": GROUP_ID,
                "part": "Part 1",
                "model_name": "Most Popular",
                "variant_notes": "All-time popularity (train)",
                "k": "N/A",
                "similarity": "N/A",
                "precision_at_10": round(pop_all_res["precision"], 4),
                "recall_at_10": round(pop_all_res["recall"], 4),
                "ndcg_at_10": round(pop_all_res["ndcg"], 4),
                "coverage": round(pop_all_res["coverage"], 4),
            },
            {
                "group_id": GROUP_ID,
                "part": "Part 1",
                "model_name": "Most Popular",
                "variant_notes": "Positive ratings (>=4) popularity (train)",
                "k": "N/A",
                "similarity": "N/A",
                "precision_at_10": round(pop_pos_res["precision"], 4),
                "recall_at_10": round(pop_pos_res["recall"], 4),
                "ndcg_at_10": round(pop_pos_res["ndcg"], 4),
                "coverage": round(pop_pos_res["coverage"], 4),
            },
        ]

        final_df = pd.DataFrame(part1_rows + rows)
        final_df.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(final_df)} rows (Part 1 + Part 2) to {args.output_csv}")


if __name__ == "__main__":
    main()