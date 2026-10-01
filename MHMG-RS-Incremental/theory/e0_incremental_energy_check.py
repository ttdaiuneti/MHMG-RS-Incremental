"""
Numerical check of the full single-object incremental update (radius, granule size,
energy E(B), K_d and gamma; Theorems 1-4).

1. Implements the exact update of the granule size |G_B(x_i)| and of E(B) when one
   object is inserted.
2. Checks it against a batch recomputation every K steps, and records two quantities:
   - friend-gain fan-out: how many existing objects gain a granule member when a
     friend (same class) is inserted;
   - shrink-cascade size: when delta_B(x_i) shrinks, how many former members leave
     the granule.

Uses MHTG_Reducer and three of the benchmark datasets.
"""
import sys
import os
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PUBLISHED_CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "IJAR-MHMG-Published", "IJAR-Code"
)
sys.path.insert(0, os.path.abspath(PUBLISHED_CODE_DIR))
from HMMG_Reducer import MHTG_Reducer  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_v3")

DATASETS = [
    {"id": "D1", "name": "Wine", "file": "1.csv", "subsample_n": None},
    {"id": "D9", "name": "Spambase", "file": "9.csv", "subsample_n": None},
    {"id": "D11", "name": "Isolet", "file": "11.csv", "subsample_n": 1200},
]

SEED = 42
INIT_FRAC = 0.8
GROUND_TRUTH_CHECK_EVERY = 25  # compare with a batch recomputation every 25 steps


def load_dataset(path, subsample_n, seed):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    if subsample_n is not None and len(y) > subsample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=subsample_n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def full_delta_and_granule(dist_matrix, y):
    """Ground truth: the original formula of HMMG_Reducer._calculate_gamma."""
    n = len(y)
    deltas = np.full(n, np.inf)
    for i in range(n):
        enemy_mask = y != y[i]
        ed = dist_matrix[i][enemy_mask]
        if len(ed) > 0:
            deltas[i] = ed.min()
    granule_sizes = np.array([np.sum(dist_matrix[i] < deltas[i]) for i in range(n)])
    return deltas, granule_sizes


def run_for_dataset(ds):
    name, ds_id = ds["name"], ds["id"]
    path = os.path.join(DATASET_DIR, ds["file"])
    print(f"\n=== {ds_id} {name} ===", flush=True)

    X, y = load_dataset(path, ds["subsample_n"], SEED)
    n_total = len(y)

    X = MinMaxScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_init = int(n_total * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]

    reducer = MHTG_Reducer()
    reduct, _, gamma_init = reducer.fit_reduction(X[init_idx], y[init_idx])
    if len(reduct) == 0:
        return {"id": ds_id, "dataset": name, "error": "empty reduct"}
    B = reduct
    XB = X[:, B]
    print(f"  |R_init|={len(B)}  n_init={n_init}  n_stream={len(stream_idx)}")

    # --- preallocate the full distance matrix (n_total x n_total), filled along the stream
    dist_matrix = np.zeros((n_total, n_total), dtype=np.float64)
    active_X = XB[init_idx]
    active_y = y[init_idx].copy()
    d0 = reducer._calculate_distances(active_X)
    dist_matrix[:n_init, :n_init] = d0

    delta, gsize = full_delta_and_granule(d0, active_y)
    E_val = float(np.sum(np.log2(gsize + 1)))

    # --- K_d (denominator of gamma, "max_potential_energy" in HMMG_Reducer.py):
    # K_d = sum_i log2(class_size(y_i)+1) = sum_classes m_c * log2(m_c+1).
    # class_counts are kept to update K_d in O(1) per insertion (Theorem 4):
    # not +log2(m+1) for the new object alone, but (m+1)*log2(m+2) - m*log2(m+1),
    # because each of the m existing objects of the class changes its contribution
    # from log2(m+1) to log2(m+2).
    unique_labels, counts = np.unique(active_y, return_counts=True)
    class_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
    K_d = float(sum(m * np.log2(m + 1) for m in class_counts.values()))

    friend_gain_fanouts = []  # number of existing objects whose granule grows when a friend is inserted
    shrink_cascade_sizes = []  # number of former members that leave the granule when delta shrinks
    ground_truth_diffs = []  # correctness check: |E_incremental - E_batch_groundtruth|
    n_degenerate_zero_delta_new = 0  # count of inserted objects with delta_new==0
    n_degenerate_zero_delta_existing = 0  # count of existing objects whose radius shrinks to 0
    n_current = n_init

    for step, idx in enumerate(stream_idx):
        x_new = XB[idx]
        y_new = y[idx]

        new_dists = np.sqrt(np.sum((active_X - x_new) ** 2, axis=1))
        dist_matrix[n_current, :n_current] = new_dists
        dist_matrix[:n_current, n_current] = new_dists

        friend_mask = active_y == y_new
        enemy_mask = ~friend_mask

        # --- friend (friend gain): radius unchanged; granule +1 if inside the old radius
        gain_mask = friend_mask & (new_dists < delta)
        n_gain = int(gain_mask.sum())
        friend_gain_fanouts.append(n_gain)
        gsize[gain_mask] += 1
        E_val += np.sum(np.log2(gsize[gain_mask] + 1) - np.log2(gsize[gain_mask] - 1 + 1))

        # --- Truong hop KE THU (enemy-shrink): delta co the giam -> granule co the mat thanh vien
        shrink_mask = enemy_mask & (new_dists < delta)
        shrink_idx = np.where(shrink_mask)[0]
        for i in shrink_idx:
            old_delta_i = delta[i]
            new_delta_i = new_dists[i]
            delta[i] = new_delta_i
            if new_delta_i == 0:
                n_degenerate_zero_delta_existing += 1
            # exact new granule size from the cached row dist_matrix[i]
            # (no distance recomputation; the row is in the cache)
            row_i = dist_matrix[i, :n_current]
            # no +1: row_i already includes dist_matrix[i,i]=0 (the object itself);
            # adding 1 would count it twice.
            new_size = int(np.sum(row_i < new_delta_i))
            evicted = gsize[i] - new_size
            shrink_cascade_sizes.append(int(evicted))
            E_val += np.log2(new_size + 1) - np.log2(gsize[i] + 1)
            gsize[i] = new_size

        # --- new object: its own radius and granule
        enemy_dists = new_dists[enemy_mask]
        delta_new = float(enemy_dists.min()) if enemy_dists.size > 0 else np.inf
        friend_dists = new_dists[friend_mask]
        # The object itself is counted only if 0 < delta_new (definition: dist<delta).
        # In the zero-margin case delta_new==0 (an enemy at distance 0, i.e. identical
        # features) the granule is EMPTY, the object itself included.
        self_count = 1 if delta_new > 0 else 0
        if delta_new == 0:
            n_degenerate_zero_delta_new += 1
        gsize_new = int(np.sum(friend_dists < delta_new)) + self_count
        E_val += np.log2(gsize_new + 1)

        # --- Theorem 4: K_d is updated in O(1);
        # m = former size of class y_new.
        m_old = class_counts.get(y_new, 0)
        m_new = m_old + 1
        K_d += (m_new * np.log2(m_new + 1)) - (m_old * np.log2(m_old + 1) if m_old > 0 else 0.0)
        class_counts[y_new] = m_new

        active_X = np.vstack([active_X, x_new[None, :]])
        active_y = np.append(active_y, y_new)
        delta = np.append(delta, delta_new)
        gsize = np.append(gsize, gsize_new)
        n_current += 1

        # --- periodic check against the batch ground truth (E, K_d, gamma)
        if step % GROUND_TRUTH_CHECK_EVERY == 0 or step == len(stream_idx) - 1:
            gt_delta, gt_gsize = full_delta_and_granule(dist_matrix[:n_current, :n_current], active_y)
            gt_E = float(np.sum(np.log2(gt_gsize + 1)))
            gt_labels, gt_counts = np.unique(active_y, return_counts=True)
            gt_K_d = float(np.sum(gt_counts * np.log2(gt_counts + 1)))
            gt_gamma = gt_E / gt_K_d if gt_K_d > 0 else None
            gamma_incremental = E_val / K_d if K_d > 0 else None
            ground_truth_diffs.append(
                {
                    "step": step,
                    "n": n_current,
                    "E_incremental": E_val,
                    "E_groundtruth": gt_E,
                    "abs_diff": abs(E_val - gt_E),
                    "K_d_incremental": K_d,
                    "K_d_groundtruth": gt_K_d,
                    "K_d_abs_diff": abs(K_d - gt_K_d),
                    "gamma_incremental": gamma_incremental,
                    "gamma_groundtruth": gt_gamma,
                    "gamma_abs_diff": (
                        abs(gamma_incremental - gt_gamma)
                        if gamma_incremental is not None and gt_gamma is not None
                        else None
                    ),
                }
            )

    max_gt_diff = max(g["abs_diff"] for g in ground_truth_diffs) if ground_truth_diffs else None
    max_kd_diff = max(g["K_d_abs_diff"] for g in ground_truth_diffs) if ground_truth_diffs else None
    max_gamma_diff = (
        max(g["gamma_abs_diff"] for g in ground_truth_diffs if g["gamma_abs_diff"] is not None)
        if ground_truth_diffs
        else None
    )

    return {
        "id": ds_id,
        "dataset": name,
        "n_total": n_total,
        "n_init": int(n_init),
        "n_stream": int(len(stream_idx)),
        "reduct_size": len(B),
        "gamma_init": gamma_init,
        "n_degenerate_zero_delta_new": n_degenerate_zero_delta_new,
        "n_degenerate_zero_delta_new_frac_of_stream": (
            n_degenerate_zero_delta_new / len(stream_idx) if len(stream_idx) > 0 else None
        ),
        "n_degenerate_zero_delta_existing": n_degenerate_zero_delta_existing,
        "correctness_check_max_abs_K_d_diff": max_kd_diff,
        "correctness_check_max_abs_gamma_diff": max_gamma_diff,
        "friend_gain_fanout": {
            "mean": float(np.mean(friend_gain_fanouts)) if friend_gain_fanouts else None,
            "median": float(np.median(friend_gain_fanouts)) if friend_gain_fanouts else None,
            "max": int(np.max(friend_gain_fanouts)) if friend_gain_fanouts else None,
            "mean_as_frac_of_n": (
                float(np.mean(friend_gain_fanouts) / np.mean([n_init, n_total]))
                if friend_gain_fanouts
                else None
            ),
        },
        "shrink_cascade_size": {
            "n_shrink_events": len(shrink_cascade_sizes),
            "mean": float(np.mean(shrink_cascade_sizes)) if shrink_cascade_sizes else None,
            "median": float(np.median(shrink_cascade_sizes)) if shrink_cascade_sizes else None,
            "max": int(np.max(shrink_cascade_sizes)) if shrink_cascade_sizes else None,
        },
        "correctness_check_max_abs_E_diff": max_gt_diff,
        "correctness_check_n_points": len(ground_truth_diffs),
        "ground_truth_diffs_sample": ground_truth_diffs[:3] + ground_truth_diffs[-3:],
    }


def main():
    results = [run_for_dataset(ds) for ds in DATASETS]
    out_path = os.path.join(os.path.dirname(__file__), "e0_check_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "ground_truth_diffs_sample"}, indent=2))
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
