"""
Experiment following the protocol of Li2026 (Neurocomputing 2026, Section 5.1):

  "the samples of each dataset were divided into a base set and an
  incremental set at a 50% ratio. The incremental set was evenly split into
  five subsets that were added sequentially. After each addition, the new
  data were merged with the current base set to form the base set for the
  next round of incremental operations."

One deliberate difference: Li2026 has an incremental algorithm that updates the
reduct itself (IRABDM); this work does not (the reduct is the output of a search,
not an input). Here B_init is selected on the 50% base set and kept fixed, and only
its evaluation (delta/granule/E/K_d/gamma) is updated as each batch arrives
(Theorem 5). As a reference, B_round is re-selected by the batch reducer on the data
seen up to each round, and Jaccard(B_init, B_round) is recorded per round.

In each round: (a) B_init, incremental (Theorem 5, the whole batch at once): time and
accuracy; (b) B_round, batch search from scratch on the data seen: time, accuracy and
Jaccard(B_init, B_round); (c) check of the incremental gamma(B_init) against a
recomputation (B_round is not checked: it is a different search problem).
"""
import sys
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLISHED_CODE_DIR = os.path.join(HERE, "..", "..", "IJAR-MHMG-Published", "IJAR-Code")
sys.path.insert(0, os.path.abspath(PUBLISHED_CODE_DIR))
from HMMG_Reducer import MHTG_Reducer  # noqa: E402

sys.path.insert(0, HERE)
from e1_pipeline import pairwise_distances_direct, full_delta_and_granule  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "theory"))
from e0_batch_insertion_check import batch_closed_form, kd_batch_update  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_v3")
SEED = 42
N_ROUNDS = 5  # Li2026: "evenly split into five subsets"
BASE_RATIO = 0.5  # Li2026: "50% ratio"

PILOT_DATASETS = [
    {"id": "D1", "name": "Wine", "file": "1.csv"},
    {"id": "D2", "name": "WDBC", "file": "2.csv"},
    {"id": "D3", "name": "Iono", "file": "3.csv"},
    {"id": "D4", "name": "Derm", "file": "4.csv"},
    {"id": "D5", "name": "Sonar", "file": "5.csv"},
    {"id": "D6", "name": "Urban", "file": "6.csv"},
    {"id": "D7", "name": "Musk", "file": "7.csv"},
    {"id": "D8", "name": "Arrhy", "file": "8.csv"},
    {"id": "D9", "name": "Spambase", "file": "9.csv"},
    {"id": "D10", "name": "Parkinsons", "file": "10.csv"},
    {"id": "D12", "name": "Letter", "file": "12.csv"},
    # D11 Isolet excluded: one batch fit_reduction takes ~3000 s at 100%, and six per
    # dataset (B_init + 5 rounds) would take too long; the same choice as for Xu2025b.
]


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    return X, y


def cv_accuracy(X_reduced, y, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc3, accs = [], []
    for train_idx, test_idx in skf.split(X_reduced, y):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_reduced[train_idx], y[train_idx])
        acc3.append(knn.score(X_reduced[test_idx], y[test_idx]))
        svm = SVC(kernel="rbf")
        svm.fit(X_reduced[train_idx], y[train_idx])
        accs.append(svm.score(X_reduced[test_idx], y[test_idx]))
    return float(np.mean(acc3)), float(np.mean(accs))


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def assemble_new_cache(cached_dist_nn, active_X, batch_X):
    """Join the old (n x n) cache and the two new blocks into an (n+k) x (n+k) cache for
    the next round. The O(n^2) copy is acceptable for 5 rounds (this script checks
    correctness, not speed; see e0_batch_size_sweep.py for timing)."""
    n = active_X.shape[0]
    k = batch_X.shape[0]
    diff_ob = active_X[:, None, :] - batch_X[None, :, :]
    D_ob = np.sqrt(np.sum(diff_ob ** 2, axis=2))
    diff_bb = batch_X[:, None, :] - batch_X[None, :, :]
    D_bb = np.sqrt(np.sum(diff_bb ** 2, axis=2))
    n_total = n + k
    full = np.empty((n_total, n_total))
    full[:n, :n] = cached_dist_nn
    full[:n, n:] = D_ob
    full[n:, :n] = D_ob.T
    full[n:, n:] = D_bb
    return full


def run_dataset(ds):
    path = os.path.join(DATASET_DIR, ds["file"])
    X_raw, y = load_dataset(path)
    X = MinMaxScaler().fit_transform(X_raw)
    n_total = len(y)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_base = int(n_total * BASE_RATIO)
    base_idx = perm[:n_base]
    incr_idx = perm[n_base:]
    subsets = np.array_split(incr_idx, N_ROUNDS)

    print(f"=== {ds['name']}: n={n_total}, base={n_base}, incr={len(incr_idx)}, "
          f"{N_ROUNDS} vong ~{len(incr_idx)//N_ROUNDS} moi vong ===")

    # --- B_init: batch fit_reduction on the 50% base (MHTG_Reducer re-scales internally,
    # so raw and min-max scaled X give the same result; the scaled X is used for
    # consistency with the incremental part below) ---
    reducer0 = MHTG_Reducer()
    t0 = time.perf_counter()
    B_init, _, gamma_init_batch = reducer0.fit_reduction(X[base_idx], y[base_idx])
    t_Binit_search = time.perf_counter() - t0
    B_init = list(B_init)
    print(f"  B_init (tren 50% base): {B_init}  (gamma={gamma_init_batch:.4f}, "
          f"{t_Binit_search:.3f}s tim kiem)")

    XB_init = X[:, B_init]

    t_init0 = time.perf_counter()
    active_X = XB_init[base_idx].copy()
    active_y = y[base_idx].copy()
    cache = pairwise_distances_direct(active_X)
    delta, gsize = full_delta_and_granule(cache, active_y)
    E_val = float(np.sum(np.log2(gsize + 1)))
    classes, counts = np.unique(active_y, return_counts=True)
    class_counts = dict(zip(classes.tolist(), counts.tolist()))
    K_d = float(sum(m * np.log2(m + 1) for m in class_counts.values()))
    t_state_init = time.perf_counter() - t_init0  # state initialization

    cumulative_idx = list(base_idx)
    rows = []
    prev_B_round = set(B_init)

    for r, subset in enumerate(subsets, start=1):
        batch_y = y[subset]
        batch_X_Binit = XB_init[subset]

        # --- (a) ours: fixed B_init, batch incremental update (Theorem 5) ---
        t0 = time.perf_counter()
        new_delta, new_gsize, E_val, full_y = batch_closed_form(
            active_X, active_y, cache, delta, gsize, E_val, batch_X_Binit, batch_y
        )
        K_d, class_counts = kd_batch_update(K_d, class_counts, batch_y)
        t_incremental = time.perf_counter() - t0  # kernel: Dinh ly 5 (delta, g, E, K_d)
        gamma_incremental = E_val / K_d

        # includes joining the cache and growing the arrays for the next round
        t0 = time.perf_counter()
        cache = assemble_new_cache(cache, active_X, batch_X_Binit)
        active_X = np.vstack([active_X, batch_X_Binit])
        t_incremental_total = t_incremental + (time.perf_counter() - t0)
        active_y = full_y
        delta, gsize = new_delta, new_gsize
        cumulative_idx = cumulative_idx + list(subset)

        acc3_Binit, accsvm_Binit = cv_accuracy(active_X, active_y)

        # --- check: gamma(B_init) recomputed from scratch on the data seen ---
        d_gt = pairwise_distances_direct(active_X)
        delta_gt, gsize_gt = full_delta_and_granule(d_gt, active_y)
        E_gt = float(np.sum(np.log2(gsize_gt + 1)))
        classes_gt, counts_gt = np.unique(active_y, return_counts=True)
        K_d_gt = float(sum(m * np.log2(m + 1)
                            for m in dict(zip(classes_gt.tolist(), counts_gt.tolist())).values()))
        gamma_gt = E_gt / K_d_gt
        gt_diff = abs(gamma_incremental - gamma_gt)

        # --- (b) B_round: batch fit_reduction from scratch on the data seen ---
        cum_idx_arr = np.array(cumulative_idx)
        reducer_r = MHTG_Reducer()
        t0 = time.perf_counter()
        B_round, _, gamma_round_batch = reducer_r.fit_reduction(X[cum_idx_arr], y[cum_idx_arr])
        t_batch_refit = time.perf_counter() - t0
        B_round = list(B_round)
        acc3_Bround, accsvm_Bround = cv_accuracy(X[cum_idx_arr][:, B_round], y[cum_idx_arr])

        jac_vs_init = jaccard(B_init, B_round)
        jac_vs_prev_round = jaccard(prev_B_round, B_round)
        prev_B_round = set(B_round)

        speedup_kernel = t_batch_refit / t_incremental if t_incremental > 0 else float("inf")
        speedup = t_batch_refit / t_incremental_total if t_incremental_total > 0 else float("inf")

        row = {
            "dataset": ds["name"], "round": r, "frac_of_total": len(cumulative_idx) / n_total,
            "n_cumulative": len(cumulative_idx), "n_this_batch": len(subset),
            "B_init_size": len(B_init), "B_round_size": len(B_round),
            "jaccard_Binit_vs_Bround": jac_vs_init, "jaccard_Bround_vs_prevround": jac_vs_prev_round,
            "t_incremental_s": t_incremental, "t_incremental_total_s": t_incremental_total,
            "t_state_init_s": t_state_init, "t_batch_refit_s": t_batch_refit,
            "speedup": speedup, "speedup_kernel": speedup_kernel,
            "gamma_incremental": gamma_incremental, "gamma_ground_truth": gamma_gt, "gt_diff": gt_diff,
            "acc3_Binit": acc3_Binit, "accsvm_Binit": accsvm_Binit,
            "acc3_Bround": acc3_Bround, "accsvm_Bround": accsvm_Bround,
            "gap3_pp": (acc3_Bround - acc3_Binit) * 100, "gapsvm_pp": (accsvm_Bround - accsvm_Binit) * 100,
        }
        rows.append(row)
        print(f"  vong {r} ({len(cumulative_idx)}/{n_total}={row['frac_of_total']*100:.0f}%): "
              f"B_round={B_round} (|.|={len(B_round)}), Jaccard(init,round)={jac_vs_init:.3f}, "
              f"gt_diff={gt_diff:.2e}, t_incr={t_incremental*1000:.2f}ms, t_batch={t_batch_refit*1000:.1f}ms, "
              f"speedup={speedup:.1f}x, gap3={row['gap3_pp']:+.2f}pp, gapsvm={row['gapsvm_pp']:+.2f}pp")

    return rows


def main():
    all_rows = []
    for ds in PILOT_DATASETS:
        all_rows.extend(run_dataset(ds))
    out_df = pd.DataFrame(all_rows)
    out_path = os.path.join(HERE, "e0_measurement9_li2026_protocol.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nDa luu: {out_path}")
    max_gt = out_df.gt_diff.max()
    print(f"Sai so ground-truth lon nhat: {max_gt:.2e} "
          f"({'PASS' if max_gt < 1e-9 else 'CANH BAO'})")


if __name__ == "__main__":
    main()
