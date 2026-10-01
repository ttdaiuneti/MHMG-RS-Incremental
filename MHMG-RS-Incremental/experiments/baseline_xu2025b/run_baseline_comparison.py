"""
Run the Xu2025b baseline (PMLCE: fixed-delta local NRS + composite entropy) on the
same 80/20 split and the same i.i.d. stream order as e1_pipeline.py (MHMG-RS), for a
paired comparison with identical seeds and order.

Incremental part: the neighborhood matrix is cached and only the row/column of the
new object is computed (with a fixed delta, neighbor sets only grow). The lower/upper
approximations and CE are recomputed from the cached matrix (vectorized).

Periodic check (every 25 steps): the incremental CE is compared with a recomputation.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from local_nrs_composite_entropy import (  # noqa: E402
    neighborhood_matrix, composite_entropy, fit_reduction, DEFAULT_DELTA,
)

PUBLISHED_CODE_DIR = os.path.join(HERE, "..", "..", "..", "IJAR-MHMG-Published", "IJAR-Code")
DATASET_DIR = os.path.join(HERE, "..", "..", "datasets_v3")
OUT_DIR = os.path.join(HERE, "..")

# same list as ALL_DATASETS in e1_pipeline.py; Isolet (D11) is skipped by the --exclude default
ALL_DATASETS = [
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
    {"id": "D11", "name": "Isolet", "file": "11.csv"},
    {"id": "D12", "name": "Letter", "file": "12.csv"},
]

SEED = 42
INIT_FRAC = 0.8
GROUND_TRUTH_CHECK_EVERY = 25


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    return X, y


def cv_accuracy(X_reduced, y, seed):
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


def run_incremental_stream(XB, y, init_idx, stream_idx, delta):
    n_total = len(y)
    n_init = len(init_idx)
    NM_cache = np.zeros((n_total, n_total), dtype=bool)

    t_init0 = time.perf_counter()
    active_X = XB[init_idx].copy()
    active_y = y[init_idx].copy()
    NM_cache[:n_init, :n_init] = neighborhood_matrix(active_X, delta)
    t_state_init = time.perf_counter() - t_init0  # state initialization time

    n_current = n_init
    per_step_time = []
    gt_checks = []

    for step, idx in enumerate(stream_idx):
        x_new = XB[idx]
        y_new = y[idx]

        t0 = time.perf_counter()
        # distances from the new object to the existing ones only (O(n)), no full recomputation
        dist_new = np.sqrt(np.sum((active_X - x_new) ** 2, axis=1))
        neighbor_new = dist_new <= delta
        NM_cache[n_current, :n_current] = neighbor_new
        NM_cache[:n_current, n_current] = neighbor_new
        NM_cache[n_current, n_current] = True  # every object is its own neighbor

        active_X = np.vstack([active_X, x_new[None, :]])
        active_y = np.append(active_y, y_new)
        n_current += 1
        per_step_time.append(time.perf_counter() - t0)

        if step % GROUND_TRUTH_CHECK_EVERY == 0 or step == len(stream_idx) - 1:
            ce_incremental = composite_entropy(NM_cache[:n_current, :n_current], active_y)
            gt_NM = neighborhood_matrix(active_X, delta)
            ce_groundtruth = composite_entropy(gt_NM, active_y)
            diff = (
                abs(ce_incremental - ce_groundtruth)
                if np.isfinite(ce_incremental) and np.isfinite(ce_groundtruth)
                else (0.0 if ce_incremental == ce_groundtruth else float("inf"))
            )
            gt_checks.append({"step": step, "n": n_current, "abs_diff": diff})

    return sum(per_step_time), active_X, active_y, gt_checks, t_state_init


def run_for_dataset(ds, delta=DEFAULT_DELTA):
    name, ds_id = ds["name"], ds["id"]
    path = os.path.join(DATASET_DIR, ds["file"])
    print(f"\n{'='*60}\n{ds_id} {name} (Xu2025b baseline, delta={delta})\n{'='*60}", flush=True)

    X, y = load_dataset(path)
    n_total = len(y)
    X = MinMaxScaler().fit_transform(X)

    # same seed/permutation as e1_pipeline.py, for a paired comparison
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_init = int(n_total * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]
    full_order_idx = np.concatenate([init_idx, stream_idx])

    t0 = time.time()
    reduct_init, _, ce_init = fit_reduction(X[init_idx], y[init_idx], delta)
    time_batch_init = time.time() - t0
    print(f"  fit_reduction(80%): |R_init|={len(reduct_init)} CE={ce_init:.4f} {time_batch_init:.2f}s")

    t0 = time.time()
    reduct_full, _, ce_full = fit_reduction(X, y, delta)
    time_batch_full = time.time() - t0
    print(f"  fit_reduction(100%): |R_full|={len(reduct_full)} CE={ce_full:.4f} {time_batch_full:.2f}s")

    if len(reduct_init) == 0:
        print("  [SKIP] reduct init rong")
        return {"dataset": ds_id, "dataset_name": name, "error": "empty reduct"}

    XB = X[:, reduct_init]
    t0 = time.time()
    stream_time, final_X, final_y, gt_checks, t_state_init = run_incremental_stream(XB, y, init_idx, stream_idx, delta)
    wall_stream = time.time() - t0
    max_gt_diff = max((g["abs_diff"] for g in gt_checks), default=None)
    print(f"  streaming ({len(stream_idx)} buoc): {stream_time:.4f}s (wall {wall_stream:.2f}s), "
          f"max ground-truth diff={max_gt_diff}")

    acc3_init, accsvm_init = cv_accuracy(final_X, final_y, SEED)
    X_full_matched = X[full_order_idx][:, reduct_full]
    y_full_matched = y[full_order_idx]
    assert np.array_equal(y_full_matched, final_y), "thu tu nhan khong khop"
    acc3_full, accsvm_full = cv_accuracy(X_full_matched, y_full_matched, SEED)

    speedup = time_batch_full / stream_time if stream_time > 0 else float("inf")
    gap3 = (acc3_full - acc3_init) * 100
    gapsvm = (accsvm_full - accsvm_init) * 100
    jaccard_init_full = len(set(reduct_init) & set(reduct_full)) / len(set(reduct_init) | set(reduct_full))
    print(f"  speedup={speedup:.1f}x  gap_3nn={gap3:.2f}%  gap_svm={gapsvm:.2f}%  "
          f"Jaccard(init,full)={jaccard_init_full:.3f}")

    return {
        "dataset": ds_id, "dataset_name": name, "delta": delta,
        "reduct_init_size": len(reduct_init), "reduct_full_size": len(reduct_full),
        "time_batch_init_sec": time_batch_init, "time_state_init_sec": t_state_init,
        "time_batch_full_sec": time_batch_full, "stream_time_sec": stream_time,
        "speedup": speedup, "gap_3nn": gap3, "gap_svm": gapsvm,
        "max_ground_truth_diff": max_gt_diff,
        # absolute accuracies (not only the gap), for a direct comparison with MHMG-RS
        # rather than of each method's own drift
        "acc_3nn_init": acc3_init, "acc_3nn_full": acc3_full,
        "acc_svm_init": accsvm_init, "acc_svm_full": accsvm_full,
        # the selected attributes (not only their number), needed for the Jaccard
        # overlap between the initial and final reducts of Xu2025b,
        # as reported for MHMG-RS
        "reduct_init_features": "|".join(map(str, sorted(reduct_init))),
        "reduct_full_features": "|".join(map(str, sorted(reduct_full))),
        "jaccard_init_full": jaccard_init_full,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=["D11"],
                         help="D11 (Isolet) is excluded by default: phase IM of Algorithm 2 "
                              "costs O(|A|^2 * n^2), about 2.3e13 operations on "
                              "Isolet (617 attributes), a property of the original "
                              "Xu2025b algorithm, not of this implementation.")
    args = parser.parse_args()

    datasets = ALL_DATASETS
    if args.only:
        datasets = [d for d in ALL_DATASETS if d["id"] in args.only]
    else:
        datasets = [d for d in ALL_DATASETS if d["id"] not in args.exclude]

    results = [run_for_dataset(ds) for ds in datasets]
    df = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "baseline_xu2025b_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDa luu: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
