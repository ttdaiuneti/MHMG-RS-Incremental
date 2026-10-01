"""
Incremental vs batch MHMG-RS on insertion-only streams.

For each dataset:
1. Batch fit_reduction on 80% (B_init) and on 100% (B_full) -- batch reference.
2. Incremental stream on the fixed B_init in two insertion orders:
   - i.i.d. (random permutation);
   - class-sorted (sorted by label; a stress test for new classes).
   Separate timers: t_delta_friend (radius + friend gain), t_shrink_cascade,
   t_kd_energy (K_d/E bookkeeping, O(1)), t_array_growth (vstack/append).
3. Frequency of zero radii (new and existing objects) in both streams.
4. Periodic check of the maintained state against a batch recomputation
   (every 25 steps).
5. Accuracy checkpoints (5 points along the stream, B_init). At the last checkpoint
   (100%), B_init is compared with B_full on the same row order as the i.i.d.
   stream (paired comparison).

Output: raw_results.csv (one row per insertion step) + accuracy_checkpoints.csv +
batch_summary.csv (incl. reduct_*_features, jaccard_overlap) +
gamma_ground_truth_check.csv, all appended per dataset (resumable).

Uses MHTG_Reducer of IJAR-MHMG-Published and the datasets in datasets_v3.
"""
import sys
import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

PUBLISHED_CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "IJAR-MHMG-Published", "IJAR-Code"
)
sys.path.insert(0, os.path.abspath(PUBLISHED_CODE_DIR))
from HMMG_Reducer import MHTG_Reducer  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_v3")
OUT_DIR = os.path.dirname(__file__)

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
N_ACC_CHECKPOINTS = 5
GROUND_TRUTH_CHECK_EVERY = 25


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    return X, y


def pairwise_distances_direct(X, chunk=200):
    """Full distance matrix with exactly the formula of the streaming update
    (sqrt(sum(diff**2)), as new_dists = sqrt(sum((active_X - x_new)**2))), not the
    Gram identity (X.X^T via BLAS gemm) of HMMG_Reducer._calculate_distances, whose
    results can differ from the streaming formula in the last bit."""
    n = X.shape[0]
    D = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = X[start:end, None, :] - X[None, :, :]
        D[start:end, :] = np.sqrt(np.sum(diff ** 2, axis=2))
    return D


def full_delta_and_granule(dist_matrix, y):
    n = len(y)
    deltas = np.full(n, np.inf)
    for i in range(n):
        enemy_mask = y != y[i]
        ed = dist_matrix[i][enemy_mask]
        if len(ed) > 0:
            deltas[i] = ed.min()
    granule_sizes = np.array([np.sum(dist_matrix[i] < deltas[i]) for i in range(n)])
    return deltas, granule_sizes


def cv_accuracy(X_reduced, y, seed):
    """5-fold CV, 3-NN (k=3) + SVM-RBF (default parameters). The reducts (B_init/B_full)
    are selected before and outside the CV (not nested); both are evaluated in the
    same way, so the comparison of B_init with B_full is paired."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc3, accs = [], []
    for train_idx, test_idx in skf.split(X_reduced, y):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_reduced[train_idx], y[train_idx])
        acc3.append(knn.score(X_reduced[test_idx], y[test_idx]))
        svm = SVC(kernel="rbf")
        svm.fit(X_reduced[train_idx], y[train_idx])
        accs.append(svm.score(X_reduced[test_idx], y[test_idx]))
    return float(np.mean(acc3)), float(np.std(acc3)), float(np.mean(accs)), float(np.std(accs))


def run_incremental_stream(XB, y, init_idx, stream_order_idx, stream_label, ds_id, ds_name):
    """Run one stream (i.i.d. or class-sorted) on a fixed B; return the rows of
    raw_results.csv, the items needed for the accuracy checkpoints, and the summary
    of the state checks."""
    n_total = len(y)
    n_init = len(init_idx)
    dist_matrix = np.zeros((n_total, n_total), dtype=np.float64)
    reducer = MHTG_Reducer()  # class names only; _calculate_distances is not used

    t_init0 = time.perf_counter()
    active_X = XB[init_idx].copy()
    active_y = y[init_idx].copy()
    d0 = pairwise_distances_direct(active_X)
    dist_matrix[:n_init, :n_init] = d0
    delta, gsize = full_delta_and_granule(d0, active_y)
    E_val = float(np.sum(np.log2(gsize + 1)))
    unique_labels, counts = np.unique(active_y, return_counts=True)
    class_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
    K_d = float(sum(m * np.log2(m + 1) for m in class_counts.values()))
    t_state_init = time.perf_counter() - t_init0  # state initialization time (D, delta, g, E, K_d)

    n_current = n_init
    rows = []
    checkpoint_steps = set(
        np.linspace(0, len(stream_order_idx) - 1, N_ACC_CHECKPOINTS, dtype=int).tolist()
    )
    checkpoint_records = []
    gt_check_records = []

    for step, idx in enumerate(stream_order_idx):
        x_new = XB[idx]
        y_new = y[idx]

        t0 = time.perf_counter()
        new_dists = np.sqrt(np.sum((active_X - x_new) ** 2, axis=1))
        dist_matrix[n_current, :n_current] = new_dists
        dist_matrix[:n_current, n_current] = new_dists

        friend_mask = active_y == y_new
        enemy_mask = ~friend_mask
        gain_mask = friend_mask & (new_dists < delta)
        n_gain = int(gain_mask.sum())
        gsize[gain_mask] += 1
        E_val += np.sum(np.log2(gsize[gain_mask] + 1) - np.log2(gsize[gain_mask] - 1 + 1))
        t_delta_friend = time.perf_counter() - t0

        t0 = time.perf_counter()
        shrink_mask = enemy_mask & (new_dists < delta)
        shrink_idx_arr = np.where(shrink_mask)[0]
        n_shrink = len(shrink_idx_arr)
        evicted_total = 0
        n_zero_existing = 0
        for i in shrink_idx_arr:
            new_delta_i = new_dists[i]
            delta[i] = new_delta_i
            if new_delta_i == 0:
                n_zero_existing += 1
            row_i = dist_matrix[i, :n_current]
            new_size = int(np.sum(row_i < new_delta_i))
            evicted_total += gsize[i] - new_size
            E_val += np.log2(new_size + 1) - np.log2(gsize[i] + 1)
            gsize[i] = new_size
        t_shrink_cascade = time.perf_counter() - t0

        t0 = time.perf_counter()
        enemy_dists = new_dists[enemy_mask]
        delta_new = float(enemy_dists.min()) if enemy_dists.size > 0 else np.inf
        friend_dists = new_dists[friend_mask]
        self_count = 1 if delta_new > 0 else 0
        n_zero_new = 1 if delta_new == 0 else 0
        gsize_new = int(np.sum(friend_dists < delta_new)) + self_count
        E_val += np.log2(gsize_new + 1)
        m_old = class_counts.get(y_new, 0)
        m_new = m_old + 1
        K_d += (m_new * np.log2(m_new + 1)) - (m_old * np.log2(m_old + 1) if m_old > 0 else 0.0)
        class_counts[y_new] = m_new
        t_kd_energy = time.perf_counter() - t0

        t0 = time.perf_counter()
        active_X = np.vstack([active_X, x_new[None, :]])
        active_y = np.append(active_y, y_new)
        delta = np.append(delta, delta_new)
        gsize = np.append(gsize, gsize_new)
        n_current += 1
        t_array_growth = time.perf_counter() - t0

        rows.append({
            "dataset": ds_id, "dataset_name": ds_name, "stream_type": stream_label,
            "step": step, "n_active_before": n_current - 1,
            "n_gain": n_gain, "n_shrink": n_shrink, "evicted_total": evicted_total,
            "n_zero_delta_new": n_zero_new, "n_zero_delta_existing": n_zero_existing,
            "t_delta_friend_sec": t_delta_friend, "t_shrink_cascade_sec": t_shrink_cascade,
            "t_kd_energy_sec": t_kd_energy, "t_array_growth_sec": t_array_growth,
            "gamma_running": E_val / K_d if K_d > 0 else None,
        })

        if step in checkpoint_steps:
            acc3m, acc3s, accsm, accss = cv_accuracy(active_X, active_y, SEED)
            checkpoint_records.append({
                "dataset": ds_id, "dataset_name": ds_name, "stream_type": stream_label,
                "step": step, "n_active": n_current, "frac_of_stream": (step + 1) / len(stream_order_idx),
                "reduct_used": "B_init", "acc_3nn_mean": acc3m, "acc_3nn_std": acc3s,
                "acc_svm_mean": accsm, "acc_svm_std": accss,
            })

        if step % GROUND_TRUTH_CHECK_EVERY == 0 or step == len(stream_order_idx) - 1:
            gt_delta, gt_gsize = full_delta_and_granule(dist_matrix[:n_current, :n_current], active_y)
            # independent check of each component, not only gamma:
            #  (1) radius and granule size from the cache vs the incrementally maintained values;
            #  (2) cached D vs distances recomputed from X on a random sample of rows.
            fin = np.isfinite(gt_delta) & np.isfinite(delta)
            delta_max_diff = float(np.max(np.abs(gt_delta[fin] - delta[fin]))) if fin.any() else 0.0
            delta_inf_mismatch = int(np.sum(np.isfinite(gt_delta) != np.isfinite(delta)))
            gsize_mismatch = int(np.sum(gt_gsize != gsize))
            rows_chk = np.random.default_rng(step).choice(n_current, size=min(64, n_current), replace=False)
            d_fresh = np.sqrt(np.sum((active_X[rows_chk][:, None, :] - active_X[None, :, :]) ** 2, axis=2))
            cache_max_diff = float(np.max(np.abs(d_fresh - dist_matrix[rows_chk, :n_current])))
            gt_E = float(np.sum(np.log2(gt_gsize + 1)))
            gt_labels, gt_counts = np.unique(active_y, return_counts=True)
            gt_K_d = float(np.sum(gt_counts * np.log2(gt_counts + 1)))
            gt_gamma = gt_E / gt_K_d if gt_K_d > 0 else None
            inc_gamma = E_val / K_d if K_d > 0 else None
            gt_check_records.append({
                "dataset": ds_id, "stream_type": stream_label, "step": step, "n": n_current,
                "gamma_incremental": inc_gamma, "gamma_groundtruth": gt_gamma,
                "abs_diff": abs(inc_gamma - gt_gamma) if (inc_gamma is not None and gt_gamma is not None) else None,
                "delta_max_abs_diff": delta_max_diff, "delta_inf_mismatch": delta_inf_mismatch,
                "gsize_mismatch": gsize_mismatch, "cache_vs_fresh_max_abs_diff": cache_max_diff,
            })

    return rows, checkpoint_records, gt_check_records, active_X, active_y, t_state_init


def run_for_dataset(ds, out_prefix):
    name, ds_id = ds["name"], ds["id"]
    path = os.path.join(DATASET_DIR, ds["file"])
    print(f"\n{'='*60}\n{ds_id} {name}\n{'='*60}", flush=True)

    X, y = load_dataset(path)
    n_total = len(y)
    X = MinMaxScaler().fit_transform(X)
    print(f"  |U|={n_total}, |C|={X.shape[1]}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_init = int(n_total * INIT_FRAC)
    init_idx = perm[:n_init]
    iid_stream_idx = perm[n_init:]
    full_order_idx = np.concatenate([init_idx, iid_stream_idx])  # same rows as active_X/active_y of the i.i.d. stream

    reducer = MHTG_Reducer()
    t0 = time.time()
    reduct_init, _, gamma_init = reducer.fit_reduction(X[init_idx], y[init_idx])
    time_batch_init = time.time() - t0
    print(f"  batch fit_reduction(80%): |R_init|={len(reduct_init)} gamma={gamma_init:.4f} {time_batch_init:.2f}s")

    t0 = time.time()
    reducer2 = MHTG_Reducer()
    reduct_full, _, gamma_full = reducer2.fit_reduction(X, y)
    time_batch_full = time.time() - t0
    print(f"  batch fit_reduction(100%): |R_full|={len(reduct_full)} gamma={gamma_full:.4f} {time_batch_full:.2f}s")

    set_init, set_full = set(reduct_init), set(reduct_full)
    jaccard = len(set_init & set_full) / len(set_init | set_full) if (set_init | set_full) else None

    batch_summary = [{
        "dataset": ds_id, "dataset_name": name, "n_total": n_total, "n_features": X.shape[1],
        "reduct_init_size": len(reduct_init), "gamma_init": gamma_init, "time_batch_init_sec": time_batch_init,
        "reduct_full_size": len(reduct_full), "gamma_full": gamma_full, "time_batch_full_sec": time_batch_full,
        "reduct_init_features": "|".join(map(str, sorted(reduct_init))),
        "reduct_full_features": "|".join(map(str, sorted(reduct_full))),
        "jaccard_overlap_init_full": jaccard,
    }]
    print(f"  Jaccard(B_init,B_full)={jaccard:.3f}")

    if len(reduct_init) == 0:
        print("  [SKIP] reduct init rong, khong the streaming")
        return

    B = reduct_init
    XB = X[:, B]

    # --- i.i.d. stream
    t0 = time.time()
    rows_iid, ckpt_iid, gt_iid, final_X_iid, final_y_iid, t_state_init_iid = run_incremental_stream(
        XB, y, init_idx, iid_stream_idx, "iid", ds_id, name
    )
    time_stream_iid = time.time() - t0
    print(f"  streaming i.i.d. ({len(iid_stream_idx)} buoc): {time_stream_iid:.2f}s")

    # accuracy at the last checkpoint: B_init (final_X_iid/final_y_iid) vs B_full,
    # with B_full evaluated on the same row order as final_X_iid/final_y_iid
    # (paired comparison: StratifiedKFold(shuffle=True) splits by row position,
    # so both sides get the same folds).
    X_full_matched = X[full_order_idx][:, reduct_full]
    y_full_matched = y[full_order_idx]
    assert np.array_equal(y_full_matched, final_y_iid), "label order mismatch: B_full is not evaluated on the stream row order"
    acc3m_f, acc3s_f, accsm_f, accss_f = cv_accuracy(X_full_matched, y_full_matched, SEED)
    ckpt_iid.append({
        "dataset": ds_id, "dataset_name": name, "stream_type": "iid",
        "step": len(iid_stream_idx) - 1, "n_active": n_total, "frac_of_stream": 1.0,
        "reduct_used": "B_full", "acc_3nn_mean": acc3m_f, "acc_3nn_std": acc3s_f,
        "acc_svm_mean": accsm_f, "acc_svm_std": accss_f,
    })

    # --- class-sorted stream (the remaining 20%, sorted by label instead of shuffled)
    class_sorted_stream_idx = iid_stream_idx[np.argsort(y[iid_stream_idx], kind="stable")]
    t0 = time.time()
    rows_cs, ckpt_cs, gt_cs, _, _, _ = run_incremental_stream(
        XB, y, init_idx, class_sorted_stream_idx, "class_sorted", ds_id, name
    )
    time_stream_cs = time.time() - t0
    print(f"  streaming class-sorted ({len(class_sorted_stream_idx)} buoc): {time_stream_cs:.2f}s")

    # the batch summary is written after the streams, to include the state
    # initialization time and the total update time of the i.i.d. stream
    batch_summary[0]["time_state_init_sec"] = t_state_init_iid
    batch_summary[0]["time_stream_update_iid_sec"] = float(sum(
        r["t_delta_friend_sec"] + r["t_shrink_cascade_sec"] + r["t_kd_energy_sec"] + r["t_array_growth_sec"]
        for r in rows_iid))
    bs_path = os.path.join(OUT_DIR, f"{out_prefix}batch_summary.csv")
    pd.DataFrame(batch_summary).to_csv(bs_path, mode="a", header=not os.path.exists(bs_path), index=False)

    all_rows = rows_iid + rows_cs
    all_ckpts = ckpt_iid + ckpt_cs
    all_gt = gt_iid + gt_cs

    raw_path = os.path.join(OUT_DIR, f"{out_prefix}raw_results.csv")
    pd.DataFrame(all_rows).to_csv(raw_path, mode="a", header=not os.path.exists(raw_path), index=False)
    ckpt_path = os.path.join(OUT_DIR, f"{out_prefix}accuracy_checkpoints.csv")
    pd.DataFrame(all_ckpts).to_csv(ckpt_path, mode="a", header=not os.path.exists(ckpt_path), index=False)
    gt_path = os.path.join(OUT_DIR, f"{out_prefix}gamma_ground_truth_check.csv")
    pd.DataFrame(all_gt).to_csv(gt_path, mode="a", header=not os.path.exists(gt_path), index=False)

    zero_new_iid = sum(r["n_zero_delta_new"] for r in rows_iid)
    zero_exist_iid = sum(r["n_zero_delta_existing"] for r in rows_iid)
    max_shrink_frac_iid = max((r["n_shrink"] / max(r["n_active_before"], 1) for r in rows_iid), default=0)
    max_gain_frac_iid = max((r["n_gain"] / max(r["n_active_before"], 1) for r in rows_iid), default=0)
    max_shrink_frac_cs = max((r["n_shrink"] / max(r["n_active_before"], 1) for r in rows_cs), default=0)
    max_gain_frac_cs = max((r["n_gain"] / max(r["n_active_before"], 1) for r in rows_cs), default=0)
    max_gt_diff = max((g["abs_diff"] for g in all_gt if g["abs_diff"] is not None), default=None)
    print(
        f"  [i.i.d.] zero_delta: new={zero_new_iid} existing={zero_exist_iid} "
        f"({(zero_new_iid+zero_exist_iid)/len(iid_stream_idx)*100:.2f}%)"
    )
    print(f"  [i.i.d.] max shrink_frac={max_shrink_frac_iid:.3f} max gain_frac={max_gain_frac_iid:.3f}")
    print(f"  [class-sorted] max shrink_frac={max_shrink_frac_cs:.3f} max gain_frac={max_gain_frac_cs:.3f}")
    print(f"  [ground-truth gamma check] max abs diff (ca 2 luong) = {max_gt_diff}")
    print(f"  Da luu: {raw_path}, {ckpt_path}, {gt_path}, batch_summary.csv (append)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="chi chay cac ID nay, vd D1 D9")
    parser.add_argument("--prefix", default="", help="tien to cho file output, vd 'stageA_'")
    args = parser.parse_args()

    datasets = ALL_DATASETS
    if args.only:
        datasets = [d for d in ALL_DATASETS if d["id"] in args.only]

    for ds in datasets:
        run_for_dataset(ds, args.prefix)

    print("\n=== HOAN TAT ===")


if __name__ == "__main__":
    main()
