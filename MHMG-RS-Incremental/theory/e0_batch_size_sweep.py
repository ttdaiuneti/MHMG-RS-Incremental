"""Batch update (Theorem 5) vs k sequential single-object updates, as a function of k.

Both sides start from the same state and end in the same persistent state:
radii, granule sizes, E, K_d and class counts, the distance cache D extended
with the k new rows/columns (preallocated to n_init + k), and the object arrays
extended by the k objects. Data: the seed-42 80/20 split of e1_pipeline.py,
projected on the B_init selected there (stageA_batch_summary.csv); the batch is
the first k objects of the stream. Each side is timed N_TIMING_REPEATS times
(median). The final states of the two sides are compared after every run.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from e0_batch_insertion_check import (
    DATASET_DIR, load_dataset, pairwise_distances_direct, full_delta_and_granule,
    batch_closed_form, kd_batch_update, sequential_single_step,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
INIT_FRAC = 0.8
N_TIMING_REPEATS = 15

SWEEP = [
    {"id": "D1", "name": "Wine", "file": "1.csv", "k_values": [3, 5, 10, 20, 35]},
    {"id": "D6", "name": "Urban", "file": "6.csv", "k_values": [5, 10, 20, 40, 80, 120]},
    {"id": "D9", "name": "Spambase", "file": "9.csv", "k_values": [5, 10, 20, 40, 80, 160, 320, 640]},
]


def initial_state(X0, y0):
    d0 = pairwise_distances_direct(X0)
    delta0, gsize0 = full_delta_and_granule(d0, y0)
    E0 = float(np.sum(np.log2(gsize0 + 1)))
    classes, counts = np.unique(y0, return_counts=True)
    cc0 = dict(zip(classes.tolist(), counts.tolist()))
    K_d0 = float(sum(m * np.log2(m + 1) for m in cc0.values()))
    return d0, delta0, gsize0, E0, K_d0, cc0


def run_batch(X0, y0, d0, delta0, gsize0, E0, K_d0, cc0, bX, by):
    n0, k = len(y0), len(by)
    cache = np.zeros((n0 + k, n0 + k))
    cache[:n0, :n0] = d0
    t0 = time.perf_counter()
    delta, gsize, E, y, D_ob, D_bb = batch_closed_form(X0, y0, cache[:n0, :n0], delta0, gsize0, E0, bX, by, return_blocks=True)
    K_d, cc = kd_batch_update(K_d0, cc0, by)
    cache[:n0, n0:] = D_ob
    cache[n0:, :n0] = D_ob.T
    cache[n0:, n0:] = D_bb
    X = np.vstack([X0, bX])
    t = time.perf_counter() - t0
    return t, (X, y, delta, gsize, E, K_d, cc, cache)


def run_sequential(X0, y0, d0, delta0, gsize0, E0, K_d0, cc0, bX, by):
    n0, k = len(y0), len(by)
    cache = np.zeros((n0 + k, n0 + k))
    cache[:n0, :n0] = d0
    aX, ay, dl, gs, E, K_d, cc = X0, y0, delta0, gsize0, E0, K_d0, dict(cc0)
    t0 = time.perf_counter()
    for j in range(k):
        aX, ay, dl, gs, E, K_d, cc = sequential_single_step(aX, ay, dl, gs, E, K_d, cc, bX[j], by[j], cache, n0 + j)
    t = time.perf_counter() - t0
    return t, (aX, ay, dl, gs, E, K_d, cc, cache)


def same_state(a, b):
    return (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
            and np.array_equal(a[3], b[3]) and abs(a[4] - b[4]) < 1e-9 and abs(a[5] - b[5]) < 1e-9
            and a[6] == b[6] and np.array_equal(a[7], b[7]))


def run_dataset(ds, b_init):
    X, y = load_dataset(os.path.join(DATASET_DIR, ds["file"]))
    X = MinMaxScaler().fit_transform(X)[:, b_init]
    perm = np.random.default_rng(SEED).permutation(len(y))
    n_init = int(len(y) * INIT_FRAC)
    X0, y0 = X[perm[:n_init]].copy(), y[perm[:n_init]].copy()
    pool = perm[n_init:]
    st0 = initial_state(X0, y0)
    rows = []
    for k in ds["k_values"]:
        if k > len(pool):
            continue
        bX, by = X[pool[:k]], y[pool[:k]]
        tb, ts, equal = [], [], True
        for _ in range(N_TIMING_REPEATS):
            t1, s1 = run_batch(X0, y0, *st0, bX, by)
            t2, s2 = run_sequential(X0, y0, *st0, bX, by)
            tb.append(t1); ts.append(t2)
            equal &= same_state(s1, s2)
        mb, ms = float(np.median(tb)), float(np.median(ts))
        print(f"  [{ds['name']}] k={k:4d}: batch={mb * 1000:8.3f}ms sequential={ms * 1000:8.3f}ms "
              f"speedup={ms / mb:.2f}x same_state={equal}", flush=True)
        rows.append({"dataset": ds["name"], "n_init": n_init, "p_B": len(b_init), "K": k,
                     "batch_ms": mb * 1000, "sequential_ms": ms * 1000, "speedup": ms / mb,
                     "same_final_state": equal})
    return rows


def main():
    summ = pd.read_csv(os.path.join(HERE, "..", "experiments", "stageA_batch_summary.csv")).set_index("dataset_name")
    rows = []
    for ds in SWEEP:
        b_init = [int(v) for v in summ.loc[ds["name"], "reduct_init_features"].split("|")]
        print(f"\n=== {ds['name']} (|B_init|={len(b_init)}) ===")
        rows.extend(run_dataset(ds, b_init))
    out = os.path.join(HERE, "e0_batch_size_sweep.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
