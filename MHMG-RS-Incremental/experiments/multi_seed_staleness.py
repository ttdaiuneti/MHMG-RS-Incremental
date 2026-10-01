"""Staleness of a frozen reduct over several random 80/20 splits.

For each seed and dataset: select B_init on the initial 80% and B_full on all
objects with the batch MHMG-RS reducer, then compare their accuracies by the same
5-fold CV used in e1_pipeline.py, on the same row order (initial objects, then
the i.i.d. stream). This reproduces the end-of-stream gap of e1_pipeline.py
without running the stream itself, which does not affect the gap.

Usage: python multi_seed_staleness.py --seeds 0 1 2 --only D1 D2
Output: --out file (default multi_seed_staleness.csv), appended, one row per seed and dataset
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from e1_pipeline import ALL_DATASETS, DATASET_DIR, INIT_FRAC, MHTG_Reducer, cv_accuracy, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def run(ds, seed):
    X, y = load_dataset(os.path.join(DATASET_DIR, ds["file"]))
    X = MinMaxScaler().fit_transform(X)
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = perm[:n_init]
    order = np.concatenate([init_idx, perm[n_init:]])
    t0 = time.time()
    b_init, _, _ = MHTG_Reducer().fit_reduction(X[init_idx], y[init_idx])
    b_full, _, _ = MHTG_Reducer().fit_reduction(X, y)
    t = time.time() - t0
    a3i, _, asi, _ = cv_accuracy(X[order][:, b_init], y[order], seed)
    a3f, _, asf, _ = cv_accuracy(X[order][:, b_full], y[order], seed)
    si, sf = set(b_init), set(b_full)
    return {"seed": seed, "dataset": ds["id"], "dataset_name": ds["name"], "n": n,
            "size_init": len(b_init), "size_full": len(b_full),
            "jaccard": len(si & sf) / len(si | sf),
            "acc_3nn_init": a3i, "acc_3nn_full": a3f, "acc_svm_init": asi, "acc_svm_full": asf,
            "gap_3nn_pp": 100 * (a3f - a3i), "gap_svm_pp": 100 * (asf - asi),
            "reduct_init": "|".join(map(str, sorted(b_init))),
            "reduct_full": "|".join(map(str, sorted(b_full))), "search_sec": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="multi_seed_staleness.csv")
    args = ap.parse_args()
    datasets = [d for d in ALL_DATASETS if not args.only or d["id"] in args.only]
    out = os.path.join(HERE, args.out)
    for seed in args.seeds:
        for ds in datasets:
            row = run(ds, seed)
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"seed {seed} {ds['name']}: gap3={row['gap_3nn_pp']:+.2f} gapsvm={row['gap_svm_pp']:+.2f} "
                  f"J={row['jaccard']:.2f} ({row['search_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
