"""Pilot: can the maintained gamma(B_init) tell when the frozen reduct has gone stale?

For every split used in the staleness analysis (seed 42 of e1_pipeline.py and the
seeds of multi_seed_w*.csv), with B_init and B_full taken from those runs, compute
on the scaled data:
  g_init      gamma(B_init) on the initial 80%   (value at selection time)
  g_end       gamma(B_init) on all objects       (value the incremental update maintains)
  g_full      gamma(B_full) on all objects
  add_best    max over a not in B_init of gamma(B_init + {a}) on all objects
  add_best_init  the same on the initial 80%
The candidate trigger signals (drop = g_init - g_end, add gain = add_best - g_end)
are then related to the end-of-stream accuracy gaps of the same runs.
Distances: sqrt of the per-attribute sum of squared differences (direct form).

Output: pilot_trigger_signal.csv, one row per split and dataset (appended; resumable).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from e1_pipeline import ALL_DATASETS, DATASET_DIR, INIT_FRAC, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pilot_trigger_signal.csv")
CHUNK = 256


def gammas(X, y, B, extra):
    """gamma(B) and gamma(B + {a}) for every a in extra, on objects (X, y)."""
    n = len(y)
    _, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    kd = float(np.sum(np.log2(cnt[inv] + 1)))
    E = np.zeros(1 + len(extra))
    XB = X[:, B]
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        base = np.zeros((e - s, n))
        for j in range(XB.shape[1]):
            base += (XB[s:e, j, None] - XB[None, :, j]) ** 2
        same = y[s:e, None] == y[None, :]
        for k, a in enumerate([None] + list(extra)):
            sq = base if a is None else base + (X[s:e, a, None] - X[None, :, a]) ** 2
            D = np.sqrt(sq)
            delta = np.where(same, np.inf, D).min(axis=1)
            E[k] += np.sum(np.log2((D < delta[:, None]).sum(axis=1) + 1))
    return E / kd


def split_rows():
    summ = pd.read_csv(os.path.join(HERE, "stageA_batch_summary.csv")).set_index("dataset_name")
    rows = []
    for d in ALL_DATASETS:
        if d["name"] in summ.index:
            s = summ.loc[d["name"]]
            rows.append((42, d["name"], d["file"], s.reduct_init_features, s.reduct_full_features))
    for w in (1, 2, 3):
        ms = pd.read_csv(os.path.join(HERE, f"multi_seed_w{w}.csv"))
        files = {d["name"]: d["file"] for d in ALL_DATASETS}
        for _, r in ms.iterrows():
            rows.append((int(r.seed), r.dataset_name, files[r.dataset_name], r.reduct_init, r.reduct_full))
    return rows


def run(job):
    seed, name, file, b_init, b_full = job
    t0 = time.time()
    X, y = load_dataset(os.path.join(DATASET_DIR, file))
    X = MinMaxScaler().fit_transform(X)
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    init_idx = perm[:int(n * INIT_FRAC)]
    Bi = [int(v) for v in str(b_init).split("|")]
    Bf = [int(v) for v in str(b_full).split("|")]
    rest = [a for a in range(X.shape[1]) if a not in Bi]
    gi = gammas(X[init_idx], y[init_idx], Bi, rest)
    ge = gammas(X, y, Bi, rest)
    gf = gammas(X, y, Bf, [])[0]
    return {"seed": seed, "dataset_name": name, "n": n, "p": X.shape[1], "size_init": len(Bi),
            "g_init": gi[0], "g_end": ge[0], "g_full": gf,
            "add_best_init": gi[1:].max() if rest else np.nan,
            "add_best": ge[1:].max() if rest else np.nan,
            "same_reduct": sorted(Bi) == sorted(Bf), "sec": time.time() - t0}


def main():
    jobs = split_rows()
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        done = set(zip(d.seed, d.dataset_name))
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    jobs.sort(key=lambda j: j[1] == "Letter")  # long ones last
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"{len(jobs)} jobs, {workers} workers", flush=True)
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run, jobs):
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"seed {r['seed']} {r['dataset_name']}: g_init={r['g_init']:.4f} g_end={r['g_end']:.4f} "
                  f"add_best={r['add_best']:.4f} ({r['sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
