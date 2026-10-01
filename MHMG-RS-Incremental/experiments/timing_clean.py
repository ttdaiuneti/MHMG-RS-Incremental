"""Re-measure the timings reported in the manuscript on an otherwise idle machine.

For MHMG-RS and Xu2025b, on the seed-42 split of e1_pipeline.py: one batch search
on 80% and one on 100% of the data, then the i.i.d. stream on the fixed B_init,
repeated REPEATS times (state initialization and update times, median reported).
Correctness checks and accuracy evaluation are switched off here; they are part
of the main run (e1_pipeline.py, run_baseline_comparison.py), whose selected
reducts this script re-derives and checks.

Usage: python timing_clean.py --method mhmg|xu [--only D1 ...]
Output: timing_clean_<method>.csv (appended, one row per dataset)

--stream-only (MHMG): re-measure only the state initialization and the stream, in a
process that runs no batch search, with B_init read from timing_clean_mhmg.csv.
Measured after a long search in the same process, the Isolet stream took 1.8 times
as long as in a fresh process (the other datasets: within 7%), so the manuscript
uses these stream times for all datasets. Output: timing_stream_mhmg.csv (overwritten).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "baseline_xu2025b"))
import e1_pipeline as mh  # noqa: E402
import run_baseline_comparison as xu  # noqa: E402

REPEATS = 5
STEP_COLS = ["t_delta_friend_sec", "t_shrink_cascade_sec", "t_kd_energy_sec", "t_array_growth_sec"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def split(n):
    perm = np.random.default_rng(42).permutation(n)
    n_init = int(n * 0.8)
    return perm[:n_init], perm[n_init:]


def time_mhmg(ds, reuse_init=False):
    mh.GROUND_TRUTH_CHECK_EVERY = 10 ** 9          # only the final-step check remains
    mh.N_ACC_CHECKPOINTS = 1
    mh.cv_accuracy = lambda *a, **k: (0.0, 0.0, 0.0, 0.0)
    X, y = mh.load_dataset(os.path.join(mh.DATASET_DIR, ds["file"]))
    X = MinMaxScaler().fit_transform(X)
    init_idx, stream_idx = split(len(y))
    if reuse_init:  # B_init taken from the main run; its search time is not re-measured
        summ = pd.concat([pd.read_csv(os.path.join(HERE, f"stage{s}_batch_summary.csv")) for s in "AB"])
        b_init = [int(v) for v in summ.set_index("dataset_name").loc[ds["name"], "reduct_init_features"].split("|")]
        t_init = float("nan")
    else:
        t0 = time.time(); b_init, _, _ = mh.MHTG_Reducer().fit_reduction(X[init_idx], y[init_idx]); t_init = time.time() - t0
    t0 = time.time(); b_full, _, _ = mh.MHTG_Reducer().fit_reduction(X, y); t_full = time.time() - t0
    streams, inits = [], []
    for _ in range(REPEATS):
        rows, _, _, _, _, t_state = mh.run_incremental_stream(X[:, b_init], y, init_idx, stream_idx, "iid", ds["id"], ds["name"])
        streams.append(sum(r[c] for r in rows for c in STEP_COLS)); inits.append(t_state)
    return b_init, b_full, t_init, t_full, float(np.median(inits)), float(np.median(streams)), float(np.min(streams)), float(np.max(streams))


def time_xu(ds):
    xu.GROUND_TRUTH_CHECK_EVERY = 10 ** 9
    X, y = xu.load_dataset(os.path.join(xu.DATASET_DIR, ds["file"]))
    X = MinMaxScaler().fit_transform(X)
    init_idx, stream_idx = split(len(y))
    t0 = time.time(); b_init, _, _ = xu.fit_reduction(X[init_idx], y[init_idx], xu.DEFAULT_DELTA); t_init = time.time() - t0
    log(f"xu {ds['name']}: batch80 {t_init:.1f}s")
    t0 = time.time(); b_full, _, _ = xu.fit_reduction(X, y, xu.DEFAULT_DELTA); t_full = time.time() - t0
    log(f"xu {ds['name']}: batch100 {t_full:.1f}s")
    streams, inits = [], []
    for r in range(REPEATS):
        st, _, _, _, t_state = xu.run_incremental_stream(X[:, b_init], y, init_idx, stream_idx, xu.DEFAULT_DELTA)
        streams.append(st); inits.append(t_state)
        log(f"xu {ds['name']}: stream repeat {r + 1}/{REPEATS}")
    return b_init, b_full, t_init, t_full, float(np.median(inits)), float(np.median(streams)), float(np.min(streams)), float(np.max(streams))


def stream_only_mhmg(only):
    mh.GROUND_TRUTH_CHECK_EVERY = 10 ** 9
    mh.N_ACC_CHECKPOINTS = 1
    mh.cv_accuracy = lambda *a, **k: (0.0, 0.0, 0.0, 0.0)
    tc = pd.read_csv(os.path.join(HERE, "timing_clean_mhmg.csv")).set_index("dataset_name")
    out = []
    for ds in mh.ALL_DATASETS:
        if only and ds["id"] not in only:
            continue
        X, y = mh.load_dataset(os.path.join(mh.DATASET_DIR, ds["file"]))
        X = MinMaxScaler().fit_transform(X)
        init_idx, stream_idx = split(len(y))
        b_init = [int(v) for v in tc.loc[ds["name"], "reduct_init_features"].split("|")]
        streams, inits = [], []
        for _ in range(REPEATS):
            rows, _, _, _, _, t_state = mh.run_incremental_stream(X[:, b_init], y, init_idx, stream_idx, "iid", ds["id"], ds["name"])
            streams.append(sum(r[c] for r in rows for c in STEP_COLS)); inits.append(t_state)
        out.append({"dataset": ds["id"], "dataset_name": ds["name"], "repeats": REPEATS,
                    "reduct_init_features": tc.loc[ds["name"], "reduct_init_features"],
                    "time_state_init_sec": float(np.median(inits)), "time_stream_sec_median": float(np.median(streams)),
                    "time_stream_sec_min": float(np.min(streams)), "time_stream_sec_max": float(np.max(streams))})
        log(f"mhmg {ds['name']}: init={out[-1]['time_state_init_sec']:.4f}s stream={out[-1]['time_stream_sec_median']:.4f}s")
    pd.DataFrame(out).to_csv(os.path.join(HERE, "timing_stream_mhmg.csv"), index=False)


def eval_only_mhmg(only):
    """Batch evaluation: gamma(B_init) recomputed from the definitions on all n
    objects (distances, radii, granule sizes, E, K_d), as a reference for the
    per-insertion cost of keeping gamma current. Objects are in the order of the
    maintained state (initial objects, then the i.i.d. stream), so the result
    equals the maintained gamma at the end of the stream.
    Output: timing_eval_mhmg.csv."""
    tc = pd.read_csv(os.path.join(HERE, "timing_clean_mhmg.csv")).set_index("dataset_name")
    out = []
    for ds in mh.ALL_DATASETS:
        if only and ds["id"] not in only:
            continue
        X, y = mh.load_dataset(os.path.join(mh.DATASET_DIR, ds["file"]))
        X = MinMaxScaler().fit_transform(X)
        b_init = [int(v) for v in tc.loc[ds["name"], "reduct_init_features"].split("|")]
        init_idx, stream_idx = split(len(y))
        order = np.concatenate([init_idx, stream_idx])
        XB, y = X[order][:, b_init], y[order]
        times, gam = [], None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            d = mh.pairwise_distances_direct(XB)
            delta, gsize = mh.full_delta_and_granule(d, y)
            E = float(np.sum(np.log2(gsize + 1)))
            _, counts = np.unique(y, return_counts=True)
            K_d = float(np.sum(counts * np.log2(counts + 1)))
            gam = E / K_d
            times.append(time.perf_counter() - t0)
            del d
        out.append({"dataset": ds["id"], "dataset_name": ds["name"], "repeats": REPEATS, "n": len(y),
                    "reduct_init_features": tc.loc[ds["name"], "reduct_init_features"],
                    "gamma_B_init_all": gam, "time_eval_sec_median": float(np.median(times)),
                    "time_eval_sec_min": float(np.min(times)), "time_eval_sec_max": float(np.max(times))})
        log(f"mhmg {ds['name']}: batch evaluation on n={len(y)}: {out[-1]['time_eval_sec_median']:.4f}s")
    pd.DataFrame(out).to_csv(os.path.join(HERE, "timing_eval_mhmg.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["mhmg", "xu"], required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reuse-init", action="store_true", help="MHMG only: take B_init from the main run")
    ap.add_argument("--stream-only", action="store_true", help="MHMG only: stream and state init only")
    ap.add_argument("--eval-only", action="store_true", help="MHMG only: batch evaluation of gamma(B_init) on all objects")
    args = ap.parse_args()
    if args.stream_only:
        return stream_only_mhmg(args.only)
    if args.eval_only:
        return eval_only_mhmg(args.only)
    table = mh.ALL_DATASETS if args.method == "mhmg" else xu.ALL_DATASETS
    out = os.path.join(HERE, f"timing_clean_{args.method}.csv")
    for ds in table:
        if args.only and ds["id"] not in args.only:
            continue
        if args.method == "mhmg":
            res = time_mhmg(ds, reuse_init=args.reuse_init)
        else:
            res = time_xu(ds)
        b_init, b_full, t_init, t_full, t_state, t_stream, s_min, s_max = res
        row = {"dataset": ds["id"], "dataset_name": ds["name"], "repeats": REPEATS,
               "reduct_init_features": "|".join(map(str, sorted(b_init))),
               "reduct_full_features": "|".join(map(str, sorted(b_full))),
               "time_batch_init_sec": t_init, "time_batch_full_sec": t_full,
               "time_state_init_sec": t_state, "time_stream_sec_median": t_stream,
               "time_stream_sec_min": s_min, "time_stream_sec_max": s_max}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"{args.method} {ds['name']}: batch80={t_init:.2f}s batch100={t_full:.2f}s "
              f"init={t_state:.4f}s stream={t_stream:.4f}s [{s_min:.4f},{s_max:.4f}]", flush=True)


if __name__ == "__main__":
    main()
