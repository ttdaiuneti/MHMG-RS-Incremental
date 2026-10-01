"""
Run the Li2026 baseline (FNDS positive-region reduction) on the same 80/20 split,
the same i.i.d. stream order and with the same cv_accuracy as run_baseline_comparison.py
(Xu2025b), for a paired comparison. Output columns match baseline_xu2025b_results.csv.

delta = 0.75, the example threshold of the paper (checked by test1_example2.py).
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reducer_fnds import fit_reduction
from incremental_fnds import run_incremental_stream

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "..", "datasets_v3")
OUT_DIR = os.path.join(HERE, "..")

ALL = [("D1","Wine","1.csv"),("D2","WDBC","2.csv"),("D3","Iono","3.csv"),
       ("D4","Derm","4.csv"),("D5","Sonar","5.csv"),("D6","Urban","6.csv"),
       ("D7","Musk","7.csv"),("D8","Arrhy","8.csv"),("D9","Spambase","9.csv"),
       ("D10","Parkinsons","10.csv"),("D11","Isolet","11.csv"),("D12","Letter","12.csv")]
SEED, INIT_FRAC, DELTA = 42, 0.8, 0.75


def load(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    return X, y


def cv_accuracy(Xr, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    a3, asv = [], []
    for tr, te in skf.split(Xr, y):
        knn = KNeighborsClassifier(n_neighbors=3).fit(Xr[tr], y[tr])
        a3.append(knn.score(Xr[te], y[te]))
        svm = SVC(kernel="rbf").fit(Xr[tr], y[tr])
        asv.append(svm.score(Xr[te], y[te]))
    return float(np.mean(a3)), float(np.mean(asv))


def run_one(code, name, fn, delta=DELTA):
    print(f"\n{'='*56}\n{code} {name} (Li2026 FNDS, delta={delta})\n{'='*56}", flush=True)
    X, y = load(os.path.join(DATASET_DIR, fn))
    n = len(y)
    X = MinMaxScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    ni = int(n * INIT_FRAC)
    init_idx, stream_idx = perm[:ni], perm[ni:]
    full_order = np.concatenate([init_idx, stream_idx])

    t0 = time.time(); red_init, pos_i = fit_reduction(X[init_idx], y[init_idx], delta); t_init = time.time()-t0
    print(f"  fit(80%): |R_init|={len(red_init)} Pos={pos_i} {t_init:.2f}s", flush=True)
    if len(red_init) == 0:
        return {"dataset": code, "dataset_name": name, "error": "empty reduct (delta too permissive)"}
    t0 = time.time(); red_full, pos_f = fit_reduction(X, y, delta); t_full = time.time()-t0
    print(f"  fit(100%): |R_full|={len(red_full)} Pos={pos_f} {t_full:.2f}s", flush=True)

    XB = X[:, red_init]
    stream_time, gt = run_incremental_stream(XB, y, init_idx, stream_idx, delta)
    max_mm = max(g["pos_mismatch"] for g in gt)
    print(f"  stream({len(stream_idx)}): {stream_time:.4f}s max_pos_mismatch={max_mm}", flush=True)

    # accuracies: init reduct on streamed order; full reduct on matched order
    final_X = X[full_order][:, red_init]; final_y = y[full_order]
    a3i, asvi = cv_accuracy(final_X, final_y, SEED)
    a3f, asvf = cv_accuracy(X[full_order][:, red_full], final_y, SEED)
    speedup = t_full/stream_time if stream_time > 0 else float("inf")
    jac = len(set(red_init) & set(red_full)) / len(set(red_init) | set(red_full))
    print(f"  speedup={speedup:.1f}x gap3={100*(a3f-a3i):.2f} gapsvm={100*(asvf-asvi):.2f} Jac={jac:.3f}", flush=True)

    return {"dataset": code, "dataset_name": name, "delta": delta,
            "reduct_init_size": len(red_init), "reduct_full_size": len(red_full),
            "time_batch_full_sec": t_full, "stream_time_sec": stream_time,
            "speedup": speedup, "gap_3nn": 100*(a3f-a3i), "gap_svm": 100*(asvf-asvi),
            "max_pos_mismatch": max_mm,
            "acc_3nn_init": a3i, "acc_3nn_full": a3f, "acc_svm_init": asvi, "acc_svm_full": asvf,
            "reduct_init_features": "|".join(map(str, red_init)),
            "reduct_full_features": "|".join(map(str, red_full)),
            "jaccard_init_full": jac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", default="baseline_li2026_results.csv")
    args = ap.parse_args()
    datasets = [d for d in ALL if (not args.only or d[0] in args.only) and d[0] not in args.exclude]
    rows = [run_one(*d) for d in datasets]
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, args.out)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
