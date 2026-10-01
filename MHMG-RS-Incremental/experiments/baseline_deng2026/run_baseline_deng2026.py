"""
Run the Deng2026 baseline (0-overlap fuzzy rough set discernibility reducer) on the
same 80/20 split and with the same cv_accuracy (5-fold, 3-NN + SVM-RBF) as the other
two baselines, for a paired comparison. Main setting t=2 (a setting in which the
attribute set is actually reduced); t=1 and t=3 are run as a sweep.

The streaming-time columns match those of the other methods.
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reducer_deng_vec import fit_reduction_vec as fit_reduction  # vectorized, khop frozenset

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "..", "datasets_v3")
OUT_DIR = os.path.join(HERE, "..")
ALL = [("D1","Wine","1.csv"),("D2","WDBC","2.csv"),("D3","Iono","3.csv"),
       ("D4","Derm","4.csv"),("D5","Sonar","5.csv"),("D6","Urban","6.csv"),
       ("D7","Musk","7.csv"),("D8","Arrhy","8.csv"),("D9","Spambase","9.csv"),
       ("D10","Parkinsons","10.csv"),("D11","Isolet","11.csv"),("D12","Letter","12.csv")]
SEED, INIT_FRAC = 42, 0.8


def load(path):
    df = pd.read_csv(path)
    return df.iloc[:, :-1].select_dtypes(include=[np.number]).values, df.iloc[:, -1].values


def cv_accuracy(Xr, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    a3, asv = [], []
    for tr, te in skf.split(Xr, y):
        a3.append(KNeighborsClassifier(n_neighbors=3).fit(Xr[tr], y[tr]).score(Xr[te], y[te]))
        asv.append(SVC(kernel="rbf").fit(Xr[tr], y[tr]).score(Xr[te], y[te]))
    return float(np.mean(a3)), float(np.mean(asv))


def run_one(code, name, fn, t=2.0, p=1.0):
    print(f"\n{'='*54}\n{code} {name} (Deng2026 0-OF, t={t})\n{'='*54}", flush=True)
    X, y = load(os.path.join(DATASET_DIR, fn))
    n = len(y); X = MinMaxScaler().fit_transform(X)
    rng = np.random.default_rng(SEED); perm = rng.permutation(n); ni = int(n*INIT_FRAC)
    init_idx, stream_idx = perm[:ni], perm[ni:]
    full_order = np.concatenate([init_idx, stream_idx])

    t0 = time.time(); red_init, ndis_i = fit_reduction(X[init_idx], y[init_idx], t, p); ti = time.time()-t0
    print(f"  fit(80%): |R_init|={len(red_init)} |DIS|={ndis_i} {ti:.2f}s", flush=True)
    if len(red_init) == 0:
        return {"dataset": code, "dataset_name": name, "error": "empty reduct"}
    t0 = time.time(); red_full, ndis_f = fit_reduction(X, y, t, p); tf = time.time()-t0
    print(f"  fit(100%): |R_full|={len(red_full)} |DIS|={ndis_f} {tf:.2f}s", flush=True)

    final_X = X[full_order][:, red_init]; final_y = y[full_order]
    a3i, asvi = cv_accuracy(final_X, final_y, SEED)
    a3f, asvf = cv_accuracy(X[full_order][:, red_full], final_y, SEED)
    jac = len(set(red_init) & set(red_full)) / len(set(red_init) | set(red_full))
    print(f"  gap3={100*(a3f-a3i):.2f} gapsvm={100*(asvf-asvi):.2f} Jac={jac:.3f} "
          f"acc3_full={100*a3f:.2f} accsvm_full={100*asvf:.2f}", flush=True)
    return {"dataset": code, "dataset_name": name, "t": t,
            "reduct_init_size": len(red_init), "reduct_full_size": len(red_full),
            "time_batch_init_sec": ti, "time_batch_full_sec": tf,
            "gap_3nn": 100*(a3f-a3i), "gap_svm": 100*(asvf-asvi),
            "acc_3nn_init": a3i, "acc_3nn_full": a3f, "acc_svm_init": asvi, "acc_svm_full": asvf,
            "reduct_init_features": "|".join(map(str, red_init)),
            "reduct_full_features": "|".join(map(str, red_full)),
            "jaccard_init_full": jac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--out", default="baseline_deng2026_results.csv")
    args = ap.parse_args()
    ds = [d for d in ALL if (not args.only or d[0] in args.only) and d[0] not in args.exclude]
    rows = [run_one(*d, t=args.t) for d in ds]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, args.out), index=False)
    print(f"\nSaved: {os.path.join(OUT_DIR, args.out)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
