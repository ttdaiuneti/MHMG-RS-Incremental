"""
Accuracy gap over time. Besides the fixed B_init, the batch reducer is re-fitted at
each checkpoint on the data seen so far (B_current), and acc(B_init) is compared
with acc(B_current) at each checkpoint (5 checkpoints, as N_ACC_CHECKPOINTS of
e1_pipeline.py).

Scope: 4 of the 12 datasets (Wine, Musk, Urban, Spambase), covering stable (Wine,
Spambase) and unstable (Musk, Urban) cases; five re-fits are too costly on Letter
and Isolet.
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
PUBLISHED_CODE_DIR = os.path.join(HERE, "..", "..", "IJAR-MHMG-Published", "IJAR-Code")
sys.path.insert(0, os.path.abspath(PUBLISHED_CODE_DIR))
from HMMG_Reducer import MHTG_Reducer  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_v3")
SEED = 42
INIT_FRAC = 0.8
N_CHECKPOINTS = 5

DATASETS = [
    {"id": "D1", "name": "Wine", "file": "1.csv"},
    {"id": "D7", "name": "Musk", "file": "7.csv"},
    {"id": "D6", "name": "Urban", "file": "6.csv"},
    {"id": "D9", "name": "Spambase", "file": "9.csv"},
]


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


def run_for_dataset(ds):
    name, ds_id = ds["name"], ds["id"]
    path = os.path.join(DATASET_DIR, ds["file"])
    print(f"\n{'='*60}\n{ds_id} {name}\n{'='*60}", flush=True)

    X, y = load_dataset(path)
    n_total = len(y)
    X = MinMaxScaler().fit_transform(X)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_init = int(n_total * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]

    reducer = MHTG_Reducer()
    t0 = time.time()
    reduct_init, _, gamma_init = reducer.fit_reduction(X[init_idx], y[init_idx])
    print(f"  B_init: |R|={len(reduct_init)} ({time.time()-t0:.2f}s)")

    checkpoints = np.linspace(0, len(stream_idx), N_CHECKPOINTS, dtype=int)
    checkpoints[-1] = len(stream_idx)  # dam bao moc cuoi = 100%
    rows = []

    for ci, ckpt in enumerate(checkpoints):
        active_idx = np.concatenate([init_idx, stream_idx[:ckpt]])
        frac = len(active_idx) / n_total
        X_active, y_active = X[active_idx], y[active_idx]

        # B_current: batch re-fit on the data seen up to this checkpoint
        t0 = time.time()
        reducer_c = MHTG_Reducer()
        reduct_current, _, gamma_current = reducer_c.fit_reduction(X_active, y_active)
        t_refit = time.time() - t0

        acc3_init, accsvm_init = cv_accuracy(X_active[:, reduct_init], y_active, SEED)
        acc3_cur, accsvm_cur = cv_accuracy(X_active[:, reduct_current], y_active, SEED)
        gap3 = (acc3_cur - acc3_init) * 100
        gapsvm = (accsvm_cur - accsvm_init) * 100

        rows.append({
            "dataset": ds_id, "dataset_name": name, "checkpoint_idx": ci,
            "frac_of_data_seen": frac, "n_active": len(active_idx),
            "reduct_init_size": len(reduct_init), "reduct_current_size": len(reduct_current),
            "gap_3nn": gap3, "gap_svm": gapsvm, "refit_time_sec": t_refit,
        })
        print(f"  moc {ci} (frac={frac:.2f}, n={len(active_idx)}): "
              f"|R_current|={len(reduct_current)} gap_3nn={gap3:.2f}% gap_svm={gapsvm:.2f}% "
              f"(refit {t_refit:.2f}s)")

    return rows


def main():
    all_rows = []
    for ds in DATASETS:
        all_rows.extend(run_for_dataset(ds))
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(HERE, "e0_measurement2_acc_gap_trajectory.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDa luu: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
