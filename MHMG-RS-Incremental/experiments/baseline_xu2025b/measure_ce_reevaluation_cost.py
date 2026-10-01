"""
Measure the cost of the periodic O(n^2) re-evaluation of the composite entropy of
the Xu2025b baseline.

The reduct_init_features stored in baseline_xu2025b_results.csv are reused (the
reduct search is not repeated). For each dataset, with the same seed, split and
stream order as run_baseline_comparison.py, the neighborhood matrix NM is built at
n = n_total (the last checkpoint), and composite_entropy(NM, y) is timed with
timeit.repeat (7 runs, minimum reported).

Output: experiments/ce_reevaluation_cost.csv
"""
import os
import sys
import timeit

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from local_nrs_composite_entropy import neighborhood_matrix, composite_entropy, DEFAULT_DELTA  # noqa: E402

EXP_DIR = os.path.join(HERE, "..")
PUBLISHED_CODE_DIR = os.path.join(HERE, "..", "..", "..", "IJAR-MHMG-Published", "IJAR-Code")
DATASET_DIR = os.path.join(HERE, "..", "..", "datasets_v3")

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
    {"id": "D12", "name": "Letter", "file": "12.csv"},
]  # D11 Isolet: excluded, same as run_baseline_comparison.py (infeasible fit_reduction cost)

SEED = 42
INIT_FRAC = 0.8
N_REPEATS = 7


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    return X, y


def main():
    results_path = os.path.join(EXP_DIR, "baseline_xu2025b_results.csv")
    prior = pd.read_csv(results_path).set_index("dataset")

    rows = []
    for ds in ALL_DATASETS:
        ds_id, name = ds["id"], ds["name"]
        cols = [int(c) for c in str(prior.loc[ds_id, "reduct_init_features"]).split("|")]

        X, y = load_dataset(os.path.join(DATASET_DIR, ds["file"]))
        n_total = len(y)
        X = MinMaxScaler().fit_transform(X)

        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n_total)
        n_init = int(n_total * INIT_FRAC)
        full_order_idx = np.concatenate([perm[:n_init], perm[n_init:]])

        XB = X[full_order_idx][:, cols]
        y_full = y[full_order_idx]

        NM = neighborhood_matrix(XB, DEFAULT_DELTA)

        timer = timeit.Timer(lambda: composite_entropy(NM, y_full))
        times = timer.repeat(repeat=N_REPEATS, number=1)
        t_min, t_median = min(times), float(np.median(times))

        print(f"{ds_id} {name}: n={n_total} |R_init|={len(cols)}  "
              f"ce_time min={t_min:.6f}s median={t_median:.6f}s over {N_REPEATS} reps", flush=True)

        rows.append({
            "dataset": ds_id, "dataset_name": name, "n": n_total,
            "reduct_init_size": len(cols),
            "ce_reeval_time_min_sec": t_min,
            "ce_reeval_time_median_sec": t_median,
            "n_repeats": N_REPEATS,
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(EXP_DIR, "ce_reevaluation_cost.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
