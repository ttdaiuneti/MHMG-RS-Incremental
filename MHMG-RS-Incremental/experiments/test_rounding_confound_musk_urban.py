"""
Effect of the rounding of gamma (round(gamma,2) in the stopping rule of
HMMG_Reducer.fit_reduction, line 75) on Musk (D7) and Urban (D6), the two datasets
with the largest accuracy gaps, and on Wine.

Only the rounding precision of the greedy loop changes; the algorithm is a copy of
MHTG_Reducer.fit_reduction with the precision as a parameter.
"""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PUBLISHED_CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "IJAR-MHMG-Published", "IJAR-Code"
)
sys.path.insert(0, os.path.abspath(PUBLISHED_CODE_DIR))
from HMMG_Reducer import MHTG_Reducer  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_v3")
SEED = 42
INIT_FRAC = 0.8


def fit_reduction_with_precision(reducer, X, y, precision):
    """Copy of fit_reduction with a parameterized rounding precision (original = 2)."""
    n_samples, n_features = X.shape
    X = MinMaxScaler().fit_transform(X)
    reduct = []
    gamma_best = 0
    unique_labels, counts = np.unique(y, return_counts=True)
    class_size_map = dict(zip(unique_labels, counts))
    max_potential_energy = np.sum([np.log2(class_size_map[label] + 1) for label in y])
    remaining = list(range(n_features))

    while remaining:
        gamma_local_max = -1
        best_feature = None
        for f in remaining:
            subset = reduct + [f]
            gamma_current = reducer._calculate_gamma(X[:, subset], y, max_potential_energy)
            if precision is not None:
                gamma_current = round(gamma_current, precision)
            if gamma_current > gamma_local_max:
                gamma_local_max = gamma_current
                best_feature = f
        if gamma_local_max > gamma_best:
            reduct.append(best_feature)
            remaining.remove(best_feature)
            gamma_best = gamma_local_max
        else:
            break
    return reduct, gamma_best


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    return X, y


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not (sa | sb):
        return None
    return len(sa & sb) / len(sa | sb)


def run(ds_id, name, filename, rows):
    print(f"\n{'='*60}\n{ds_id} {name}\n{'='*60}")
    path = os.path.join(DATASET_DIR, filename)
    X, y = load_dataset(path)
    n_total = len(y)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_init = int(n_total * INIT_FRAC)
    init_idx = perm[:n_init]

    reducer = MHTG_Reducer()
    for precision in [2, 3, 4, 5, None]:
        r_init, g_init = fit_reduction_with_precision(reducer, X[init_idx], y[init_idx], precision)
        r_full, g_full = fit_reduction_with_precision(reducer, X, y, precision)
        jac = jaccard(r_init, r_full)
        label = f"round={precision}" if precision is not None else "khong lam tron"
        print(f"  {label:18s}: |R_init|={len(r_init):3d} |R_full|={len(r_full):3d} "
              f"Jaccard={jac:.3f}  R_init={sorted(r_init)}  R_full={sorted(r_full)}")
        rows.append({
            "dataset": ds_id, "dataset_name": name,
            "precision": precision if precision is not None else 15,  # "no rounding" ~ full float precision
            "jaccard_init_full": jac,
            "reduct_init_size": len(r_init), "reduct_full_size": len(r_full),
        })


if __name__ == "__main__":
    rows = []
    run("D7", "Musk", "7.csv", rows)
    run("D6", "Urban", "6.csv", rows)
    # Wine: measured at every precision (Jaccard 1.000 at p3, p4 and p5).
    run("D1", "Wine", "1.csv", rows)

    out_path = os.path.join(os.path.dirname(__file__), "rounding_confound_jaccard.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
