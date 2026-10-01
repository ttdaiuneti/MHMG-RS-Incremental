"""Compare the dependency computed by the published batch reducer, whose
distances use the Gram identity sqrt(|a|^2+|b|^2-2ab), with the dependency
computed from direct coordinate differences, on each dataset's B_full.
Letter and Isolet are omitted for memory. Output: reducer_distance_check.csv"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from e1_pipeline import (ALL_DATASETS, DATASET_DIR, MHTG_Reducer, full_delta_and_granule,
                         load_dataset, pairwise_distances_direct)

HERE = os.path.dirname(os.path.abspath(__file__))
summ = pd.concat([pd.read_csv(os.path.join(HERE, f"stage{s}_batch_summary.csv")) for s in "AB"]).set_index("dataset_name")
rows = []
for ds in ALL_DATASETS:
    if ds["id"] in ("D11", "D12"):
        continue
    X, y = load_dataset(os.path.join(DATASET_DIR, ds["file"]))
    X = MinMaxScaler().fit_transform(X)
    B = [int(v) for v in summ.loc[ds["name"], "reduct_full_features"].split("|")]
    r = MHTG_Reducer()
    _, c = np.unique(y, return_counts=True)
    K = float(np.sum(c * np.log2(c + 1)))
    g_gram = r._calculate_gamma(X[:, B], y, K)
    D = pairwise_distances_direct(X[:, B])
    _, g = full_delta_and_granule(D, y)
    g_direct = float(np.sum(np.log2(g + 1)) / K)
    rows.append({"dataset_name": ds["name"], "gamma_gram": g_gram, "gamma_direct": g_direct,
                 "abs_diff": abs(g_gram - g_direct),
                 "max_abs_distance_diff": float(np.abs(r._calculate_distances(X[:, B]) - D).max())})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "reducer_distance_check.csv"), index=False)
