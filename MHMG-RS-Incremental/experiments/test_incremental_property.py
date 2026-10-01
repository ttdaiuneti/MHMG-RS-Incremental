"""Randomized property test of the single-object incremental updates.

300 random streams with integer-valued attributes (many duplicates, ties and
zero radii), one to four classes, classes absent from the initial set, and
class-sorted orders. After EVERY insertion the maintained radii, granule sizes
and distance cache are compared with a batch evaluation, and at the end of each
stream gamma is recomputed independently from the raw data.
Run: python test_incremental_property.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_pipeline as mh
mh.cv_accuracy = lambda *a, **k: (0, 0, 0, 0)
mh.GROUND_TRUTH_CHECK_EVERY = 1          # check after EVERY insertion
def brute(X, y):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    n = len(y); d = np.full(n, np.inf)
    for i in range(n):
        e = D[i][y != y[i]]
        if e.size: d[i] = e.min()
    g = np.array([(D[i] < d[i]).sum() for i in range(n)])
    u, c = np.unique(y, return_counts=True)
    return d, g, np.log2(g + 1).sum() / (c * np.log2(c + 1)).sum()
rng = np.random.default_rng(0); worst = 0; cases = 0
for trial in range(300):
    n = rng.integers(8, 60); p = rng.integers(1, 4); q = rng.integers(1, 5)
    X = rng.integers(0, rng.integers(2, 5), size=(n, p)).astype(float) / 4   # heavy ties/duplicates
    y = rng.integers(0, q, size=n)
    if trial % 3 == 0:            # a class absent from the initial set appears in the stream
        y[: n // 2] = y[: n // 2] % max(q - 1, 1)
    n0 = max(2, n // 2); init = np.arange(n0); stream = np.arange(n0, n)
    if trial % 2: stream = stream[np.argsort(y[stream], kind="stable")]
    rows, _, gt, fX, fy, _ = mh.run_incremental_stream(X, y, init, stream, "t", "T", "T")
    for g in gt:
        assert g["delta_max_abs_diff"] == 0 and g["delta_inf_mismatch"] == 0 and g["gsize_mismatch"] == 0 and g["cache_vs_fresh_max_abs_diff"] == 0, (trial, g)
        worst = max(worst, g["abs_diff"]); cases += 1
    d, gg, gam = brute(fX, fy)                      # fully independent recomputation at the end
    worst = max(worst, abs(gam - rows[-1]["gamma_running"]))
print(f"{cases} per-insertion checks over 300 random streams: radii, granules, cache exact; max |gamma diff| = {worst:.1e}")
