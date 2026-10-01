"""
Deng2026 reducer, vectorized (numpy) version equivalent to reducer_deng.py that does
not materialize millions of frozensets, so it runs on large datasets (Letter).

Mathematically equivalent to reducer_deng.build_dis_clauses + forward_filter:
  a different-class pair (i,j) is in DIS(a)  <=>  R_a(i,j)^t + lambda(i)^t <= 1
     <=>  R_a(i,j) <= thr_i,  thr_i = (1 - lambda_i^t)^(1/t)
     <=>  |x_ia - x_ja| >= gap_i,  gap_i = 1 - thr_i.
  pair in DIS(A)  <=>  max_a |x_ia - x_ja| >= gap_i (and different classes).
  reduct = greedy hitting set by the number of uncovered pairs each attribute covers,
  ties broken by the smallest index -- identical to forward_filter of the frozenset version.

Greedy selection and pruning run over row blocks, which bounds memory by O(block*n)
instead of O(n^2).
"""
import numpy as np
from reducer_deng import lambda_vec


def _gaps(X, y, t, p):
    lam = lambda_vec(X, y, p)
    thr = np.power(1.0 - np.power(lam, t), 1.0 / t)  # thr_i
    return 1.0 - thr  # gap_i


def _in_disA(X, y, gap, block=1000):
    """Boolean n x n: (i,j) diff-class & max_a|x_ia-x_ja| >= gap_i."""
    n = X.shape[0]
    InD = np.zeros((n, n), dtype=bool)
    for s in range(0, n, block):
        e = min(s + block, n)
        maxabs = np.abs(X[s:e, None, :] - X[None, :, :]).max(axis=2)  # (blk,n)
        diffc = y[s:e, None] != y[None, :]
        InD[s:e] = diffc & (maxabs >= gap[s:e, None] - 1e-12)
    return InD


def _attr_cover_count(X, a, gap, InD, covered, block=1000):
    """Number of uncovered pairs (InD & ~covered) that attribute a covers: |x_ia-x_ja|>=gap_i."""
    n = X.shape[0]
    cnt = 0
    xa = X[:, a]
    for s in range(0, n, block):
        e = min(s + block, n)
        cov_a = np.abs(xa[s:e, None] - xa[None, :]) >= gap[s:e, None] - 1e-12
        cnt += int(np.count_nonzero(InD[s:e] & (~covered[s:e]) & cov_a))
    return cnt


def _apply_attr(X, a, gap, InD, covered, block=1000):
    """Mark covered |= (InD & cover_a) for attribute a."""
    n = X.shape[0]
    xa = X[:, a]
    for s in range(0, n, block):
        e = min(s + block, n)
        cov_a = np.abs(xa[s:e, None] - xa[None, :]) >= gap[s:e, None] - 1e-12
        covered[s:e] |= (InD[s:e] & cov_a)


def _covers_all(X, cols, gap, InD, block=1000):
    """True if the columns cols cover every pair in InD."""
    n = X.shape[0]
    for s in range(0, n, block):
        e = min(s + block, n)
        need = InD[s:e].copy()
        if not need.any():
            continue
        cov = np.zeros_like(need)
        for a in cols:
            xa = X[:, a]
            cov |= np.abs(xa[s:e, None] - xa[None, :]) >= gap[s:e, None] - 1e-12
        if np.any(need & ~cov):
            return False
    return True


def fit_reduction_vec(X, y, t=2.0, p=1.0, block=500):
    n, d = X.shape
    gap = _gaps(X, y, t, p)
    InD = _in_disA(X, y, gap, block)
    total = int(InD.sum())
    if total == 0:
        return [], 0
    covered = np.zeros((n, n), dtype=bool)
    red = []
    covered_cnt = 0
    while covered_cnt < total:
        best_a, best_cnt = -1, -1
        for a in range(d):
            if a in red:
                continue
            c = _attr_cover_count(X, a, gap, InD, covered, block)
            if c > best_cnt:  # tie-break: smallest index (increasing a)
                best_cnt, best_a = c, a
        if best_a < 0 or best_cnt == 0:
            break
        _apply_attr(X, best_a, gap, InD, covered, block)
        red.append(best_a)
        covered_cnt = int(covered.sum())
    # backward prune (index tang dan, giong forward_filter)
    for a in sorted(red):
        rest = [c for c in red if c != a]
        if _covers_all(X, rest, gap, InD, block):
            red = rest
    return sorted(red), total
