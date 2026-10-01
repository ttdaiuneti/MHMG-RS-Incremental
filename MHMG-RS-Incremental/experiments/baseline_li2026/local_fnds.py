"""
Li2026 baseline (FNDS): fuzzy neighborhood decision system and positive-region
reduction, implemented from Def. 3, 7, 8, 9 and Algorithms 1-2 of Li, Zhang, Wang
(2026), Neurocomputing 702:134640.

Fuzzy similarity (Def. 3):  R_a(u_i,u_j) = 1 - |u_i^a - u_j^a|   (data scaled to [0,1]).
Condition granule (Def. 7): [u_i]^d_B(u_j) = min_a R_a  if >= delta, else 0.
Decision granule (Def. 8): the same over the decision attributes.
  -- Original FNDS: fuzzy decisions (Table 2), used by test1_example2.py.
  -- Crisp UCI labels: R_d = 1 for the same class, 0 otherwise, so the decision
     granule is 1 (>=delta) for the same class and 0 otherwise.
Positive region (Def. 9): u_i in Pos if [u_i]^d_C(u_j) <= [u_i]^d_D(u_j) for every j.
"""
import numpy as np


def cond_granule_matrix(X, delta):
    """[u_i]^delta_C(u_j) for all pairs, over ALL condition attributes C (columns of X).
    Returns an n x n matrix: min_a (1-|x_ia - x_ja|) if >= delta, else 0."""
    n = X.shape[0]
    G = np.ones((n, n), dtype=float)
    # R_C = min_a (1 - |x_ia - x_ja|) = 1 - max_a |x_ia - x_ja|  (1-|.| decreases in |.|)
    for start in range(0, n, 200):
        end = min(start + 200, n)
        diff = np.abs(X[start:end, None, :] - X[None, :, :])  # (block, n, d)
        maxdiff = diff.max(axis=2)                            # max_a |.|
        Rc = 1.0 - maxdiff
        Rc[Rc < delta] = 0.0
        G[start:end, :] = Rc
    return G


def cond_granule_matrix_subset(X, cols, delta):
    if len(cols) == 0:
        # empty B: min over the empty set = +inf -> R_B=1 for every pair (no discrimination)
        n = X.shape[0]
        M = np.ones((n, n))
        M[M < delta] = 0.0
        return M
    return cond_granule_matrix(X[:, cols], delta)


def dec_granule_matrix_fuzzy(Dvals, delta):
    """Original FNDS: fuzzy decisions. Dvals has shape (n, k), k fuzzy decision attributes.
    [u_i]^delta_D(u_j) = min_k (1-|D_ik - D_jk|) if >= delta, else 0."""
    n = Dvals.shape[0]
    diff = np.abs(Dvals[:, None, :] - Dvals[None, :, :])
    Rd = 1.0 - diff.max(axis=2)
    Rd[Rd < delta] = 0.0
    return Rd


def dec_granule_matrix_crisp(y, delta):
    """Crisp labels: R_d = 1 for the same class, 0 otherwise.
    Decision granule = 1 (>=delta always holds for delta<=1) for the same class, else 0."""
    same = (y[:, None] == y[None, :]).astype(float)  # 1 for the same class
    same[same < delta] = 0.0  # for delta<=1: same=1 is kept, same=0 -> 0
    return same


def positive_region(Gc, Gd):
    """Def. 9: u_i in Pos if Gc[i,j] <= Gd[i,j] for EVERY j."""
    return np.where((Gc <= Gd + 1e-12).all(axis=1))[0]
