"""
Xu2025b baseline (PMLCE): local neighborhood rough set + composite entropy.

Implemented from Def. 1, 2, 3, 5 of Xu & Ye (2025), Pattern Recognition 159:111141.
delta = 0.15, the default of the fixed-radius NRS baseline in
IJAR-MHMG-Published/IJAR-Code/baseline_comparison/NRS_Reduction.py (not tuned here).
"""
import numpy as np
import time

DEFAULT_DELTA = 0.15


def neighborhood_matrix(X, delta, chunk=200):
    """NM[i,j] = 1 if dist(x_i,x_j) <= delta (Def. 1 of Xu2025b).
    Computed in row chunks instead of a full (n,n,d) broadcast, which does not fit in
    memory on high-dimensional data (e.g. Isolet, d~600, n~6000). Distances use direct
    coordinate differences (not the Gram identity), as in the MHMG-RS code."""
    n = X.shape[0]
    NM = np.empty((n, n), dtype=bool)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = X[start:end, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        NM[start:end, :] = dist <= delta
    return NM


def lower_upper_by_class(NM, y):
    """Return (lower_count, upper_count) for each object, with respect to its own class
    (used to compute CE) -- Def. 1."""
    n = len(y)
    lower_mask = np.zeros(n, dtype=bool)
    upper_mask = np.zeros(n, dtype=bool)
    for label in np.unique(y):
        Z_i = (y == label)
        # lower: [x]_delta subset of Z_i  <=>  every neighbor of x belongs to Z_i
        neighbor_in_Zi = NM & Z_i[None, :]
        lower_i = np.all(neighbor_in_Zi == NM, axis=1) & (y == label)
        # upper: [x]_delta meets Z_i  <=>  x has at least one neighbor in Z_i
        upper_i = np.any(NM & Z_i[None, :], axis=1) & Z_i
        lower_mask |= lower_i
        upper_mask |= upper_i
    return lower_mask, upper_mask


def composite_entropy(NM, y):
    """CE(B,Z) = -sum_i (|Z_i|/|U|) * ln(|lower_i|/|upper_i|) -- Def. 2.
    Smaller is better (opposite direction to the gamma of MHMG-RS)."""
    n = len(y)
    ce = 0.0
    for label in np.unique(y):
        Z_i = (y == label)
        n_Zi = int(Z_i.sum())
        neighbor_in_Zi = NM & Z_i[None, :]
        lower_i = np.all(neighbor_in_Zi == NM, axis=1) & Z_i
        upper_i = np.any(NM & Z_i[None, :], axis=1) & Z_i
        n_lower = int(lower_i.sum())
        n_upper = int(upper_i.sum())
        if n_upper == 0:
            continue  # empty Z_i; cannot happen since every Z_i has at least one object
        if n_lower == 0:
            return float("inf")  # Proposition 1(3): CE -> infinity
        ce -= (n_Zi / n) * np.log(n_lower / n_upper)
    return ce


def _ce_subset(X, y, delta, cols):
    """CE(B,Z) for B = column indices 'cols' (empty -> CE=+inf, handled separately)."""
    if len(cols) == 0:
        return float("inf")
    return composite_entropy(neighborhood_matrix(X[:, cols], delta), y)


def fit_reduction(X, y, delta=DEFAULT_DELTA):
    """Algorithm 2 (PMLCE-S) of Xu2025b with all three phases. A plain forward-greedy
    search from the empty set cannot replace it: with few attributes on multi-class
    data, CE = +inf because the lower approximation of some class is empty.

    Phase 1 (IM): from the FULL set A, compute IM(a,A,Z)=CE(A-{a},Z)-CE(A,Z) for each a;
      the attributes with IM>0 form the initial B.
    Phase 2 (SM): repeatedly add the attribute of A-B with the largest
      SM(a,B,Z)=CE(B,Z)-CE(B+{a},Z), until CE(B,Z) == CE(A,Z).
    Phase 3 (backward pruning): remove b from B if CE(B-{b},Z) == CE(B,Z) (redundant).
    """
    t0 = time.time()
    n_features = X.shape[1]
    A = list(range(n_features))
    ce_A = _ce_subset(X, y, delta, A)

    # Pha 1: IM
    B = []
    for a in A:
        rest = [c for c in A if c != a]
        ce_rest = _ce_subset(X, y, delta, rest)
        im = ce_rest - ce_A
        if im > 1e-9:
            B.append(a)

    # Phase 2: SM, until CE(B) == CE(A) (or no attribute is left)
    guard = 0
    while _ce_subset(X, y, delta, B) != ce_A and len(B) < n_features and guard < n_features:
        guard += 1
        remaining = [a for a in A if a not in B]
        if not remaining:
            break
        ce_B = _ce_subset(X, y, delta, B)
        best_a, best_sm = None, -float("inf")
        for a in remaining:
            ce_Ba = _ce_subset(X, y, delta, B + [a])
            sm = ce_B - ce_Ba if np.isfinite(ce_B) or np.isfinite(ce_Ba) else 0.0
            if ce_B == float("inf") and ce_Ba == float("inf"):
                sm = 0.0  # both sides +inf -> indistinguishable, treated as 0
            elif ce_B == float("inf") and np.isfinite(ce_Ba):
                sm = float("inf")  # leaving +inf is an absolute improvement
            if sm > best_sm:
                best_sm = sm
                best_a = a
        if best_a is None:
            break
        B.append(best_a)

    # Pha 3: backward prune
    changed = True
    while changed:
        changed = False
        ce_B = _ce_subset(X, y, delta, B)
        for b in list(B):
            rest = [c for c in B if c != b]
            if _ce_subset(X, y, delta, rest) == ce_B:
                B.remove(b)
                changed = True
                break

    duration = time.time() - t0
    ce_final = _ce_subset(X, y, delta, B)
    return sorted(B), duration, ce_final
