"""
FNDS positive-region reducer (Li2026) for crisp labels.
- discernibility (Def. 11): f_ij = {a | [u_i]^d_a(u_j) <= [u_i]^d_D(u_j)}
- a reduct B preserves Pos <=> B meets every clause f_ij (Theorem 1, hitting set)
- RABAF (Algorithm 2): greedy by attribute frequency + backward pruning.

Crisp labels: [u_i]^d_D(u_j) = 1 for the same class, 0 otherwise.
=> constraining clauses come only from DIFFERENT-class pairs: f_ij = {a | R_a(i,j) < delta}.
"""
import numpy as np


def build_clauses(X, y, delta):
    """Return the list of clauses (frozensets of attribute indices) to hit.
    Only different-class pairs constrain; f_ij = {a | 1-|x_ia-x_ja| < delta}
    = {a | |x_ia - x_ja| > 1-delta}. u_i must belong to Pos_C to count (Lemma 1)."""
    n, d = X.shape
    # Pos_C over all of C
    from local_fnds import cond_granule_matrix, dec_granule_matrix_crisp, positive_region
    Gc = cond_granule_matrix(X, delta)
    Gd = dec_granule_matrix_crisp(y, delta)
    pos = set(int(i) for i in positive_region(Gc, Gd))

    thr = 1.0 - delta  # |x_ia-x_ja| > thr  <=>  R_a < delta
    clauses = []
    for i in range(n):
        if i not in pos:
            continue
        diff_i = np.abs(X[i][None, :] - X)  # (n, d)
        diff_class = y != y[i]              # different-class pair
        for j in np.where(diff_class)[0]:
            f = np.where(diff_i[j] > thr + 1e-12)[0]  # attributes that discern i and j
            if len(f) == 0:
                # u_i in Pos but not separable from a different-class u_j -> contradiction
                # (excluded by Lemma 1); defensive: skip
                continue
            clauses.append(frozenset(int(a) for a in f))
    return clauses, pos


def rabaf(clauses, n_features):
    """Algorithm 2: greedy hitting set theo tan suat + backward prune."""
    remaining = list(clauses)
    red = set()
    while remaining:
        freq = np.zeros(n_features)
        for cl in remaining:
            for a in cl:
                freq[a] += 1
        if freq.max() == 0:
            break
        a_star = int(freq.argmax())
        red.add(a_star)
        remaining = [cl for cl in remaining if a_star not in cl]
    # backward pruning: drop a if every clause is still hit by Red-{a}
    for a in sorted(red):
        rest = red - {a}
        if all(len(cl & rest) > 0 for cl in clauses):
            red = rest
    return sorted(red)


def fit_reduction(X, y, delta):
    """Tra ve (reduct, pos_size). reduct la positive-region reduct heuristic."""
    clauses, pos = build_clauses(X, y, delta)
    if not clauses:
        return [], len(pos)
    red = rabaf(clauses, X.shape[1])
    return red, len(pos)


def positive_region_size(X, y, cols, delta):
    """|Pos_B(D)| for B = cols. Used to verify that a reduct preserves Pos."""
    from local_fnds import cond_granule_matrix_subset, dec_granule_matrix_crisp, positive_region
    Gc = cond_granule_matrix_subset(X, cols, delta)
    Gd = dec_granule_matrix_crisp(y, delta)
    return len(positive_region(Gc, Gd))
