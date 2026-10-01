"""
Deng2026 (0-overlap fuzzy rough set) discernibility reducer for crisp labels
(Def. 2.9, 3.1, 3.2 of the paper).

R_a(x,y) = 1 - |x_a - y_a|.   R_B = inf_{a in B} R_a  (= min).
lambda(x) = R_N([x]_D)(x) = inf_{u: d(u)!=d(x)} N_p(R_A(x,u))
          = min over different-class u of (1 - R_A(x,u)^p)  (R_A on the full attribute set, fixed reference).
DIS(a) = {(x,y): O_t(R_a(x,y), lambda(x)) = 0, d(x)!=d(y)}.
DIS(A) = union_a DIS(a).
Reduct B: DIS(B)=DIS(A) and minimal  <=>  B hits every clause {a: (x,y) in DIS(a)}
  for every pair (x,y) in DIS(A). Forward filtering = greedy hitting set + pruning.
"""
import numpy as np
from overlap_fns import O_t, N_p


def _RA_full(X, p=1.0):
    """R_A(x,u) = min_a (1-|x_a-u_a|) = 1 - max_a|x_a-u_a|, over all columns of X."""
    n = X.shape[0]
    RA = np.empty((n, n))
    for s in range(0, n, 200):
        e = min(s + 200, n)
        RA[s:e] = 1.0 - np.abs(X[s:e, None, :] - X[None, :, :]).max(axis=2)
    return RA


def lambda_vec(X, y, p=1.0):
    """lambda(x) = min_{u: d(u)!=d(x)} N_p(R_A(x,u)). Reference tren full A."""
    RA = _RA_full(X)
    n = len(y)
    lam = np.empty(n)
    NR = N_p(RA, p)  # 1 - R_A^p
    for i in range(n):
        enemy = y != y[i]
        lam[i] = NR[i, enemy].min() if enemy.any() else 1.0  # no enemy -> inf ~ 1
    return lam


def build_dis_clauses(X, y, t=1.0, p=1.0, lam=None):
    """Return a dict {(i,j): frozenset of attributes}, the clause of every pair in DIS(A),
    and the set DIS(A) (pairs). Only different-class pairs are considered."""
    n, d = X.shape
    if lam is None:
        lam = lambda_vec(X, y, p)
    clauses = {}
    lam_t = np.power(lam, t)
    for i in range(n):
        enemy = np.where(y != y[i])[0]
        if len(enemy) == 0:
            continue
        # R_a(i,j) for every attribute a and every enemy j (clipped to [0,1] to avoid -eps^t = nan)
        Ra = np.clip(1.0 - np.abs(X[i][None, :] - X[enemy]), 0.0, 1.0)  # (n_enemy, d)
        # DIS(a) contains (i,j) iff O_t(Ra, lam[i]) = 0  <=>  Ra^t + lam[i]^t <= 1
        cond = (np.power(Ra, t) + lam_t[i]) <= 1.0 + 1e-12  # (n_enemy, d) bool
        for r, j in enumerate(enemy):
            attrs = np.where(cond[r])[0]
            if len(attrs) > 0:
                clauses[(i, int(j))] = frozenset(int(a) for a in attrs)
            # empty attribute set: pair (i,j) is not in DIS(A) -> no constraint
    return clauses


def build_dis_clauses_t1_crosscheck(X, y, lam=None):
    """Independent path for t=1, p=1: DIS(a) contains (i,j) <=> R_a(i,j) <= 1 - lam[i].
    Used to cross-check build_dis_clauses(t=1)."""
    n, d = X.shape
    if lam is None:
        lam = lambda_vec(X, y, 1.0)
    clauses = {}
    for i in range(n):
        enemy = np.where(y != y[i])[0]
        if len(enemy) == 0:
            continue
        Ra = 1.0 - np.abs(X[i][None, :] - X[enemy])
        thr = 1.0 - lam[i]
        cond = Ra <= thr + 1e-12
        for r, j in enumerate(enemy):
            attrs = np.where(cond[r])[0]
            if len(attrs) > 0:
                clauses[(i, int(j))] = frozenset(int(a) for a in attrs)
    return clauses


def forward_filter(clauses, n_features):
    """Greedy hitting set by attribute frequency + backward pruning (as in RABAF)."""
    remaining = list(clauses.values())
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
    for a in sorted(red):
        rest = red - {a}
        if all(len(cl & rest) > 0 for cl in clauses.values()):
            red = rest
    return sorted(red)


def dis_set_of_subset(clauses, B):
    """DIS(B) = set of pairs (i,j) whose clause meets B."""
    Bs = set(B)
    return set(k for k, cl in clauses.items() if len(cl & Bs) > 0)


def fit_reduction(X, y, t=1.0, p=1.0):
    clauses = build_dis_clauses(X, y, t, p)
    if not clauses:
        return [], 0
    red = forward_filter(clauses, X.shape[1])
    return red, len(clauses)
