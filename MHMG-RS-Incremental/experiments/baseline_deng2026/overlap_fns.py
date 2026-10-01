"""
0-overlap functions and fuzzy negation for the Deng2026 baseline
(Def. 2.3, 2.5, 2.6 of the paper).

O_t(x,y) = max{x^t + y^t - 1, 0}   (0-overlap function, Example 2.6(1); t=1 => Lukasiewicz).
N_p(x)   = 1 - x^p                  (fuzzy negation, Example 2.4(2); p=1 => 1-x).
"""
import numpy as np


def O_t(x, y, t=1.0):
    """0-overlap function O_t(x,y)=max{x^t+y^t-1,0}. Vectorized."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    return np.maximum(np.power(x, t) + np.power(y, t) - 1.0, 0.0)


def N_p(x, p=1.0):
    """Fuzzy negation N_p(x)=1-x^p."""
    x = np.asarray(x, dtype=float)
    return 1.0 - np.power(x, p)


# ---- check the properties of Def. 2.5 (0-overlap function) and Def. 2.3 (negation) ----
def _test_properties():
    rng = np.random.default_rng(0)
    xs = rng.random(2000); ys = rng.random(2000)
    ok = True
    for t in [0.5, 1.0, 2.0]:
        # (1) doi xung
        ok &= np.allclose(O_t(xs, ys, t), O_t(ys, xs, t))
        # (2) tang theo y: O(x,y) <= O(x,z) khi y<=z
        z = np.maximum(ys, rng.random(2000))
        ok &= np.all(O_t(xs, ys, t) <= O_t(xs, z, t) + 1e-12)
        # (3) O(x,y)=0 khi xy=0
        ok &= np.all(O_t(np.zeros(5), rng.random(5), t) == 0.0)
        # (4) O(x,y)=1 iff xy=1
        ok &= (O_t(1.0, 1.0, t) == 1.0)
        ok &= np.all(O_t(rng.random(5)*0.999, np.ones(5), t) < 1.0)
    # (neutral element 1) O(1,x)=x holds only for t=1 (Lukasiewicz). For t!=1,
    # O_t(1,x)=x^t != x, so O_t is a general 0-overlap function without neutral element 1.
    # Deng2026 uses an O with neutral element 1, so t=1 is the reference setting.
    ok &= np.allclose(O_t(np.ones(2000), xs, 1.0), xs)
    for p in [0.5, 1.0, 2.0]:
        ok &= (abs(N_p(0.0, p) - 1.0) < 1e-12) and (abs(N_p(1.0, p) - 0.0) < 1e-12)
        a = np.sort(rng.random(100))
        ok &= np.all(np.diff(N_p(a, p)) <= 1e-12)  # giam
    return ok


if __name__ == "__main__":
    print("TEST 2 (0-OF & negation properties):", "PASS" if _test_properties() else "FAIL")
