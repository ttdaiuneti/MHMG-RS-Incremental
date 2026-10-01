"""
Numerical check of Theorem 5 (batch insertion, order invariance).

1. Correctness: does the closed-form update for inserting k objects at once agree
   with a recomputation from scratch on the full X?
2. Order invariance: do k sequential single-object insertions (the Theorem 1-4 code
   of e1_pipeline.py), in several orders, give the same final state?
3. Speed: is the vectorized batch formula (reusing the cached (n x n) block and
   computing only the (n x k) and (k x k) blocks) faster than k sequential calls?

Distances come from pairwise_distances_direct/full_delta_and_granule of
e1_pipeline.py (a single distance formula throughout).
"""
import sys
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e1_pipeline import pairwise_distances_direct, full_delta_and_granule  # noqa: E402

PUBLISHED_CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "IJAR-MHMG-Published", "IJAR-Code"
)
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_v3")

DATASETS = [
    {"id": "D1", "name": "Wine", "file": "1.csv"},
    {"id": "D9", "name": "Spambase", "file": "9.csv"},
    {"id": "D6", "name": "Urban", "file": "6.csv"},
]

SEED = 42
INIT_FRAC = 0.8
BATCH_SIZE = 15
N_ORDER_TRIALS = 3   # number of different sequential insertion orders tested for order invariance
N_TIMING_REPEATS = 20


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    return X, y


def batch_closed_form(active_X, active_y, cached_dist_nn, delta_old, gsize_old,
                       E_old, batch_X, batch_y, return_blocks=False):
    """Theorem 5, closed form, with the same locality as the sequential path (no full
    matrix assembly and rescan, which costs O(n^2)):
    (a) compute only the two NEW blocks (n x k), (k x k), vectorized;
    (b) re-read one ROW of cached_dist_nn (a view, no matrix copy) only for the
        objects whose radius shrinks (shrunk_idx, a small subset);
    (c) friend gain for ALL existing objects is one vectorized (n x k) operation
        and does not need cached_dist_nn."""
    n = active_X.shape[0]
    k = batch_X.shape[0]

    # Split the (n,k) block into CHUNK rows of active_X to avoid a full (n,k,d)
    # temporary array (up to ~1-2 GB for large n,k); the formula stays
    # sqrt(sum(diff**2)) (not the Gram identity).
    chunk = 200
    D_ob = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff_chunk = active_X[start:end, None, :] - batch_X[None, :, :]
        D_ob[start:end, :] = np.sqrt(np.sum(diff_chunk ** 2, axis=2))
    diff_bb = batch_X[:, None, :] - batch_X[None, :, :]
    D_bb = np.sqrt(np.sum(diff_bb ** 2, axis=2))          # (k, k), luon nho

    friend_ob = active_y[:, None] == batch_y[None, :]      # (n, k)
    enemy_ob = ~friend_ob

    # --- radii of EXISTING objects ---
    min_new_enemy = np.where(enemy_ob, D_ob, np.inf).min(axis=1)
    delta_new = np.minimum(delta_old, min_new_enemy)
    shrunk_mask = delta_new < delta_old
    shrunk_idx = np.where(shrunk_mask)[0]

    # --- friend gain: new friends inside the NEW radius, for all n objects (vectorized) ---
    within_new_radius = friend_ob & (D_ob < delta_new[:, None])
    n_new_friends = within_new_radius.sum(axis=1).astype(np.int64)

    # --- enemy shrink: re-read only shrunk_idx (one-row view, no matrix copy) ---
    gsize_after_shrink = gsize_old.copy().astype(np.int64)
    for i in shrunk_idx:
        row_i = cached_dist_nn[i, :]                      # O(n) view, no full copy
        gsize_after_shrink[i] = int(np.sum(row_i < delta_new[i]))

    gsize_new_old = gsize_after_shrink + n_new_friends     # combine shrink + gain, in order
    E_new = E_old + float(np.sum(np.log2(gsize_new_old + 1) - np.log2(gsize_old + 1)))

    # --- radius + granule of the k NEW objects (reusing D_ob/D_bb) ---
    enemy_bo = batch_y[:, None] != active_y[None, :]        # (k, n)
    min_existing_enemy = np.where(enemy_bo, D_ob.T, np.inf).min(axis=1)
    enemy_bb = batch_y[:, None] != batch_y[None, :]
    np.fill_diagonal(enemy_bb, False)
    min_batch_enemy = np.where(enemy_bb, D_bb, np.inf).min(axis=1)
    delta_batch = np.minimum(min_existing_enemy, min_batch_enemy)

    friend_bo = ~enemy_bo
    friend_bb = ~enemy_bb
    np.fill_diagonal(friend_bb, False)  # avoid counting the object itself twice (self_count is separate)
    n_friends_from_old = (friend_bo & (D_ob.T < delta_batch[:, None])).sum(axis=1)
    n_friends_from_batch = (friend_bb & (D_bb < delta_batch[:, None])).sum(axis=1)
    self_count = (delta_batch > 0).astype(np.int64)
    gsize_batch = (n_friends_from_old + n_friends_from_batch + self_count).astype(np.int64)
    E_new += float(np.sum(np.log2(gsize_batch + 1)))

    full_delta = np.concatenate([delta_new, delta_batch])
    full_gsize = np.concatenate([gsize_new_old, gsize_batch])
    full_y = np.concatenate([active_y, batch_y])
    if return_blocks:
        return full_delta, full_gsize, E_new, full_y, D_ob, D_bb
    return full_delta, full_gsize, E_new, full_y


def kd_batch_update(K_d_old, class_counts, batch_y):
    K_d = K_d_old
    cc = dict(class_counts)
    classes, counts = np.unique(batch_y, return_counts=True)
    for c, dm in zip(classes.tolist(), counts.tolist()):
        m_old = cc.get(c, 0)
        m_new = m_old + dm
        K_d += (m_new * np.log2(m_new + 1)) - (m_old * np.log2(m_old + 1) if m_old > 0 else 0.0)
        cc[c] = m_new
    return K_d, cc


def sequential_single_step(active_X, active_y, delta, gsize, E_val, K_d, class_counts,
                            x_new, y_new, dist_cache, n_current):
    """Exactly the per-step logic of e1_pipeline.py::run_incremental_stream
    (Theorems 1-4), including the reuse of the preallocated dist_cache (row_i is not
    recomputed on shrink), so that the speed comparison with Method A is fair
    (both use the cache)."""
    new_dists = np.sqrt(np.sum((active_X - x_new) ** 2, axis=1))
    dist_cache[n_current, :n_current] = new_dists
    dist_cache[:n_current, n_current] = new_dists

    friend_mask = active_y == y_new
    enemy_mask = ~friend_mask
    gain_mask = friend_mask & (new_dists < delta)
    gsize = gsize.copy()
    gsize[gain_mask] += 1
    E_val += np.sum(np.log2(gsize[gain_mask] + 1) - np.log2(gsize[gain_mask] - 1 + 1))

    delta = delta.copy()
    shrink_mask = enemy_mask & (new_dists < delta)
    for i in np.where(shrink_mask)[0]:
        new_delta_i = new_dists[i]
        delta[i] = new_delta_i
        row_i = dist_cache[i, :n_current]
        new_size = int(np.sum(row_i < new_delta_i))
        E_val += np.log2(new_size + 1) - np.log2(gsize[i] + 1)
        gsize[i] = new_size

    enemy_dists = new_dists[enemy_mask]
    delta_new = float(enemy_dists.min()) if enemy_dists.size > 0 else np.inf
    friend_dists = new_dists[friend_mask]
    self_count = 1 if delta_new > 0 else 0
    gsize_new = int(np.sum(friend_dists < delta_new)) + self_count
    E_val += np.log2(gsize_new + 1)

    m_old = class_counts.get(y_new, 0)
    m_new = m_old + 1
    K_d += (m_new * np.log2(m_new + 1)) - (m_old * np.log2(m_old + 1) if m_old > 0 else 0.0)
    class_counts = dict(class_counts)
    class_counts[y_new] = m_new

    active_X = np.vstack([active_X, x_new[None, :]])
    active_y = np.append(active_y, y_new)
    delta = np.append(delta, delta_new)
    gsize = np.append(gsize, gsize_new)
    return active_X, active_y, delta, gsize, E_val, K_d, class_counts


def run_dataset(ds, rng):
    path = os.path.join(DATASET_DIR, ds["file"])
    X, y = load_dataset(path)
    X = MinMaxScaler().fit_transform(X)

    n_total = len(y)
    perm = rng.permutation(n_total)
    n_init = int(n_total * INIT_FRAC)
    init_idx = perm[:n_init]
    pool_idx = perm[n_init:]
    if len(pool_idx) < BATCH_SIZE:
        print(f"  [{ds['name']}] pool qua nho ({len(pool_idx)} < {BATCH_SIZE}), bo qua")
        return []

    batch_idx = pool_idx[:BATCH_SIZE]
    active_X0 = X[init_idx].copy()
    active_y0 = y[init_idx].copy()
    batch_X = X[batch_idx]
    batch_y = y[batch_idx]

    d0 = pairwise_distances_direct(active_X0)
    delta0, gsize0 = full_delta_and_granule(d0, active_y0)
    E0 = float(np.sum(np.log2(gsize0 + 1)))
    classes0, counts0 = np.unique(active_y0, return_counts=True)
    class_counts0 = dict(zip(classes0.tolist(), counts0.tolist()))
    K_d0 = float(sum(m * np.log2(m + 1) for m in class_counts0.values()))

    # ---------- Method C: ground truth, recomputed from scratch on the full X ----------
    full_X_gt = np.vstack([active_X0, batch_X])
    full_y_gt = np.concatenate([active_y0, batch_y])
    d_gt = pairwise_distances_direct(full_X_gt)
    delta_gt, gsize_gt = full_delta_and_granule(d_gt, full_y_gt)
    E_gt = float(np.sum(np.log2(gsize_gt + 1)))
    classes_gt, counts_gt = np.unique(full_y_gt, return_counts=True)
    K_d_gt = float(sum(m * np.log2(m + 1) for m in dict(zip(classes_gt.tolist(), counts_gt.tolist())).values()))
    gamma_gt = E_gt / K_d_gt

    # ---------- Method A: batch (closed-form) update, reusing the cached (n x n) ----------
    t0 = time.perf_counter()
    delta_A, gsize_A, E_A, full_y_A = batch_closed_form(
        active_X0, active_y0, d0, delta0, gsize0, E0, batch_X, batch_y
    )
    K_d_A, _ = kd_batch_update(K_d0, class_counts0, batch_y)
    t_batch = time.perf_counter() - t0
    gamma_A = E_A / K_d_A

    rows = []
    diff_delta_AC = float(np.max(np.abs(delta_A - delta_gt)))
    diff_gsize_AC = int(np.max(np.abs(gsize_A - gsize_gt)))
    diff_E_AC = abs(E_A - E_gt)
    diff_Kd_AC = abs(K_d_A - K_d_gt)
    diff_gamma_AC = abs(gamma_A - gamma_gt)
    rows.append({
        "dataset": ds["name"], "check": "A_vs_C_closed_form_vs_ground_truth",
        "max_diff_delta": diff_delta_AC, "max_diff_gsize": diff_gsize_AC,
        "diff_E": diff_E_AC, "diff_Kd": diff_Kd_AC, "diff_gamma": diff_gamma_AC,
    })
    print(f"  [{ds['name']}] A vs C (ground truth): "
          f"max|delta diff|={diff_delta_AC:.2e}, max|gsize diff|={diff_gsize_AC}, "
          f"|E diff|={diff_E_AC:.2e}, |Kd diff|={diff_Kd_AC:.2e}, |gamma diff|={diff_gamma_AC:.2e}")

    # ---------- Method B: SEQUENTIAL insertion in N_ORDER_TRIALS different orders ----------
    n_max = n_init + BATCH_SIZE
    t_seq_total = 0.0
    for trial in range(N_ORDER_TRIALS):
        order = rng.permutation(BATCH_SIZE)
        aX, ay, dl, gs, Ev, Kd, cc = active_X0.copy(), active_y0.copy(), delta0.copy(), gsize0.copy(), E0, K_d0, dict(class_counts0)
        cache = np.zeros((n_max, n_max), dtype=np.float64)
        cache[:n_init, :n_init] = d0
        n_cur = n_init
        t0 = time.perf_counter()
        for j in order:
            aX, ay, dl, gs, Ev, Kd, cc = sequential_single_step(
                aX, ay, dl, gs, Ev, Kd, cc, batch_X[j], batch_y[j], cache, n_cur
            )
            n_cur += 1
        t_seq_total += time.perf_counter() - t0
        gamma_B = Ev / Kd

        # reorder (old active_idx + batch idx) for comparison with delta_gt/gsize_gt
        order_of_rows = np.concatenate([np.arange(n_init), n_init + order])
        inv = np.argsort(order_of_rows)
        delta_B_aligned = dl[inv]
        gsize_B_aligned = gs[inv]

        diff_delta_BC = float(np.max(np.abs(delta_B_aligned - delta_gt)))
        diff_gsize_BC = int(np.max(np.abs(gsize_B_aligned - gsize_gt)))
        diff_E_BC = abs(Ev - E_gt)
        diff_Kd_BC = abs(Kd - K_d_gt)
        diff_gamma_BC = abs(gamma_B - gamma_gt)
        rows.append({
            "dataset": ds["name"], "check": f"B_order{trial}_sequential_vs_ground_truth",
            "max_diff_delta": diff_delta_BC, "max_diff_gsize": diff_gsize_BC,
            "diff_E": diff_E_BC, "diff_Kd": diff_Kd_BC, "diff_gamma": diff_gamma_BC,
        })
        print(f"  [{ds['name']}] B order#{trial} vs C: "
              f"max|delta diff|={diff_delta_BC:.2e}, max|gsize diff|={diff_gsize_BC}, "
              f"|E diff|={diff_E_BC:.2e}, |Kd diff|={diff_Kd_BC:.2e}, |gamma diff|={diff_gamma_BC:.2e}")

    # ---------- Speed: batch (vectorized) vs mean of one sequential insertion ----------
    batch_times, seq_times = [], []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        batch_closed_form(active_X0, active_y0, d0, delta0, gsize0, E0, batch_X, batch_y)
        batch_times.append(time.perf_counter() - t0)

        aX, ay, dl, gs, Ev, Kd, cc = active_X0.copy(), active_y0.copy(), delta0.copy(), gsize0.copy(), E0, K_d0, dict(class_counts0)
        cache = np.zeros((n_max, n_max), dtype=np.float64)
        cache[:n_init, :n_init] = d0
        n_cur = n_init
        t0 = time.perf_counter()
        for j in range(BATCH_SIZE):
            aX, ay, dl, gs, Ev, Kd, cc = sequential_single_step(
                aX, ay, dl, gs, Ev, Kd, cc, batch_X[j], batch_y[j], cache, n_cur
            )
            n_cur += 1
        seq_times.append(time.perf_counter() - t0)

    mean_batch = float(np.mean(batch_times))
    mean_seq = float(np.mean(seq_times))
    speedup = mean_seq / mean_batch if mean_batch > 0 else float("nan")
    print(f"  [{ds['name']}] toc do: batch={mean_batch*1000:.3f}ms, "
          f"sequential={mean_seq*1000:.3f}ms (K={BATCH_SIZE}), speedup={speedup:.2f}x")
    rows.append({
        "dataset": ds["name"], "check": "timing",
        "max_diff_delta": mean_batch, "max_diff_gsize": mean_seq,
        "diff_E": speedup, "diff_Kd": np.nan, "diff_gamma": np.nan,
    })
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    for ds in DATASETS:
        print(f"\n=== {ds['name']} (K={BATCH_SIZE}) ===")
        all_rows.extend(run_dataset(ds, rng))

    out_df = pd.DataFrame(all_rows)
    out_path = os.path.join(os.path.dirname(__file__), "e0_batch_insertion_check.csv")
    out_df.to_csv(out_path, index=False)

    max_diff = out_df.loc[out_df.check != "timing", "max_diff_delta"].astype(float).max()
    print(f"\nDa luu: {out_path}")
    print(f"Sai so lon nhat tren MOI kiem tra dung dan (delta): {max_diff:.2e}")
    if max_diff < 1e-9:
        print("PASS: cong thuc lo + tinh bat-bien-thu-tu khop ground-truth trong sai so dau phay dong.")
    else:
        print("CANH BAO: sai so vuot 1e-9, can kiem tra lai truoc khi dua vao ban thao.")


if __name__ == "__main__":
    main()
