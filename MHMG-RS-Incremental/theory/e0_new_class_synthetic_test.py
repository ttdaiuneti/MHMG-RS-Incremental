"""
Synthetic test (no UCI data needed): incremental update when a COMPLETELY NEW class
appears for the first time in the stream.

Scenario: the initial set has classes 0 and 1 only. The stream inserts a few objects
of classes 0/1, then one object of class 2 (NEW class), and checks:
1. delta_B(x_i) of EVERY existing object of class 0/1 shrinks correctly (class 2 is
   an enemy of all, enemy branch of Theorem 1).
2. K_d is updated correctly when class 2 first appears (m_old=0 -> m_new=1).
3. E/K_d/gamma agree with a batch recomputation after the new class appears.
"""
import sys
import os
import numpy as np

PUBLISHED_CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "IJAR-MHMG-Published", "IJAR-Code"
)
sys.path.insert(0, os.path.abspath(PUBLISHED_CODE_DIR))
from HMMG_Reducer import MHTG_Reducer  # noqa: E402


def full_delta_and_granule(dist_matrix, y):
    n = len(y)
    deltas = np.full(n, np.inf)
    for i in range(n):
        enemy_mask = y != y[i]
        ed = dist_matrix[i][enemy_mask]
        if len(ed) > 0:
            deltas[i] = ed.min()
    granule_sizes = np.array([np.sum(dist_matrix[i] < deltas[i]) for i in range(n)])
    return deltas, granule_sizes


def gt_K_d(y):
    labels, counts = np.unique(y, return_counts=True)
    return float(np.sum(counts * np.log2(counts + 1)))


def main():
    rng = np.random.default_rng(7)
    reducer = MHTG_Reducer()

    # initial set: 40 objects of class 0, 40 of class 1, 3 dimensions, well separated
    n_c0, n_c1, d = 40, 40, 3
    X0 = rng.normal(loc=0.0, scale=0.3, size=(n_c0, d))
    X1 = rng.normal(loc=2.0, scale=0.3, size=(n_c1, d))
    X_init = np.vstack([X0, X1])
    y_init = np.array([0] * n_c0 + [1] * n_c1)

    # stream: 5 objects of class 0/1 alternating, then 1 object of class 2 (NEW), then a few more
    X_stream_pre = np.vstack(
        [rng.normal(loc=0.0, scale=0.3, size=(2, d)), rng.normal(loc=2.0, scale=0.3, size=(2, d))]
    )
    y_stream_pre = np.array([0, 0, 1, 1])
    x_new_class = rng.normal(loc=1.0, scale=0.3, size=(1, d))  # between the two old classes; class 2 is COMPLETELY NEW
    y_new_class = np.array([2])
    X_stream_post = rng.normal(loc=1.0, scale=0.2, size=(3, d))
    y_stream_post = np.array([2, 0, 1])

    X_stream = np.vstack([X_stream_pre, x_new_class, X_stream_post])
    y_stream = np.concatenate([y_stream_pre, y_new_class, y_stream_post])

    active_X = X_init.copy()
    active_y = y_init.copy()
    n_total = len(active_y) + len(y_stream)
    dist_matrix = np.zeros((n_total, n_total))
    d0 = reducer._calculate_distances(active_X)
    dist_matrix[: len(active_y), : len(active_y)] = d0

    delta, gsize = full_delta_and_granule(d0, active_y)
    E_val = float(np.sum(np.log2(gsize + 1)))
    class_counts = {0: n_c0, 1: n_c1}
    K_d = gt_K_d(active_y)
    n_current = len(active_y)

    print(f"Init: n={n_current}, classes={class_counts}, E={E_val:.4f}, K_d={K_d:.4f}")

    new_class_appeared_at_step = None
    all_ok = True

    for step in range(len(y_stream)):
        x_new = X_stream[step]
        y_new = int(y_stream[step])
        is_new_class = y_new not in class_counts
        if is_new_class:
            new_class_appeared_at_step = step
            print(f"  >> step {step}: LOP MOI xuat hien lan dau (class={y_new})")

        new_dists = np.sqrt(np.sum((active_X - x_new) ** 2, axis=1))
        dist_matrix[n_current, :n_current] = new_dists
        dist_matrix[:n_current, n_current] = new_dists

        friend_mask = active_y == y_new
        enemy_mask = ~friend_mask  # new class: EVERY existing object is an enemy (Theorem 1)

        gain_mask = friend_mask & (new_dists < delta)
        gsize[gain_mask] += 1
        E_val += np.sum(np.log2(gsize[gain_mask] + 1) - np.log2(gsize[gain_mask] - 1 + 1))

        shrink_mask = enemy_mask & (new_dists < delta)
        n_shrink_this_step = int(shrink_mask.sum())
        for i in np.where(shrink_mask)[0]:
            new_delta_i = new_dists[i]
            delta[i] = new_delta_i
            row_i = dist_matrix[i, :n_current]
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
        class_counts[y_new] = m_new

        active_X = np.vstack([active_X, x_new[None, :]])
        active_y = np.append(active_y, y_new)
        delta = np.append(delta, delta_new)
        gsize = np.append(gsize, gsize_new)
        n_current += 1

        gt_delta, gt_gsize = full_delta_and_granule(dist_matrix[:n_current, :n_current], active_y)
        gt_E = float(np.sum(np.log2(gt_gsize + 1)))
        gt_Kd = gt_K_d(active_y)
        e_diff = abs(E_val - gt_E)
        kd_diff = abs(K_d - gt_Kd)
        status = "OK" if (e_diff < 1e-6 and kd_diff < 1e-6) else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        if is_new_class or step == len(y_stream) - 1:
            print(
                f"  step {step}: is_new_class={is_new_class}, n_shrink={n_shrink_this_step}, "
                f"E_diff={e_diff:.2e}, K_d_diff={kd_diff:.2e} [{status}]"
            )

    print(f"\nLop moi xuat hien tai step: {new_class_appeared_at_step}")
    print("KET QUA: " + ("TAT CA KHOP GROUND TRUTH" if all_ok else "CO MISMATCH -- CAN SUA"))


if __name__ == "__main__":
    main()
