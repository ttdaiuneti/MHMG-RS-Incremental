"""
Incremental positive-region maintenance for the Li2026 baseline (Section 4 of the
paper) under a FIXED reduct B_init, symmetric to run_incremental_stream of Xu2025b:
the remaining 20% are streamed, Pos_{B_init}(D) is updated after each insertion and
periodically (every 25 steps) compared with a recomputation. Each step is timed.

Section 4 of Li2026: inserting u_new can (a) remove some objects from Pos (Del set)
because a new different-class pair violates [u_i]^d_C(u_new) > [u_i]^d_D(u_new); (b) add
u_new itself to Pos if the condition holds with every existing object. Under a fixed
B_init, a boolean array in_pos and the cached condition-granule matrix are kept.

Crisp labels: [u_i]^d_D(u_j) = 1 for the same class, 0 otherwise.
Pos condition: for every j, [u_i]^d_C(u_j) <= [u_i]^d_D(u_j).
  - j in the same class: RHS=1, always satisfied.
  - j in another class: RHS=0, requires granule_C(i,j)=0, i.e. R_C(i,j)<delta.
=> u_i in Pos  <=>  every object j of ANOTHER class has R_C(i,j) < delta.
"""
import time
import numpy as np


def _RC_row(XB_i, XB_all, delta):
    """R_C(i, .) = 1 - max_a |x_ia - x_ja| over the (selected) B_init columns."""
    maxdiff = np.abs(XB_all - XB_i[None, :]).max(axis=1)
    return 1.0 - maxdiff  # not thresholded; compared with delta where used


def pos_mask_batch(XB, y, delta):
    """Recompute Pos: u_i in Pos <=> every j of another class has R_C(i,j) < delta."""
    n = XB.shape[0]
    in_pos = np.ones(n, dtype=bool)
    for start in range(0, n, 200):
        end = min(start + 200, n)
        diff = np.abs(XB[start:end, None, :] - XB[None, :, :]).max(axis=2)
        RC = 1.0 - diff  # (block, n)
        for r, i in enumerate(range(start, end)):
            enemy = y != y[i]
            # violated if some enemy j has R_C(i,j) >= delta
            if np.any(RC[r][enemy] >= delta - 1e-12):
                in_pos[i] = False
    return in_pos


def run_incremental_stream(XB, y, init_idx, stream_idx, delta,
                           gt_check_every=25):
    """Duy tri Pos_{B_init} khi stream. Tra ve (stream_time, gt_checks)."""
    n_total = len(y)
    active = list(init_idx)
    XB_active = XB[init_idx].copy()
    y_active = y[init_idx].copy()
    in_pos = pos_mask_batch(XB_active, y_active, delta)

    per_step = []
    gt_checks = []
    for step, idx in enumerate(stream_idx):
        x_new = XB[idx]
        y_new = y[idx]
        t0 = time.perf_counter()
        # R_C(new, existing)
        RC_new = 1.0 - np.abs(XB_active - x_new[None, :]).max(axis=1)  # (n_cur,)
        enemy_new = y_active != y_new
        # (a) existing Pos members may drop out: if u_i and new differ in class and
        #     R_C(i,new) >= delta, u_i violates the condition -> leaves Pos
        drop = enemy_new & (RC_new >= delta - 1e-12) & in_pos
        in_pos[drop] = False
        # (b) new object: in Pos if no existing enemy has R_C>=delta
        new_in = not np.any(RC_new[enemy_new] >= delta - 1e-12) if enemy_new.any() else True
        # update the active set
        XB_active = np.vstack([XB_active, x_new[None, :]])
        y_active = np.append(y_active, y_new)
        in_pos = np.append(in_pos, new_in)
        active.append(idx)
        per_step.append(time.perf_counter() - t0)

        if step % gt_check_every == 0 or step == len(stream_idx) - 1:
            gt = pos_mask_batch(XB_active, y_active, delta)
            mismatch = int(np.sum(gt != in_pos))
            gt_checks.append({"step": step, "n": len(y_active),
                              "pos_mismatch": mismatch})

    return sum(per_step), gt_checks
