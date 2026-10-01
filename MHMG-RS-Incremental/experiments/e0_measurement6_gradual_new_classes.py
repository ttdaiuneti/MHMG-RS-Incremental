"""
Stream in which several new classes appear one after another, on real data
(Dermatology, 6 classes): 3 classes are held out of the initial set and appear
gradually in the stream (theory/e0_new_class_synthetic_test.py covers a single new
class on synthetic data).

Design: initial set = 80% of the 3 'known' classes (1,2,3). Stream = the remaining 20%
of those classes + ALL objects of the 3 'unseen' classes (4,5,6), shuffled, so that
each new class has one well-defined first appearance.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLISHED_CODE_DIR = os.path.join(HERE, "..", "..", "IJAR-MHMG-Published", "IJAR-Code")
sys.path.insert(0, os.path.abspath(PUBLISHED_CODE_DIR))
from HMMG_Reducer import MHTG_Reducer  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_v3")
SEED = 42
INIT_FRAC = 0.8
KNOWN_CLASSES = [1, 2, 3]
HELD_OUT_CLASSES = [4, 5, 6]


def pairwise_distances_direct(X, chunk=200):
    n = X.shape[0]
    D = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = X[start:end, None, :] - X[None, :, :]
        D[start:end, :] = np.sqrt(np.sum(diff ** 2, axis=2))
    return D


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


def main():
    path = os.path.join(DATASET_DIR, "4.csv")  # Dermatology
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = df.iloc[:, -1].values
    X = MinMaxScaler().fit_transform(X)
    n_total = len(y)
    print(f"Dermatology: n={n_total}, d={X.shape[1]}, classes={sorted(np.unique(y))}")

    rng = np.random.default_rng(SEED)
    known_idx_all = np.where(np.isin(y, KNOWN_CLASSES))[0]
    held_idx_all = np.where(np.isin(y, HELD_OUT_CLASSES))[0]
    rng.shuffle(known_idx_all)

    n_known_init = int(len(known_idx_all) * INIT_FRAC)
    init_idx = known_idx_all[:n_known_init]
    stream_known_idx = known_idx_all[n_known_init:]
    stream_idx = np.concatenate([stream_known_idx, held_idx_all])
    rng.shuffle(stream_idx)

    print(f"init: {len(init_idx)} doi tuong, lop {sorted(np.unique(y[init_idx]))}")
    print(f"stream: {len(stream_idx)} doi tuong, lop {sorted(np.unique(y[stream_idx]))} "
          f"(3 lop {HELD_OUT_CLASSES} CHUA TUNG xuat hien trong init)")

    reducer = MHTG_Reducer()
    reduct_init, _, gamma_init = reducer.fit_reduction(X[init_idx], y[init_idx])
    print(f"B_init: |R|={len(reduct_init)} (chi hoc tu 3 lop da biet)")
    if len(reduct_init) == 0:
        print("[DUNG] reduct init rong")
        return
    XB = X[:, reduct_init]

    n_total_stream = len(stream_idx)
    dist_matrix = np.zeros((n_total, n_total))
    active_X = XB[init_idx].copy()
    active_y = y[init_idx].copy()
    d0 = pairwise_distances_direct(active_X)
    dist_matrix[:len(init_idx), :len(init_idx)] = d0
    delta, gsize = full_delta_and_granule(d0, active_y)
    E_val = float(np.sum(np.log2(gsize + 1)))
    n_current = len(init_idx)

    first_seen = set(np.unique(active_y).tolist())
    rows = []
    gt_max_diff = 0.0

    for step, idx in enumerate(stream_idx):
        x_new = XB[idx]
        y_new = y[idx]
        is_new_class_event = y_new not in first_seen
        if is_new_class_event:
            first_seen.add(y_new)

        new_dists = np.sqrt(np.sum((active_X - x_new) ** 2, axis=1))
        dist_matrix[n_current, :n_current] = new_dists
        dist_matrix[:n_current, n_current] = new_dists

        friend_mask = active_y == y_new
        enemy_mask = ~friend_mask
        gain_mask = friend_mask & (new_dists < delta)
        n_gain = int(gain_mask.sum())
        gsize[gain_mask] += 1
        E_val += np.sum(np.log2(gsize[gain_mask] + 1) - np.log2(gsize[gain_mask] - 1 + 1))

        shrink_mask = enemy_mask & (new_dists < delta)
        shrink_idx_arr = np.where(shrink_mask)[0]
        n_shrink = len(shrink_idx_arr)
        for i in shrink_idx_arr:
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

        active_X = np.vstack([active_X, x_new[None, :]])
        active_y = np.append(active_y, y_new)
        delta = np.append(delta, delta_new)
        gsize = np.append(gsize, gsize_new)
        n_current += 1

        shrink_frac = n_shrink / max(n_current - 1, 1)
        gain_frac = n_gain / max(n_current - 1, 1)
        rows.append({
            "step": step, "y_new": int(y_new), "is_new_class_event": is_new_class_event,
            "n_active_before": n_current - 1, "n_gain": n_gain, "n_shrink": n_shrink,
            "shrink_frac": shrink_frac, "gain_frac": gain_frac,
        })

        # periodic check (every 20 steps, at every new-class event and at the last step)
        if step % 20 == 0 or is_new_class_event or step == len(stream_idx) - 1:
            gt_delta, gt_gsize = full_delta_and_granule(dist_matrix[:n_current, :n_current], active_y)
            gt_E = float(np.sum(np.log2(gt_gsize + 1)))
            diff = abs(E_val - gt_E)
            gt_max_diff = max(gt_max_diff, diff)
            if is_new_class_event:
                print(f"  step {step}: LOP MOI xuat hien lan dau (class={int(y_new)}), "
                      f"shrink_frac={shrink_frac:.3f} gain_frac={gain_frac:.3f}, "
                      f"ground-truth diff={diff:.2e}")

    df_rows = pd.DataFrame(rows)
    new_class_rows = df_rows[df_rows.is_new_class_event]
    other_rows = df_rows[~df_rows.is_new_class_event]
    print(f"\n=== TONG KET ===")
    print(f"So su kien lop-moi-xuat-hien: {len(new_class_rows)} (dung ky vong: {len(HELD_OUT_CLASSES)})")
    print(f"shrink_frac tai su kien lop moi: mean={new_class_rows.shrink_frac.mean():.3f} "
          f"min={new_class_rows.shrink_frac.min():.3f} max={new_class_rows.shrink_frac.max():.3f}")
    print(f"shrink_frac buoc thuong (khong phai lop moi): mean={other_rows.shrink_frac.mean():.4f} "
          f"max={other_rows.shrink_frac.max():.3f}")
    print(f"Ground-truth max abs diff toan luong: {gt_max_diff:.2e}")

    out_path = os.path.join(HERE, "e0_measurement6_gradual_new_classes.csv")
    df_rows.to_csv(out_path, index=False)
    print(f"Da luu: {out_path}")


if __name__ == "__main__":
    main()
