"""
Sanity test with a known answer: reproduces Example 2 of Li2026
(page 3, Table 2, delta=0.75). Published result:
  Pos^0.75_C(D) = {u1,u2,u3,u4,u5,u7,u9}   (u6, u8 excluded).
The matrices M_C and M_D printed in the paper are compared entry by entry.

Must pass before the implementation is used on real data.
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from local_fnds import cond_granule_matrix, dec_granule_matrix_fuzzy, positive_region

# Table 2 (FNDS): condition a1..a4, decision fuzzy d1,d2
C = np.array([
    [0.22, 0.28, 0.19, 0.25],
    [0.32, 0.35, 0.29, 0.38],
    [0.25, 0.30, 0.32, 0.28],
    [0.55, 0.48, 0.59, 0.62],
    [0.42, 0.38, 0.45, 0.35],
    [0.68, 0.72, 0.65, 0.78],
    [0.85, 0.78, 0.95, 0.95],
    [0.72, 0.68, 0.75, 0.65],
    [0.90, 0.95, 0.85, 0.98],
])
Dv = np.array([
    [0.58, 0.63],
    [0.65, 0.60],
    [0.61, 0.65],
    [0.62, 0.80],
    [0.64, 0.75],
    [0.57, 0.74],
    [0.72, 0.90],
    [0.79, 0.85],
    [0.77, 0.88],
])
delta = 0.75

Gc = cond_granule_matrix(C, delta)
Gd = dec_granule_matrix_fuzzy(Dv, delta)

# --- compare selected entries of M_C, M_D with the paper ---
# M_C[0,1] (u1,u2)=0.87 ; M_C[0,6] (u1,u7)=0 ; M_D[0,1]=0.93 ; M_D[0,6]=0
checks = [
    ('M_C[u1,u2]', Gc[0,1], 0.87),
    ('M_C[u1,u3]', Gc[0,2], 0.87),
    ('M_C[u1,u7]', Gc[0,6], 0.00),
    ('M_C[u4,u6]', Gc[3,5], 0.76),
    ('M_C[u4,u8]', Gc[3,7], 0.80),
    ('M_D[u1,u2]', Gd[0,1], 0.93),
    ('M_D[u1,u7]', Gd[0,6], 0.00),
    ('M_D[u3,u7]', Gd[2,6], 0.75),
    ('M_D[u2,u9]', Gd[1,8], 0.00),
]
print('=== element checks vs PDF (M_C, M_D) ===')
ok_mat = True
for name, got, exp in checks:
    hit = abs(got - exp) < 0.005
    ok_mat &= hit
    print(f'  {name}: got {got:.2f}  expected {exp:.2f}  {"OK" if hit else "MISMATCH"}')

pos = positive_region(Gc, Gd)
pos_labels = sorted(int(i) + 1 for i in pos)  # u-index 1-based
expected = [1, 2, 3, 4, 5, 7, 9]
print('\n=== positive region ===')
print('  got     :', pos_labels)
print('  expected:', expected)
ok_pos = (pos_labels == expected)
print('  ', 'OK' if ok_pos else 'MISMATCH')

print('\nTEST 1', 'PASS' if (ok_mat and ok_pos) else 'FAIL')
sys.exit(0 if (ok_mat and ok_pos) else 1)
