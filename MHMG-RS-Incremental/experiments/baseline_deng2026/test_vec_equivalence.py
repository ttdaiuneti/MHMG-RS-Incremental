"""Check that the vectorized Deng2026 reducer used in the experiments selects the
same reduct as the reference (frozenset) implementation, on 120-object samples
of four datasets. Run: python test_vec_equivalence.py"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reducer_deng as ref  # noqa: E402
import reducer_deng_vec as vec  # noqa: E402

for f in ["1.csv", "3.csv", "5.csv", "10.csv"]:
    df = pd.read_csv(os.path.join(HERE, "..", "..", "datasets_v3", f))
    X = MinMaxScaler().fit_transform(df.iloc[:, :-1].values)
    y = df.iloc[:, -1].values
    idx = np.random.default_rng(42).permutation(len(y))[:120]
    a = ref.fit_reduction(X[idx], y[idx], t=2.0)
    b = vec.fit_reduction_vec(X[idx], y[idx], t=2.0)
    ra, rb = (a[0] if isinstance(a, tuple) else a), (b[0] if isinstance(b, tuple) else b)
    assert list(ra) == list(rb), (f, ra, rb)
    print(f, list(ra), "equal")
