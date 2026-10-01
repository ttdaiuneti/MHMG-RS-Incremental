"""Build the corrected benchmark files used from v3 onward.

The CSVs of the published MHMG-RS code (IJAR-MHMG-Published/IJAR-Code/12-datasets)
have no header row, but were read with pandas' default header=0, which drops the
first object of every dataset. Files 4 (Dermatology) and 8 (Arrhythmia) were in
addition produced by reading the UCI file that way, imputing missing values with
KNNImputer(n_neighbors=5) on the attributes, and writing the result, so their first
line is the first UCI record turned into (mangled) column names.

This script writes every dataset with an explicit header and all objects:
  * files 1-3, 5-7, 9-12: read with header=None (no other change);
  * files 4 and 8: rebuilt from the original UCI files (ARRHYTHMIA_URL, DERM_URL),
    with the same KNNImputer(n_neighbors=5) on the attributes, now over all objects.
"""
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "IJAR-MHMG-Published", "IJAR-Code", "12-datasets")
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
REBUILD = {4: UCI + "dermatology/dermatology.data", 8: UCI + "arrhythmia/arrhythmia.data"}


def write(df, i):
    df.columns = [f"a{j}" for j in range(df.shape[1] - 1)] + ["class"]
    df.to_csv(os.path.join(HERE, f"{i}.csv"), index=False)


for i in range(1, 13):
    if i in REBUILD:
        with urllib.request.urlopen(REBUILD[i]) as r:
            raw = pd.read_csv(r, header=None, na_values="?")
        X = KNNImputer(n_neighbors=5).fit_transform(raw.iloc[:, :-1].values)
        df = pd.DataFrame(np.column_stack([X, raw.iloc[:, -1].values]))
        df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    else:
        df = pd.read_csv(os.path.join(SRC, f"{i}.csv"), header=None)
    write(df, i)
    print(i, df.shape)
