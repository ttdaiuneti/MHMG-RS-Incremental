"""Relate the gamma-based trigger signals of pilot_trigger_signal.csv to the end-of-stream
accuracy gaps (seed 42 from stageA_accuracy_checkpoints.csv, other seeds from multi_seed_w*.csv)."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))

sig = pd.read_csv(os.path.join(HERE, "pilot_trigger_signal.csv"))
ms = pd.concat([pd.read_csv(os.path.join(HERE, f"multi_seed_w{w}.csv")) for w in (1, 2, 3)])
ck = pd.read_csv(os.path.join(HERE, "stageA_accuracy_checkpoints.csv"))
ck = ck[ck.stream_type == "iid"]
last = ck.loc[ck.groupby(["dataset_name", "reduct_used"]).step.idxmax()].set_index(["dataset_name", "reduct_used"])
s42 = pd.DataFrame([{"seed": 42, "dataset_name": d,
                     "gap_3nn_pp": 100 * (last.loc[(d, "B_full")].acc_3nn_mean - last.loc[(d, "B_init")].acc_3nn_mean),
                     "gap_svm_pp": 100 * (last.loc[(d, "B_full")].acc_svm_mean - last.loc[(d, "B_init")].acc_svm_mean)}
                    for d in last.index.get_level_values(0).unique()])
gaps = pd.concat([ms[["seed", "dataset_name", "gap_3nn_pp", "gap_svm_pp"]], s42])
d = sig.merge(gaps, on=["seed", "dataset_name"], how="inner", validate="one_to_one")
print(f"runs: {len(d)} (signals {len(sig)}, gaps {len(gaps)})")

d["drop"] = d.g_init - d.g_end                   # decrease of the maintained gamma
d["add_gain"] = d.add_best - d.g_end             # best single-attribute gain at the end
d["oracle"] = d.g_full - d.g_end                 # needs a new search; reference only
d["gap_max"] = d[["gap_3nn_pp", "gap_svm_pp"]].max(axis=1)
d["bad"] = (d[["gap_3nn_pp", "gap_svm_pp"]] > 5).any(axis=1)
# reducer's own rule (gamma rounded to 2 decimals): B_init no longer locally optimal
d["trig_add"] = d.add_best.round(2) > d.g_end.round(2)
d["trig_drop"] = d.g_end.round(2) < d.g_init.round(2)

print("\nSpearman with the accuracy gap (3-NN / SVM):")
for s in ["drop", "add_gain", "oracle"]:
    r3 = spearmanr(d[s], d.gap_3nn_pp); rs = spearmanr(d[s], d.gap_svm_pp)
    print(f"  {s:9s} rho3={r3.statistic:+.2f} (p={r3.pvalue:.3g})  rhoS={rs.statistic:+.2f} (p={rs.pvalue:.3g})")

print("\nTriggers (fires / mean gap when fired vs not / bad runs caught / same reduct when fired):")
for t in ["trig_add", "trig_drop"]:
    f, nf = d[d[t]], d[~d[t]]
    u = mannwhitneyu(f.gap_max, nf.gap_max) if len(f) and len(nf) else None
    print(f"  {t}: fires {len(f)}/{len(d)}; gap3 {f.gap_3nn_pp.mean():+.2f} vs {nf.gap_3nn_pp.mean():+.2f}; "
          f"gapS {f.gap_svm_pp.mean():+.2f} vs {nf.gap_svm_pp.mean():+.2f}; "
          f"bad caught {f.bad.sum()}/{d.bad.sum()}; unchanged reduct among fired {f.same_reduct.sum()}"
          + (f"; MWU p={u.pvalue:.3g}" if u else ""))

print("\nBad runs (>5 pp loss):")
print(d[d.bad][["seed", "dataset_name", "size_init", "gap_3nn_pp", "gap_svm_pp", "drop", "add_gain", "trig_add", "trig_drop"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nPer dataset:")
print(d.groupby("dataset_name")[["drop", "add_gain", "gap_3nn_pp", "gap_svm_pp", "trig_add"]].mean()
      .to_string(float_format=lambda v: f"{v:.3f}"))

print("\nRecall on reduct changes (B_full != B_init):")
for t in ["trig_add", "trig_drop"]:
    ch, st = d[~d.same_reduct], d[d.same_reduct]
    print(f"  {t}: changed {len(ch)}, fired {ch[t].sum()} | unchanged {len(st)}, fired {st[t].sum()}")
