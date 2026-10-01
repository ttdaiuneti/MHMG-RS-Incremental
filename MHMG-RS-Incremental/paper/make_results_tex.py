"""Generate every number and table of Section 5 from the raw result files.

Writes paper/generated/numbers.tex (one \\newcommand per number quoted in the
text) and paper/generated/tab_*.tex (table bodies). Nothing in Section 5 is
typed by hand; re-run this script after any experiment is re-run.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")
THE = os.path.join(HERE, "..", "theory")
DATA = os.path.join(HERE, "..", "datasets_v3")
OUT = os.path.join(HERE, "generated")
os.makedirs(OUT, exist_ok=True)

ORDER = ["Wine", "WDBC", "Iono", "Derm", "Sonar", "Urban", "Musk", "Arrhy",
         "Spambase", "Parkinsons", "Isolet", "Letter"]
LONG = {"Iono": "Ionosphere", "Derm": "Dermatology", "Arrhy": "Arrhythmia"}
STEP_COLS = ["t_delta_friend_sec", "t_shrink_cascade_sec", "t_kd_energy_sec", "t_array_growth_sec"]
macros = {}


def m(name, value):
    macros[name] = value


def e(name):
    return pd.read_csv(os.path.join(EXP, name))


def sci(x, digits=1):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    return f"{x / 10**exp:.{digits}f}\\times10^{{{exp}}}"


def write_table(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def pval(p):
    """Relation and value, used in the text as $p\\macro$."""
    return f"={p:.3f}" if p >= 0.001 else "<0.001"


# ---------------------------------------------------------------- main run
summ = pd.concat([e(f"stage{s}_batch_summary.csv") for s in "AB"]).set_index("dataset_name")
raw = pd.concat([e(f"stage{s}_raw_results.csv") for s in "AB"]).reset_index(drop=True)
ckpt = pd.concat([e(f"stage{s}_accuracy_checkpoints.csv") for s in "AB"]).reset_index(drop=True)
gt = pd.concat([e(f"stage{s}_gamma_ground_truth_check.csv") for s in "AB"]).reset_index(drop=True)
iid = raw[raw.stream_type == "iid"]
name_of = raw.drop_duplicates("dataset").set_index("dataset").dataset_name

# datasets table
rows = []
for i, ds in enumerate(ORDER, 1):
    df = pd.read_csv(os.path.join(DATA, f"{i}.csv"))
    rows.append(f"D{i} & {LONG.get(ds, ds)} & {len(df)} & {df.shape[1] - 1} & {df.iloc[:, -1].nunique()} \\\\")
write_table("tab_datasets.tex", rows)
_arr = pd.read_csv(os.path.join(DATA, "8.csv")).iloc[:, -1].value_counts()
m("arrSmall", f"{(_arr < 5).sum()}"); m("arrMinClass", f"{_arr.min()}")
m("nMin", f"{int(summ.n_total.min())}")
m("nMax", f"{int(summ.n_total.max()):,}".replace(",", "{,}"))

# exactness
gt["dataset_name"] = gt.dataset.map(name_of)
rows = []
for i, ds in enumerate(ORDER, 1):
    g = gt[gt.dataset_name == ds]
    rows.append(f"D{i} & {LONG.get(ds, ds)} & {len(g)} & ${sci(g.abs_diff.max())}$ \\\\")
write_table("tab_exactness.tex", rows)
m("gtCheckpoints", f"{len(gt)}")
m("gtMaxDiff", f"{sci(gt.abs_diff.max(), 2)}")
m("gtMaxDiffIid", f"{sci(gt[gt.stream_type == 'iid'].abs_diff.max())}")
m("gtMaxDiffSorted", f"{sci(gt[gt.stream_type == 'class_sorted'].abs_diff.max())}")
assert gt.delta_max_abs_diff.max() == 0 and gt.gsize_mismatch.max() == 0
assert gt.delta_inf_mismatch.max() == 0 and gt.cache_vs_fresh_max_abs_diff.max() == 0

# speedup
t_stream = iid.groupby("dataset_name")[STEP_COLS].sum().sum(axis=1)
# Timings: taken from the clean re-measurement (timing_clean.py, idle machine,
# stream median of 5) when present; its reducts must equal those of the main run.
CLEAN = os.path.exists(os.path.join(EXP, "timing_clean_mhmg.csv"))
if CLEAN:
    tc = e("timing_clean_mhmg.csv").set_index("dataset_name")
    assert set(tc.index) == set(summ.index), "timing_clean_mhmg.csv incomplete"
    for d in tc.index:
        assert tc.loc[d, "reduct_init_features"] == summ.loc[d, "reduct_init_features"], d
        assert tc.loc[d, "reduct_full_features"] == summ.loc[d, "reduct_full_features"], d
    summ = summ.copy()
    summ["time_batch_full_sec"] = tc.time_batch_full_sec
    summ["time_state_init_sec"] = tc.time_state_init_sec
    ok = tc.time_batch_init_sec.notna()
    summ.loc[ok[ok].index, "time_batch_init_sec"] = tc.time_batch_init_sec[ok]
    t_stream = tc.time_stream_sec_median.reindex(t_stream.index)
    # Stream and state-init times re-measured in a process without batch search
    # (timing_clean.py --stream-only); see the note in timing_clean.py.
    ts = e("timing_stream_mhmg.csv").set_index("dataset_name")
    assert set(ts.index) == set(summ.index), "timing_stream_mhmg.csv incomplete"
    assert (ts.reduct_init_features == tc.reduct_init_features.reindex(ts.index)).all()
    summ["time_state_init_sec"] = ts.time_state_init_sec
    t_stream = ts.time_stream_sec_median.reindex(t_stream.index)
else:
    print("WARNING: timing_clean_mhmg.csv missing; timings from the main run")
m("timingSource", "clean" if CLEAN else "main")
sp = pd.DataFrame({"n": summ.n_total, "p": summ.n_features, "full": summ.time_batch_full_sec,
                   "stream": t_stream, "init": summ.time_state_init_sec})
sp["speedup"] = sp.full / sp.stream
# Batch evaluation of gamma(B_init) on all n objects (timing_clean.py --eval-only)
# against the mean time of one update: the per-insertion cost of keeping gamma current.
tev = e("timing_eval_mhmg.csv").set_index("dataset_name")
assert set(tev.index) == set(sp.index), "timing_eval_mhmg.csv incomplete"
sp["eval"] = tev.time_eval_sec_median
sp["m"] = sp.n - (sp.n * 0.8).astype(int)
sp["evratio"] = sp["eval"] / (sp.stream / sp.m)
sp.to_csv(os.path.join(EXP, "speedup_final.csv"), index_label="dataset_name")  # read by generate_figures.py
rows = []
for ds in sp.sort_values("n").index:
    r = sp.loc[ds]
    rows.append(f"{ds} & {int(r.n)} & {int(r.p)} & {r.full:.2f} & {r['eval']:.4f} & {r.stream:.3f} & "
                f"{r.evratio:.0f}$\\times$ & {r.speedup:.0f}$\\times$ \\\\")
write_table("tab_speedup.tex", rows)
m("evrMin", f"{sp.evratio.min():.0f}"); m("evrMax", f"{sp.evratio.max():.0f}")
m("evrMinDs", sp.evratio.idxmin()); m("evrMaxDs", sp.evratio.idxmax())
# independent batch evaluation (attributes in sorted order, all distances recomputed)
# vs the maintained gamma at the end of the i.i.d. stream
_gl = gt[gt.stream_type == "iid"].reset_index(drop=True)
_gl = _gl.loc[_gl.groupby("dataset").step.idxmax()].set_index("dataset")
_ind = (_gl.gamma_incremental - tev.reset_index().set_index("dataset").gamma_B_init_all).abs()
_ind.index = _ind.index.map(name_of)
tie = _ind[_ind > 1e-12].sort_values(ascending=False)
m("tieN", f"{len(tie)}"); m("tieOthers", f"{len(_ind) - len(tie)}"); m("tieRest", f"{sci(_ind[_ind <= 1e-12].max())}")
m("tieList", " and ".join(f"${sci(v)}$ on {d}" for d, v in tie.items()))
m("spdMin", f"{sp.speedup.min():.0f}")
m("spdMax", f"{sp.speedup.max():.0f}")
m("spdMinDs", sp.speedup.idxmin())
m("spdMaxDs", sp.speedup.idxmax())
m("streamLen", "20\\%")
share = iid.groupby("dataset_name")[STEP_COLS].sum()
share = share.div(share.sum(axis=1), axis=0) * 100
m("deltaShareMin", f"{share.t_delta_friend_sec.min():.0f}")
m("deltaShareMax", f"{share.t_delta_friend_sec.max():.0f}")
m("arrayShareMin", f"{share.t_array_growth_sec.min():.0f}")
m("arrayShareMax", f"{share.t_array_growth_sec.max():.0f}")
m("initOverStreamMax", f"{(sp.init / sp.stream).max():.1f}")

# staleness, seed 42
fin = ckpt[(ckpt.stream_type == "iid")]
last = fin.loc[fin.groupby(["dataset_name", "reduct_used"]).step.idxmax()].set_index(["dataset_name", "reduct_used"])
gap = pd.DataFrame({
    "g3": [100 * (last.loc[(d, "B_full")].acc_3nn_mean - last.loc[(d, "B_init")].acc_3nn_mean) for d in ORDER],
    "gs": [100 * (last.loc[(d, "B_full")].acc_svm_mean - last.loc[(d, "B_init")].acc_svm_mean) for d in ORDER],
    "jac": [summ.loc[d].jaccard_overlap_init_full for d in ORDER],
    "ratio": [int(summ.loc[d].n_total * 0.8) / summ.loc[d].n_features for d in ORDER]}, index=ORDER)
rows = [f"{d} & ${r.g3:+.2f}$ & ${r.gs:+.2f}$ & {r.jac:.3f} & {r.ratio:.2f} \\\\" for d, r in gap.iterrows()]
write_table("tab_accgap.tex", rows)
w3, ws = wilcoxon(gap.g3), wilcoxon(gap.gs)
m("stW3", f"{w3.statistic:.1f}"); m("stP3", pval(w3.pvalue))
m("stWs", f"{ws.statistic:.1f}"); m("stPs", pval(ws.pvalue))
m("stMean3", f"{gap.g3.mean():.2f}"); m("stMed3", f"{gap.g3.median():.2f}")
m("stMeans", f"{gap.gs.mean():.2f}"); m("stMeds", f"{gap.gs.median():.2f}")
m("stMaxAbs3", f"{gap.g3.abs().max():.2f}"); m("stMaxAbs3Ds", gap.g3.abs().idxmax())
m("stMaxAbss", f"{gap.gs.abs().max():.2f}"); m("stMaxAbssDs", gap.gs.abs().idxmax())
m("stPos3", f"{(gap.g3 > 0).sum()}"); m("stPoss", f"{(gap.gs > 0).sum()}")
m("stWithinThree", f"{((gap.g3.abs() <= 3) & (gap.gs.abs() <= 3)).sum()}")

# staleness over seeds (seed 42 from the main run + seeds in multi_seed_w*.csv)
ms = pd.concat([pd.read_csv(os.path.join(EXP, f)) for f in sorted(os.listdir(EXP))
                if f.startswith("multi_seed_w") and f.endswith(".csv")])
base = gap.drop(index="Isolet").reset_index().rename(columns={"index": "dataset_name"})
base = base.assign(seed=42, gap_3nn_pp=base.g3, gap_svm_pp=base.gs, jaccard=base.jac,
                   size_init=[int(summ.loc[d].reduct_init_size) for d in base.dataset_name])
ms = pd.concat([ms[["seed", "dataset_name", "gap_3nn_pp", "gap_svm_pp", "jaccard", "size_init"]],
                base[["seed", "dataset_name", "gap_3nn_pp", "gap_svm_pp", "jaccard", "size_init"]]])
deg = ms[ms.size_init <= 1]
m("msDeg", f"{len(deg)}")
m("msDegList", ", ".join(f"{r.dataset_name} (seed {int(r.seed)})" for _, r in deg.iterrows()) or "none")
nd_ = ms[ms.size_init > 1]
m("msNoDegMaxAbs", f"{nd_[['gap_3nn_pp', 'gap_svm_pp']].abs().max().max():.1f}")
topn = nd_.assign(a=nd_[["gap_3nn_pp", "gap_svm_pp"]].abs().max(axis=1)).sort_values("a").iloc[-1]
m("msNoDegMaxAbsDs", topn.dataset_name); m("msNoDegMaxAbsSeed", f"{int(topn.seed)}")
m("msNoDegMeanThree", f"{nd_.gap_3nn_pp.mean():.2f}"); m("msNoDegMeans", f"{nd_.gap_svm_pp.mean():.2f}")
if len(deg):
    m("msDegGapThree", f"{deg.gap_3nn_pp.max():.0f}"); m("msDegGaps", f"{deg.gap_svm_pp.max():.0f}")
n_seeds = ms.seed.nunique()
m("nSeeds", f"{n_seeds}")
agg = ms.groupby("dataset_name").agg(g3m=("gap_3nn_pp", "mean"), g3s=("gap_3nn_pp", "std"),
                                     gsm=("gap_svm_pp", "mean"), gss=("gap_svm_pp", "std"),
                                     g3max=("gap_3nn_pp", lambda x: x.abs().max()),
                                     gsmax=("gap_svm_pp", lambda x: x.abs().max()),
                                     jm=("jaccard", "mean"), n=("seed", "nunique"))
agg = agg.loc[[d for d in ORDER if d in agg.index]]
if agg.n.nunique() != 1 or agg.n.iloc[0] != n_seeds:
    print("WARNING: unequal number of seeds per dataset:", agg.n.to_dict())
def pm(mu, sd):
    return f"${mu:+.2f}\\pm{sd:.2f}$" if np.isfinite(sd) else f"${mu:+.2f}$"


rows = [f"{d} & {pm(r.g3m, r.g3s)} & {pm(r.gsm, r.gss)} & {r.g3max:.1f} & {r.gsmax:.1f} & {r.jm:.2f} \\\\"
        for d, r in agg.iterrows()]
write_table("tab_multiseed.tex", rows)
mw3, mws = wilcoxon(agg.g3m), wilcoxon(agg.gsm)
m("msP3", pval(mw3.pvalue)); m("msPs", pval(mws.pvalue))
m("msMean3", f"{agg.g3m.mean():.2f}"); m("msMeans", f"{agg.gsm.mean():.2f}")
m("msPos3", f"{(agg.g3m > 0).sum()}"); m("msPoss", f"{(agg.gsm > 0).sum()}")
# robustness of the average to the degenerate initial reduct(s)
aggnd = nd_.groupby("dataset_name")[["gap_3nn_pp", "gap_svm_pp"]].mean()
m("msMeanNoDegThree", f"{aggnd.gap_3nn_pp.mean():.2f}"); m("msMeanNoDegs", f"{aggnd.gap_svm_pp.mean():.2f}")
m("msMedThree", f"{agg.g3m.median():.2f}"); m("msMeds", f"{agg.gsm.median():.2f}")
m("msRunPosThree", f"{(ms.gap_3nn_pp > 0).sum()}"); m("msRunPoss", f"{(ms.gap_svm_pp > 0).sum()}")
m("msRunNegThree", f"{(ms.gap_3nn_pp < 0).sum()}"); m("msRunNegs", f"{(ms.gap_svm_pp < 0).sum()}")
m("msNds", f"{len(agg)}")
big = (ms[["gap_3nn_pp", "gap_svm_pp"]] > 5).any(axis=1)   # a loss (B_init worse) above 5 pp
m("msBigFrac", f"{100 * big.mean():.0f}")
m("msBigN", f"{big.sum()}"); m("msRuns", f"{len(ms)}")
m("msMaxAbs", f"{ms[['gap_3nn_pp', 'gap_svm_pp']].abs().max().max():.1f}")
top = ms.assign(a=ms[["gap_3nn_pp", "gap_svm_pp"]].abs().max(axis=1)).sort_values("a").iloc[-1]
m("msMaxAbsDs", top.dataset_name); m("msMaxAbsSeed", f"{int(top.seed)}")

# exploratory correlates on the seed-averaged gaps
nd = pd.Series({d: gap.loc[d, "ratio"] for d in agg.index})
corr = []
for lab, x in [("$n/p$", nd), ("Jaccard($B_{init},B_{full}$)", agg.jm)]:
    r3, p3 = spearmanr(x, agg.g3m.abs()); rs, ps = spearmanr(x, agg.gsm.abs())
    corr.append((lab, r3, p3, rs, ps))
write_table("tab_corr.tex", [f"{l} & $\\rho={a:+.3f},\\ p={b:.3f}$ & $\\rho={c:+.3f},\\ p={d:.3f}$ \\\\"
                             for l, a, b, c, d in corr])
m("corrMinP", f"{min(min(c[2], c[4]) for c in corr):.3f}")
m("corrAnyBonf", "yes" if min(min(c[2], c[4]) for c in corr) < 0.05 / 4 else "no")

# rounding confound
rc = e("rounding_confound_jaccard.csv")
for ds in ["Wine", "Musk", "Urban"]:
    r = rc[rc.dataset_name == ds].set_index("precision").jaccard_init_full
    key = {"Wine": "Wine", "Musk": "Musk", "Urban": "Urban"}[ds]
    m(f"rc{key}Two", f"{r.loc[2]:.3f}"); m(f"rc{key}Exact", f"{r.loc[r.index.max()]:.3f}")
    s = rc[rc.dataset_name == ds].set_index("precision")
    m(f"rc{key}SizeTwo", f"{int(s.loc[2].reduct_init_size)}/{int(s.loc[2].reduct_full_size)}")
    m(f"rc{key}SizeExact", f"{int(s.loc[s.index.max()].reduct_init_size)}/{int(s.loc[s.index.max()].reduct_full_size)}")

# trajectory
tr = e("e0_measurement2_acc_gap_trajectory.csv")
m("trDatasets", ", ".join(tr.dataset_name.drop_duplicates()))
for ds in tr.dataset_name.unique():
    t = tr[tr.dataset_name == ds].sort_values("checkpoint_idx")
    k = ds.replace(" ", "")
    m(f"tr{k}Three", "\\!\\to\\!".join(f"{v:.1f}" for v in t.gap_3nn))
    m(f"tr{k}Svm", "\\!\\to\\!".join(f"{v:.1f}" for v in t.gap_svm))

# 50/50 protocol
pr = e("e0_measurement9_li2026_protocol.csv")
fr = pr[pr["round"] == pr["round"].max()].sort_values("jaccard_Binit_vs_Bround")
rows = [f"{r.dataset} & {r.jaccard_Binit_vs_Bround:.3f} & {r.speedup:.0f}$\\times$ & ${r.gap3_pp:+.2f}$ & ${r.gapsvm_pp:+.2f}$ \\\\"
        for _, r in fr.iterrows()]
write_table("tab_li2026.tex", rows)
m("prRounds", f"{len(pr)}"); m("prGtMax", f"{sci(pr.gt_diff.max())}")
m("prSpdMin", f"{pr.speedup.min():.0f}"); m("prSpdMax", f"{pr.speedup.max():.0f}")
m("prSpdLastMin", f"{fr.speedup.min():.0f}"); m("prSpdLastMax", f"{fr.speedup.max():.0f}")
m("prKernelMin", f"{pr.speedup_kernel.min():.0f}"); m("prKernelMax", f"{pr.speedup_kernel.max():.0f}")
m("prAssemblyShare", f"{100 * ((pr.t_incremental_total_s - pr.t_incremental_s) / pr.t_incremental_total_s).median():.0f}")
for ds in ["Sonar", "Musk", "Letter"]:
    j = pr[pr.dataset == ds].sort_values("round").jaccard_Binit_vs_Bround
    m(f"pr{ds}JacZero", f"{(j == 0).sum()}")
    m(f"pr{ds}Jac", ", ".join(f"{v:.2f}" for v in j))
pmax = fr.set_index("dataset")[["gap3_pp", "gapsvm_pp"]].abs().max(axis=1)
m("prMaxGapDs", pmax.idxmax()); m("prMaxGap", f"{pmax.max():.2f}")
# total cost 50/50 vs 80/20 (MHMG-RS): base search + state init + incremental catch-up.
# The 50% base search time is printed in the run log, not stored in the CSV.
import re
with open(os.path.join(EXP, "run_v3_all.log")) as f:
    seg9 = f.read().split("e0_measurement9_li2026_protocol_pilot.py")[1]
base_search = {mm.group(1): float(mm.group(2)) for mm in
               re.finditer(r"=== (\w+): n=.*?\n  B_init .*?, ([0-9.]+)s tim kiem", seg9, re.S)}
t5050 = pr.groupby("dataset").apply(
    lambda g: base_search[g.name] + g.t_state_init_s.iloc[0] + g.t_incremental_total_s.sum())
t8020 = (summ.time_batch_init_sec + summ.time_state_init_sec + t_stream).loc[t5050.index]
ratio = t5050 / t8020
m("prCostMin", f"{ratio.min():.2f}"); m("prCostMax", f"{ratio.max():.2f}")
m("prCostCheaper", f"{(ratio < 1).sum()}/{len(ratio)}")
lt = re.search(r"=== Letter: .*?\n  B_init \(tren 50% base\): \[([0-9, ]*)\]  \(gamma=([0-9.]+)", seg9)
m("prLetterInitSize", f"{len(lt.group(1).split(','))}")
m("prLetterInitGamma", f"{float(lt.group(2)):.2f}")
m("prLetterFullSize", f"{int(fr.set_index('dataset').loc['Letter'].B_round_size)}")

# zero margin
zm = iid.groupby("dataset_name").agg(steps=("step", "count"),
                                      znew=("n_zero_delta_new", "sum"), zex=("n_zero_delta_existing", "sum"))
zm["zsteps"] = iid.assign(z=(iid.n_zero_delta_new + iid.n_zero_delta_existing) > 0).groupby("dataset_name").z.sum()
hit = zm[zm.zsteps > 0]
m("zmDatasets", f"{len(hit)}")
m("zmList", "; ".join(f"{d}: {int(r.zsteps)}/{int(r.steps)} insertions ({100 * r.zsteps / r.steps:.1f}\\%), "
                      f"{int(r.znew + r.zex)} affected objects" for d, r in hit.iterrows()))

# shrink fractions
iid = iid.assign(sf=iid.n_shrink / iid.n_active_before, gf=iid.n_gain / iid.n_active_before)
msf = iid.groupby("dataset_name").sf.mean() * 100
m("sfMeanMin", f"{msf.min():.3f}"); m("sfMeanMinDs", msf.idxmin())
m("sfMeanMax", f"{msf.max():.2f}"); m("sfMeanMaxDs", msf.idxmax())
allst = raw.assign(sf=raw.n_shrink / raw.n_active_before, gf=raw.n_gain / raw.n_active_before)
m("maxFrac", f"{100 * allst[['sf', 'gf']].max().max():.1f}")

# cold start
cs = e("e0_measurement6_gradual_new_classes.csv")
ev = cs[cs.is_new_class_event.astype(bool)]
m("csEvents", ", ".join(f"{100 * v:.1f}" for v in ev.shrink_frac))
m("csMean", f"{100 * ev.shrink_frac.mean():.1f}")
with open(os.path.join(EXP, "run_v3_all.log")) as f:
    log = f.read()
seg = log.split("e0_measurement6_gradual_new_classes.py")[1]
m("csGt", sci(float(seg.split("Ground-truth max abs diff toan luong:")[1].split()[0])))

# CE re-evaluation cost
ce = e("ce_reevaluation_cost.csv").set_index("dataset_name")
m("ceLetter", f"{ce.loc['Letter'].ce_reeval_time_min_sec:.2f}")
m("ceSmallMax", f"{ce.drop(index='Letter').ce_reeval_time_min_sec.max():.3f}")

# baselines (full precision)
cmp_ = e("comparison_accuracy_size.csv").set_index("dataset_name")
cmp_ = cmp_.loc[[d for d in ORDER if d in cmp_.index]]
sizes = ["ours_size", "xu_size", "li_size", "deng_size"]
rows = []
for d, r in cmp_.iterrows():
    mn = r[sizes].min()
    rows.append(f"{d} & " + " & ".join((f"\\textbf{{{int(r[c])}}}" if r[c] == mn else f"{int(r[c])}") for c in sizes) + " \\\\")
means = cmp_[sizes].mean()
rows.append("\\midrule")
rows.append("Mean & " + " & ".join((f"\\textbf{{{v:.1f}}}" if v == means.min() else f"{v:.1f}") for v in means) + " \\\\")
write_table("tab_baseline_size.tex", rows)
rows = []
for d, r in cmp_.iterrows():
    cells = []
    for clf in ["3nn", "svm"]:
        cols = [f"{mm}_{clf}" for mm in ["ours", "xu", "li", "deng"]]
        mx = r[cols].round(1).max()
        cells += [(f"\\textbf{{{r[c]:.1f}}}" if round(r[c], 1) == mx else f"{r[c]:.1f}") for c in cols]
    rows.append(f"{d} & " + " & ".join(cells) + " \\\\")
write_table("tab_baseline_accuracy.tex", rows)
m("sizeOurs", f"{means.ours_size:.1f}"); m("sizeXu", f"{means.xu_size:.1f}")
m("sizeLi", f"{means.li_size:.1f}"); m("sizeDeng", f"{means.deng_size:.1f}")
m("sizeSmallest", f"{(cmp_.ours_size <= cmp_[sizes].min(axis=1)).sum()}")
m("nCmp", f"{len(cmp_)}")
smaller = cmp_[cmp_.ours_size > cmp_[sizes].min(axis=1)]
m("sizeNotSmallest", ", ".join(smaller.index))
NAMES = {"xu": "Xu2025b", "li": "Li2026", "deng": "Deng2026"}
parts = []
for d, r in smaller.iterrows():
    who = [k for k in NAMES if r[f"{k}_size"] < r.ours_size]
    diffs = [r[f"ours_{c}"] - r[f"{k}_{c}"] for k in who for c in ("3nn", "svm")]
    parts.append(f"{d} ({' and '.join(NAMES[k] for k in who)}: "
                 f"{'/'.join(str(int(r[f'{k}_size'])) for k in who)} vs.\\ {int(r.ours_size)} attributes, "
                 f"{min(diffs):.1f}--{max(diffs):.1f}\\,pp lower accuracy)")
m("sizeNotSmallestDetail", "; ".join(parts))
def holm(ps):
    """Holm-adjusted p-values (step-down, monotone), same order as the input."""
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


# Family for multiple testing: the three baselines under one classifier (Holm).
for clf, ck in [("3nn", "Three"), ("svm", "Svm")]:
    raw = []
    for mm, key in [("xu", "Xu"), ("li", "Li"), ("deng", "Deng")]:
        d = cmp_[f"ours_{clf}"] - cmp_[f"{mm}_{clf}"]
        m(f"acc{key}{ck}Diff", f"{d.mean():+.1f}")
        m(f"acc{key}{ck}Wins", f"{(d > 0).sum()}")
        raw.append(wilcoxon(cmp_[f"ours_{clf}"], cmp_[f"{mm}_{clf}"]).pvalue)
    adj = holm(np.array(raw))
    for (mm, key), pr, pa in zip([("xu", "Xu"), ("li", "Li"), ("deng", "Deng")], raw, adj):
        m(f"acc{key}{ck}P", pval(pr)); m(f"acc{key}{ck}Padj", pval(pa))
    m(f"accSig{ck}", ", ".join(NAMES[mm] for mm, pa in zip(["xu", "li", "deng"], adj) if pa < 0.05) or "none")

# cross-method Jaccard (ours vs Xu2025b)
xu = e("baseline_xu2025b_results.csv").set_index("dataset_name")
cj = pd.DataFrame({"ours": summ.jaccard_overlap_init_full, "xu": xu.jaccard_init_full}).dropna()
m("cjOurs", f"{cj.ours.mean():.3f}"); m("cjXu", f"{cj.xu.mean():.3f}")
m("cjP", pval(wilcoxon(cj.ours, cj.xu).pvalue))

# total pipeline: batch search on 80% + state initialization + stream updates
xr = e("timing_clean_xu.csv") if os.path.exists(os.path.join(EXP, "timing_clean_xu.csv")) else None
xb = e("baseline_xu2025b_results.csv").set_index("dataset_name")
if xr is not None:
    xr = xr.set_index("dataset_name")
    for d in xr.index:
        assert set(str(xr.loc[d, "reduct_init_features"]).split("|")) == \
            set(str(xb.loc[d, "reduct_init_features"]).split("|")), d
    xt = xr.time_batch_init_sec + xr.time_state_init_sec + xr.time_stream_sec_median
else:
    print("WARNING: timing_clean_xu.csv missing; Xu2025b timings from the main run")
    xt = xb.time_batch_init_sec + xb.time_state_init_sec + xb.stream_time_sec
ot = summ.time_batch_init_sec + summ.time_state_init_sec + t_stream
tt = pd.DataFrame({"n": summ.n_total, "p": summ.n_features, "ours_total_sec": ot,
                   "xu_total_sec": xt}).dropna().sort_values("p")
tt["xu_over_ours"] = tt.xu_total_sec / tt.ours_total_sec
tt.to_csv(os.path.join(EXP, "total_reduction_time_comparison_final.csv"))
rows = []
for d, r in tt.iterrows():
    o, x = r.ours_total_sec, r.xu_total_sec
    rows.append(f"{d} & {int(r.n)} & {int(r.p)} & " + (f"\\textbf{{{o:.3f}}} & {x:.3f}" if o < x else f"{o:.3f} & \\textbf{{{x:.3f}}}") + " \\\\")
write_table("tab_total_time.tex", rows)
m("ttOursFaster", f"{(tt.xu_over_ours > 1).sum()}")
m("ttOursFasterList", ", ".join(tt[tt.xu_over_ours > 1].index))
hi = tt[tt.p >= 147]; lo = tt[tt.p <= 34]
m("ttHiMin", f"{hi.xu_over_ours.min():.1f}"); m("ttHiMax", f"{hi.xu_over_ours.max():.1f}")
m("ttHiWins", f"{(hi.xu_over_ours > 1).sum()}/{len(hi)}")
m("ttLoWins", f"{(lo.xu_over_ours < 1).sum()}/{len(lo)}")
m("ttLoMin", f"{(1 / lo.xu_over_ours).min():.1f}"); m("ttLoMax", f"{(1 / lo.xu_over_ours).max():.1f}")

# Deng2026 t sweep
sw = e("deng2026_t_sweep.csv").groupby("t")[["size", "acc_3nn", "acc_svm"]].mean()
write_table("tab_deng_sweep.tex", [f"{t} & {r['size']:.1f} & {r.acc_3nn:.1f} & {r.acc_svm:.1f} \\\\" for t, r in sw.iterrows()])
m("dengOneSize", f"{sw.loc[1, 'size']:.1f}"); m("dengOneThree", f"{sw.loc[1, 'acc_3nn']:.1f}")
m("dengOneSvm", f"{sw.loc[1, 'acc_svm']:.1f}")
m("dengOneRatio", f"{sw.loc[1, 'size'] / cmp_.ours_size.mean():.0f}")
m("oursMeanThree", f"{cmp_.ours_3nn.mean():.1f}"); m("oursMeanSvm", f"{cmp_.ours_svm.mean():.1f}")

# batch-size sweep and batch closed-form check
bs = pd.read_csv(os.path.join(THE, "e0_batch_size_sweep.csv"))
for ds in ["Wine", "Urban", "Spambase"]:
    s = bs[bs.dataset == ds].sort_values("K")
    m(f"bs{ds}N", f"{int(s.n_init.iloc[0])}")
    m(f"bs{ds}Min", f"{s.speedup.min():.2f}"); m(f"bs{ds}Max", f"{s.speedup.max():.2f}")
    m(f"bs{ds}First", f"{s.speedup.iloc[0]:.2f}"); m(f"bs{ds}Last", f"{s.speedup.iloc[-1]:.2f}")
    m(f"bs{ds}KFirst", f"{int(s.K.iloc[0])}"); m(f"bs{ds}KLast", f"{int(s.K.iloc[-1])}")
bc = pd.read_csv(os.path.join(THE, "e0_batch_insertion_check.csv"))
bc = bc[bc.check != "timing"]
m("bcGamma", f"{sci(bc.diff_gamma.max())}"); m("bcE", f"{sci(bc.diff_E.max())}")
m("bcKd", f"{sci(bc.diff_Kd.max())}")
assert bc.max_diff_delta.max() == 0 and bc.max_diff_gsize.max() == 0

# earlier run (archived; split differed because one object per dataset was missing)
arch = os.path.join(EXP, "_archive_v2_pre_header_fix")
oa = pd.read_csv(os.path.join(arch, "stageA_accuracy_checkpoints.csv"))
oa = oa[oa.stream_type == "iid"]
ol = oa.loc[oa.groupby(["dataset_name", "reduct_used"]).step.idxmax()].set_index(["dataset_name", "reduct_used"])
og = {d: max(abs(100 * (ol.loc[(d, "B_full")].acc_3nn_mean - ol.loc[(d, "B_init")].acc_3nn_mean)),
             abs(100 * (ol.loc[(d, "B_full")].acc_svm_mean - ol.loc[(d, "B_init")].acc_svm_mean)))
      for d in oa.dataset_name.unique()}
top2 = sorted(og.items(), key=lambda kv: -kv[1])[:2]
m("oldGapA", f"{top2[1][1]:.0f}"); m("oldGapB", f"{top2[0][1]:.0f}")
m("oldGapDs", f"{top2[1][0]} and {top2[0][0]}")
# Xu2025b initial phase on Isolet: composite entropy over attribute pairs, O(p^2 n^2)
ni, pi = int(summ.loc["Isolet"].n_total), int(summ.loc["Isolet"].n_features)
m("xuIsoletOps", f"{sci(pi ** 2 * ni ** 2)}")

# batch reducer: Gram-identity distances vs direct differences
rd = e("reducer_distance_check.csv").set_index("dataset_name")
m("rdMax", f"{sci(rd.abs_diff.max())}"); m("rdMaxDs", rd.abs_diff.idxmax())
m("rdExact", f"{(rd.abs_diff == 0).sum()}/{len(rd)}")
m("rdDist", f"{sci(rd.max_abs_distance_diff.max(), 0)}")

# re-fit signal: decrease of the maintained gamma(B_init) (rounded to 2 decimals as in the reducer)
ps = e("pilot_trigger_signal.csv").merge(ms, on=["seed", "dataset_name"], how="inner", validate="one_to_one")
assert len(ps) == len(ms), (len(ps), len(ms))
fire = ps.g_end.round(2) < ps.g_init.round(2)
bigp = (ps[["gap_3nn_pp", "gap_svm_pp"]] > 5).any(axis=1)
quiet = ps[~fire]
m("trgFire", f"{fire.sum()}"); m("trgRuns", f"{len(ps)}")
nc, nb = (fire & bigp).sum(), bigp.sum()
m("trgBig", f"all {nb}" if nc == nb else f"{nc} of the {nb}")
m("trgQuietMedThree", f"{quiet.gap_3nn_pp.median():.2f}"); m("trgQuietMeds", f"{quiet.gap_svm_pp.median():.2f}")
m("trgQuietMaxThree", f"{quiet.gap_3nn_pp.max():.1f}"); m("trgQuietMaxs", f"{quiet.gap_svm_pp.max():.1f}")
m("trgFireMedThree", f"{ps[fire].gap_3nn_pp.median():.2f}"); m("trgFireMeds", f"{ps[fire].gap_svm_pp.median():.2f}")

# hardware and memory
m("memIsolet", f"{summ.loc['Isolet'].n_total ** 2 * 8 / 1e6:.0f}")
m("memLetter", f"{summ.loc['Letter'].n_total ** 2 * 8 / 1e9:.1f}")

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write("% generated by make_results_tex.py -- do not edit\n")
    for k, v in macros.items():
        k = k.replace("3", "Three")
        assert k.isalpha(), k
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros; tables:", sorted(x for x in os.listdir(OUT) if x.startswith("tab_")))
