"""
Figures of the main experiment (style of
IJAR-MHMG-Published/IJAR-Code/generate_other_plots.py). Figure 2 shows the accuracy
gap per dataset as a bar chart. All values are read from stageA/B_*.csv.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'figs')
os.makedirs(OUTDIR, exist_ok=True)

# Font matched to the manuscript's own LaTeX rendering (article class =
# Computer Modern): DejaVu Serif for body text + 'cm' mathtext for math,
# no external LaTeX/dvipng dependency needed. Full box + inward ticks,
# no gridlines -- matches the journal-figure convention used by the
# reference papers (e.g. Xu2025b Figs 4-5).
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.unicode_minus': True,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})


def style_axes(ax):
    ax.tick_params(direction='in', top=True, right=True, which='both')
    for spine in ax.spines.values():
        spine.set_visible(True)

bs = pd.concat([pd.read_csv(os.path.join(HERE, 'stageA_batch_summary.csv')),
                pd.read_csv(os.path.join(HERE, 'stageB_batch_summary.csv'))])
ckpt = pd.concat([pd.read_csv(os.path.join(HERE, 'stageA_accuracy_checkpoints.csv')),
                   pd.read_csv(os.path.join(HERE, 'stageB_accuracy_checkpoints.csv'))])
raw = pd.concat([pd.read_csv(os.path.join(HERE, 'stageA_raw_results.csv')),
                  pd.read_csv(os.path.join(HERE, 'stageB_raw_results.csv'))])

final = ckpt[ckpt.frac_of_stream == 1.0]
piv = final.pivot_table(index=['dataset', 'dataset_name'], columns='reduct_used',
                          values=['acc_3nn_mean', 'acc_svm_mean']).reset_index()
piv.columns = ['dataset', 'dataset_name'] + ['_'.join(c) for c in piv.columns[2:]]
piv = piv.merge(bs[['dataset', 'n_total', 'n_features', 'time_batch_full_sec',
                     'jaccard_overlap_init_full']], on='dataset')
piv['n_over_d'] = piv.n_total * 0.8 / piv.n_features

iid = raw[raw.stream_type == 'iid']
stream_time = iid.groupby('dataset')[
    ['t_delta_friend_sec', 't_shrink_cascade_sec', 't_kd_energy_sec', 't_array_growth_sec']
].sum().sum(axis=1)
piv = piv.merge(stream_time.rename('total_stream_sec').reset_index(), on='dataset')
# Speedup as reported in the manuscript (clean timings), written by ../paper/make_results_tex.py
spd = pd.read_csv(os.path.join(HERE, 'speedup_final.csv'))
piv = piv.merge(spd[['dataset_name', 'speedup']], on='dataset_name')
piv['gap_3nn'] = (piv.acc_3nn_mean_B_full - piv.acc_3nn_mean_B_init) * 100
piv['gap_svm'] = (piv.acc_svm_mean_B_full - piv.acc_svm_mean_B_init) * 100

order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12']
piv['dataset'] = pd.Categorical(piv['dataset'], categories=order, ordered=True)
piv = piv.sort_values('dataset')

name_map = {'D1': 'Wine', 'D2': 'WDBC', 'D3': 'Iono.', 'D4': 'Derm.', 'D5': 'Sonar',
            'D6': 'Urban', 'D7': 'Musk', 'D8': 'Arrhy.', 'D9': 'Spam', 'D10': 'Park.',
            'D11': 'Isolet', 'D12': 'Letter'}
xlabels = [f"{d}\n({name_map[d]})" for d in order]
x = np.arange(12)

# ── Figure 1: speedup (log scale), bar chart ──
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x, piv['speedup'].values, color='#2ca02c', edgecolor='black', linewidth=0.5)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel('Speedup over batch search (log scale)', fontsize=11)
style_axes(ax)
for xi, v in zip(x, piv['speedup'].values):
    ax.text(xi, v * 1.15, f'{v:.0f}$\\times$', ha='center', va='bottom', fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_speedup_log.pdf'))
plt.close(fig)

# ── Figure 2: gap (B_full - B_init) per dataset, grouped bar, 3-NN + SVM ──
fig, ax = plt.subplots(figsize=(9, 4.5))
w = 0.35
ax.bar(x - w / 2, piv['gap_3nn'].values, width=w, label='3-NN', color='#1f77b4', edgecolor='black', linewidth=0.4)
ax.bar(x + w / 2, piv['gap_svm'].values, width=w, label='SVM-RBF', color='#ff7f0e', edgecolor='black', linewidth=0.4)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel('Accuracy gap, B\\_full $-$ B\\_init (pp)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis='y', linestyle=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_gap_per_dataset.pdf'))
plt.close(fig)

# ── Figure 3: Jaccard(B_init,B_full) vs |gap|, exploratory ──
fig, ax = plt.subplots(figsize=(6, 4.5))
abs_gap3 = piv['gap_3nn'].abs().values
abs_gapsvm = piv['gap_svm'].abs().values
jac = piv['jaccard_overlap_init_full'].values
ax.scatter(jac, abs_gap3, color='#1f77b4', s=45, zorder=3, label='3-NN')
ax.scatter(jac, abs_gapsvm, color='#ff7f0e', s=45, marker='^', zorder=3, label='SVM-RBF')
for _, row in piv.iterrows():
    ax.annotate(name_map[row['dataset']], (row['jaccard_overlap_init_full'], abs(row['gap_3nn'])),
                fontsize=7, va='bottom', xytext=(2, 2), textcoords='offset points')
ax.set_xlabel('Jaccard(B\\_init, B\\_full)', fontsize=11)
ax.set_ylabel('$|$accuracy gap$|$ (pp)', fontsize=11)
rho3, p3 = stats.spearmanr(jac, abs_gap3)
rhosvm, psvm = stats.spearmanr(jac, abs_gapsvm)
# rho/p inprinted via print below; caption belongs in LaTeX, not in the PDF art.
ax.legend(fontsize=9)
print(f'fig_gap_vs_jaccard Spearman: 3-NN rho={rho3:.2f} p={p3:.3f}; '
      f'SVM rho={rhosvm:.2f} p={psvm:.3f}')
ax.grid(True, linestyle=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_gap_vs_jaccard.pdf'))
plt.close(fig)

# remove the earlier n/d scatter figure, which is no longer produced
old_fig = os.path.join(OUTDIR, 'fig_accgap_vs_nd.pdf')
if os.path.exists(old_fig):
    os.remove(old_fig)
    print(f'Da xoa hinh cu (khong con hop le): {old_fig}')

print('Da sinh: fig_speedup_log.pdf, fig_gap_per_dataset.pdf, fig_gap_vs_jaccard.pdf trong', OUTDIR)
pd.set_option('display.width', 160)
print(piv[['dataset', 'speedup', 'gap_3nn', 'gap_svm', 'jaccard_overlap_init_full', 'n_over_d']].round(3).to_string(index=False))
