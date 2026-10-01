"""
Synthesis figures (DejaVu Serif + mathtext 'cm' to match the manuscript fonts;
full box, inward ticks, no grid -- as in generate_analysis_figures.py).

1. fig_jaccard_vs_gap.pdf      -- Jaccard vs |gap| for both methods
2. fig_total_time_vs_d.pdf     -- ratio of total times (Xu/MHMG) vs p (log-log)
3. fig_batch_speedup_vs_k.pdf  -- batch vs sequential speedup vs k (Wine/Urban/Spambase)

All values are read from the source CSV files.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, '..', 'experiments')
THEORY_DIR = os.path.join(HERE, '..', 'theory')
OUTDIR = os.path.join(HERE, 'figs')
os.makedirs(OUTDIR, exist_ok=True)

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUE, ORANGE = CAT[0], CAT[1]
TEXT_PRIMARY, TEXT_SECONDARY = '#0b0b0b', '#52514e'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': TEXT_PRIMARY, 'axes.linewidth': 0.8,
    'text.color': TEXT_PRIMARY, 'axes.labelcolor': TEXT_PRIMARY,
    'xtick.color': TEXT_PRIMARY, 'ytick.color': TEXT_PRIMARY,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.unicode_minus': True,
})


def style_axes(ax, log_x=False, log_y=False):
    ax.tick_params(direction='in', top=True, right=True, which='both')
    for spine in ax.spines.values():
        spine.set_visible(True)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')


# ---------------------------------------------------------------------------
# 1. Jaccard vs |gap|, ca 2 phuong phap
# ---------------------------------------------------------------------------
bs = pd.concat([pd.read_csv(os.path.join(EXP_DIR, 'stageA_batch_summary.csv')),
                pd.read_csv(os.path.join(EXP_DIR, 'stageB_batch_summary.csv'))])
bs = bs[bs.dataset != 'D11'][['dataset', 'dataset_name', 'jaccard_overlap_init_full']]

ckpt = pd.concat([pd.read_csv(os.path.join(EXP_DIR, 'stageA_accuracy_checkpoints.csv')),
                   pd.read_csv(os.path.join(EXP_DIR, 'stageB_accuracy_checkpoints.csv'))])
final = ckpt[ckpt.frac_of_stream == 1.0]
piv = final.pivot_table(index=['dataset', 'dataset_name'], columns='reduct_used',
                          values=['acc_3nn_mean', 'acc_svm_mean']).reset_index()
piv.columns = ['dataset', 'dataset_name'] + ['_'.join(c) for c in piv.columns[2:]]
piv['gap3'] = (piv.acc_3nn_mean_B_full - piv.acc_3nn_mean_B_init) * 100
piv['gapsvm'] = (piv.acc_svm_mean_B_full - piv.acc_svm_mean_B_init) * 100

mhmg = bs.merge(piv[['dataset', 'dataset_name', 'gap3', 'gapsvm']], on=['dataset', 'dataset_name'])
mhmg = mhmg.rename(columns={'jaccard_overlap_init_full': 'jaccard'})
mhmg['method'] = 'MHMG-RS'

xu = pd.read_csv(os.path.join(EXP_DIR, 'baseline_xu2025b_results.csv'))[
    ['dataset', 'dataset_name', 'jaccard_init_full', 'gap_3nn', 'gap_svm']
].rename(columns={'jaccard_init_full': 'jaccard', 'gap_3nn': 'gap3', 'gap_svm': 'gapsvm'})
xu['method'] = 'Xu2025b'

both = pd.concat([mhmg, xu], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, metric, panel in zip(axes, ['gap3', 'gapsvm'], ['(a) 3-NN', '(b) SVM-RBF']):
    for i, (method, color, marker) in enumerate([('MHMG-RS', BLUE, 'o'), ('Xu2025b', ORANGE, '^')]):
        sub = both[both.method == method]
        ax.scatter(sub.jaccard, sub[metric].abs(), color=color, marker=marker, s=60,
                   edgecolor='white', linewidth=0.8, label=method, zorder=4, alpha=0.9)
    # highlight Musk for both methods
    musk = both[(both.dataset_name == 'Musk')]
    for _, row in musk.iterrows():
        ax.annotate('Musk', (row.jaccard, abs(row[metric])), fontsize=8, color=TEXT_SECONDARY,
                    xytext=(6, 6), textcoords='offset points')
    ax.set_xlabel('Jaccard($B_{init}, B_{full}$)')
    ax.set_title(panel, fontsize=11)  # nhan panel; caption o LaTeX
    ax.set_xlim(-0.03, 1.05)
    style_axes(ax)
axes[0].set_ylabel('|Accuracy gap| (pp)')
axes[0].legend(frameon=False, fontsize=9, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_jaccard_vs_gap.pdf'))
plt.close(fig)
print('fig_jaccard_vs_gap.pdf OK')

# ---------------------------------------------------------------------------
# 2. Ratio of total times (Xu/MHMG) vs p
# ---------------------------------------------------------------------------
# written by make_results_tex.py (clean timings, as reported in the manuscript)
tt = pd.read_csv(os.path.join(EXP_DIR, 'total_reduction_time_comparison_final.csv'))
tt = tt.rename(columns={'p': 'n_features'})
tt['ratio'] = tt.xu_total_sec / tt.ours_total_sec  # >1: MHMG faster; <1: Xu faster

fig, ax = plt.subplots(figsize=(6.5, 4.5))
mhmg_wins = tt.ratio > 1
ax.scatter(tt.n_features[mhmg_wins], tt.ratio[mhmg_wins], color=BLUE, s=70,
           edgecolor='white', linewidth=0.8, zorder=4, label='Ours faster')
ax.scatter(tt.n_features[~mhmg_wins], tt.ratio[~mhmg_wins], color=ORANGE, s=70,
           marker='^', edgecolor='white', linewidth=0.8, zorder=4, label='Xu2025b faster')
# Derm/Iono have p~34 and similar ratios -> labels offset to avoid overlap
label_offsets = {'Derm': (5, 8), 'Iono': (5, -10)}
for _, row in tt.iterrows():
    dx, dy = label_offsets.get(row.dataset_name, (5, 4))
    ax.annotate(row.dataset_name, (row.n_features, row.ratio), fontsize=7.5,
                color=TEXT_SECONDARY, xytext=(dx, dy), textcoords='offset points')
ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle='--', zorder=2)
ax.set_xlabel('Original attribute count $d$ (log scale)')
ax.set_ylabel('Total-time ratio, Xu2025b / Ours (log scale)')
ax.legend(frameon=False, fontsize=9, loc='upper left')
style_axes(ax, log_x=True, log_y=True)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_total_time_vs_d.pdf'))
plt.close(fig)
print('fig_total_time_vs_d.pdf OK')

# ---------------------------------------------------------------------------
# 3. Speedup vs k (batch vs sequential), 3 datasets
# ---------------------------------------------------------------------------
sweep = pd.read_csv(os.path.join(THEORY_DIR, 'e0_batch_size_sweep.csv'))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i, name in enumerate(['Wine', 'Urban', 'Spambase']):
    sub = sweep[sweep.dataset == name].sort_values('K')
    ax.plot(sub.K, sub.speedup, marker='o', markersize=5, linewidth=2, color=CAT[i], label=name)
ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle='--', zorder=2)
ax.set_xlabel('Batch size $k$ (log scale)')
ax.set_ylabel('Speedup, batch vs. sequential insertion')
ax.legend(frameon=False, fontsize=9, loc='upper left')
style_axes(ax, log_x=True)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_batch_speedup_vs_k.pdf'))
plt.close(fig)
print('fig_batch_speedup_vs_k.pdf OK')

print('\nDone. All 3 figures in', OUTDIR)
