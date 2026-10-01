"""
Analysis figures for Section 5 (Experiments).

1. fig_trajectory.pdf        -- accuracy gap over time (line)
2. fig_gap_heatmap.pdf       -- accuracy gap, 12 datasets x 2 classifiers (diverging heatmap)
3. fig_rounding_confound.pdf -- Jaccard vs rounding precision (Wine/Musk/Urban)
4. fig_baseline_scaling.pdf  -- cost of re-evaluating CE (Xu2025b) vs dataset size

Colors: a palette safe for color-vision deficiency.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, '..', 'experiments')
OUTDIR = os.path.join(HERE, 'figs')
os.makedirs(OUTDIR, exist_ok=True)

# palette safe for color-vision deficiency
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUE, RED, GRAY_MID = '#2a78d6', '#e34948', '#f0efec'
TEXT_PRIMARY, TEXT_SECONDARY = '#0b0b0b', '#52514e'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': TEXT_PRIMARY, 'axes.linewidth': 0.8,
    'text.color': TEXT_PRIMARY, 'axes.labelcolor': TEXT_PRIMARY,
    'xtick.color': TEXT_PRIMARY, 'ytick.color': TEXT_PRIMARY,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    # Font matched to the manuscript's own LaTeX rendering (article class =
    # Computer Modern): DejaVu Serif body text + 'cm' mathtext, no external
    # LaTeX/dvipng dependency. Matches the journal-figure convention (full
    # box, inward ticks, no gridlines) seen in the reference papers.
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.unicode_minus': True,
})


def style_axes(ax):
    ax.tick_params(direction='in', top=True, right=True, which='both')
    for spine in ax.spines.values():
        spine.set_visible(True)

# ---------------------------------------------------------------------------
# 1. Trajectory: accuracy gap over time (Wine, Musk, Urban, Spambase)
# ---------------------------------------------------------------------------
traj = pd.read_csv(os.path.join(EXP_DIR, 'e0_measurement2_acc_gap_trajectory.csv'))
datasets_order = ['Wine', 'Spambase', 'Musk', 'Urban']  # stable first, unstable second

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
for ax, metric, panel in zip(axes, ['gap_3nn', 'gap_svm'], ['(a) 3-NN', '(b) SVM-RBF']):
    for i, name in enumerate(datasets_order):
        sub = traj[traj.dataset_name == name].sort_values('frac_of_data_seen')
        ax.plot(sub.frac_of_data_seen * 100, sub[metric], marker='o', markersize=5,
                linewidth=2, color=CAT[i], label=name)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=':')
    ax.set_xlabel('% of stream seen')
    ax.set_title(panel, fontsize=11)  # nhan panel; caption o LaTeX
    style_axes(ax)
axes[0].set_ylabel('Accuracy gap, B$_{current}$ $-$ B$_{init}$ (pp)')
axes[0].legend(frameon=False, fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_trajectory.pdf'))
plt.close(fig)
print('fig_trajectory.pdf OK')

# ---------------------------------------------------------------------------
# 2. Heatmap: accuracy gap, 12 datasets x 2 classifiers (diverging blue-red)
# ---------------------------------------------------------------------------
bs = pd.concat([pd.read_csv(os.path.join(EXP_DIR, 'stageA_batch_summary.csv')),
                pd.read_csv(os.path.join(EXP_DIR, 'stageB_batch_summary.csv'))])
ckpt = pd.concat([pd.read_csv(os.path.join(EXP_DIR, 'stageA_accuracy_checkpoints.csv')),
                   pd.read_csv(os.path.join(EXP_DIR, 'stageB_accuracy_checkpoints.csv'))])
final = ckpt[ckpt.frac_of_stream == 1.0]
piv = final.pivot_table(index=['dataset', 'dataset_name'], columns='reduct_used',
                          values=['acc_3nn_mean', 'acc_svm_mean']).reset_index()
piv.columns = ['dataset', 'dataset_name'] + ['_'.join(c) for c in piv.columns[2:]]
piv['gap_3nn'] = (piv.acc_3nn_mean_B_full - piv.acc_3nn_mean_B_init) * 100
piv['gap_svm'] = (piv.acc_svm_mean_B_full - piv.acc_svm_mean_B_init) * 100
order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12']
piv['dataset'] = pd.Categorical(piv['dataset'], categories=order, ordered=True)
piv = piv.sort_values('dataset')

mat = piv[['gap_3nn', 'gap_svm']].values
names = piv['dataset_name'].tolist()
vmax = np.abs(mat).max()
cmap = mcolors.LinearSegmentedColormap.from_list('div', [BLUE, GRAY_MID, RED], N=256)

fig, ax = plt.subplots(figsize=(4.5, 6))
im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks([0, 1])
ax.set_xticklabels(['3-NN', 'SVM-RBF'])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        txt_color = 'white' if abs(v) > vmax * 0.55 else TEXT_PRIMARY
        weight = 'bold' if abs(v) > 5 else 'normal'
        ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8.5,
                color=txt_color, fontweight=weight)
cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.08)
cbar.set_label('gap (pp)', fontsize=9)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_gap_heatmap.pdf'))
plt.close(fig)
print('fig_gap_heatmap.pdf OK')

# ---------------------------------------------------------------------------
# 3. Rounding confound: Jaccard vs rounding precision (Wine, Musk, Urban)
#
# Read from experiments/rounding_confound_jaccard.csv (produced by
# experiments/test_rounding_confound_musk_urban.py).
# ---------------------------------------------------------------------------
_round_df = pd.read_csv(os.path.join(EXP_DIR, 'rounding_confound_jaccard.csv'))
_round_df = _round_df[_round_df['precision'] <= 5]  # drop the "no rounding" row (precision=15), identical to p5
rounding_data = {
    name: dict(zip(g['precision'], g['jaccard_init_full']))
    for name, g in _round_df.groupby('dataset_name', sort=False)
}
rounding_data = {k: rounding_data[k] for k in ['Wine', 'Musk', 'Urban']}  # fixed plotting order and colors
# Wine converges to Jaccard=1.0 (rounding explains the whole difference);
# Musk/Urban stay below 1.0.
fig, ax = plt.subplots(figsize=(5.5, 4))
precisions = [2, 3, 4, 5]
for i, (name, vals) in enumerate(rounding_data.items()):
    ys = [vals[p] for p in precisions]
    ax.plot(precisions, ys, marker='o', markersize=6, linewidth=2, color=CAT[i], label=name)
ax.set_xticks(precisions)
ax.set_xlabel('$\\gamma$ rounding precision (decimal places)')
ax.set_ylabel('Jaccard($B_{init}, B_{full}$)')
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=':')
ax.legend(frameon=False, fontsize=9, loc='lower right')
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_rounding_confound.pdf'))
plt.close(fig)
print('fig_rounding_confound.pdf OK')

# ---------------------------------------------------------------------------
# 4. Baseline scaling: cost of re-evaluating CE (Xu2025b) vs dataset size
#
# Read from experiments/ce_reevaluation_cost.csv (produced by
# baseline_xu2025b/measure_ce_reevaluation_cost.py: timeit.repeat, 7 runs,
# minimum, on the neighborhood matrix of each dataset after the whole stream).
# ---------------------------------------------------------------------------
scaling_df = pd.read_csv(os.path.join(EXP_DIR, 'ce_reevaluation_cost.csv'))
scaling_df = scaling_df.sort_values('n')
names_s = scaling_df['dataset_name'].tolist()
ns = scaling_df['n'].tolist()
costs = scaling_df['ce_reeval_time_min_sec'].tolist()  # all > 0, no floor needed

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(len(names_s)), costs, color=BLUE, edgecolor='white', linewidth=0.5)
ax.set_yscale('log')
ax.set_xticks(range(len(names_s)))
ax.set_xticklabels(names_s, rotation=40, ha='right')
ax.set_ylabel('CE re-evaluation cost per checkpoint (s, log scale)')
style_axes(ax)
for i, (bar, n) in enumerate(zip(bars, ns)):
    ax.text(bar.get_x() + bar.get_width() / 2, costs[i] * 1.3, f'n={n}',
            ha='center', fontsize=7.5, color=TEXT_SECONDARY)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_baseline_scaling.pdf'))
plt.close(fig)
print('fig_baseline_scaling.pdf OK')

print('\nDone. All 4 figures in', OUTDIR)
