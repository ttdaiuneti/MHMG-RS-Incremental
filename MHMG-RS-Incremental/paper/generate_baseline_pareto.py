"""
Reduct size vs accuracy of the four methods (MHMG-RS, Xu2025b, Li2026, Deng2026),
two panels (3-NN | SVM-RBF). Each point is (reduct size, accuracy) of one method on
one dataset; large outlined markers are the per-method means.

Read from experiments/baseline_comparison_4way.csv. No title in the figure (the
caption is in the LaTeX source). Style as in generate_analysis_figures.py.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, '..', 'experiments')
OUTDIR = os.path.join(HERE, 'figs')

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # Ours, Xu, Li, Deng
TEXT_PRIMARY, TEXT_SECONDARY = '#0b0b0b', '#52514e'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': TEXT_PRIMARY, 'axes.linewidth': 0.8,
    'text.color': TEXT_PRIMARY, 'axes.labelcolor': TEXT_PRIMARY,
    'xtick.color': TEXT_PRIMARY, 'ytick.color': TEXT_PRIMARY,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.unicode_minus': True,
})


def style_axes(ax):
    ax.tick_params(direction='in', top=True, right=True, which='both')
    for spine in ax.spines.values():
        spine.set_visible(True)


df = pd.read_csv(os.path.join(EXP_DIR, 'baseline_comparison_4way.csv'))
methods = [('Ours', 'ours'), ('Xu2025b', 'xu'), ('Li2026', 'li'), ('Deng2026', 'deng')]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharex=True)
for ax, clf, label in [(axes[0], '3nn', '3-NN accuracy (\\%)'),
                       (axes[1], 'svm', 'SVM-RBF accuracy (\\%)')]:
    for i, (name, key) in enumerate(methods):
        sz = df[f'{key}_size'].values
        acc = df[f'{key}_{clf}'].values
        ax.scatter(sz, acc, s=22, color=CAT[i], alpha=0.55, edgecolors='none', zorder=2)
        # mean marker (vien den, lon)
        ax.scatter(sz.mean(), acc.mean(), s=140, color=CAT[i], edgecolors=TEXT_PRIMARY,
                   linewidths=1.3, marker='D', zorder=4, label=name)
    ax.set_xscale('log')
    ax.set_xlabel('Reduct size (attributes, log scale)')
    ax.set_ylabel(label)
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels(['4', '8', '16', '32', '64'])
    style_axes(ax)

axes[0].legend(frameon=False, fontsize=8.5, loc='lower left', handletextpad=0.3)
fig.tight_layout()
out = os.path.join(OUTDIR, 'fig_baseline_pareto.pdf')
fig.savefig(out)
print('saved', out)
