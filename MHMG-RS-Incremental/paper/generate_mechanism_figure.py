"""
Geometric illustration for Section 3: the safety radius delta_B(x_i) and Theorems 1-2
(friend invariance, friend gain, enemy shrink), on hand-placed 2-D points.

Three panels:
(a) Initial state: delta_B(x_i) = distance to the nearest enemy.
(b) A friend (same class) is inserted inside the current ball -> friend gain;
    the radius is unchanged and the granule grows by 1.
(c) An enemy (other class) closer than the old nearest enemy is inserted -> enemy
    shrink; the radius decreases and one former member leaves the granule.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'figs')
os.makedirs(OUTDIR, exist_ok=True)

BLUE, ORANGE = '#2a78d6', '#eb6834'   # categorical slot 1, 2 (class A, class B)
TEXT_PRIMARY, TEXT_SECONDARY, GRAY_LIGHT = '#0b0b0b', '#52514e', '#c9c8c2'

plt.rcParams.update({
    'font.size': 10, 'text.color': TEXT_PRIMARY,
    'axes.edgecolor': GRAY_LIGHT, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.unicode_minus': True,
})

# --- hand-placed coordinates ---
xi = np.array([5, 5])                       # x_i, focal object
A_friends = np.array([[4, 4], [6, 6], [3, 6]])      # same class as x_i
B_enemies_far = np.array([[1, 8], [9, 2]])          # other class, far away (no effect)
B_nearest_old = np.array([8, 5])                    # initial nearest enemy

delta_old = np.linalg.norm(B_nearest_old - xi)      # = 3.0

new_friend = np.array([5, 7])               # (b) inserted friend, at distance 2 < delta_old from x_i
new_enemy = np.array([6.5, 3.5])            # (c) inserted enemy, at distance sqrt(4.5)=2.121 < delta_old from x_i
delta_new = np.linalg.norm(new_enemy - xi)


def dist(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))


def draw_base(ax, panel_label):
    # panel labels (a)/(b)/(c) only; the caption is in the LaTeX source.
    ax.scatter(*xi, color=BLUE, s=140, marker='o', zorder=5,
               edgecolor='white', linewidth=1.2, label=None)
    ax.annotate('$x_i$', xi, textcoords='offset points', xytext=(8, 6), fontsize=11)
    for p in A_friends:
        ax.scatter(*p, color=BLUE, s=70, marker='o', zorder=4, edgecolor='white', linewidth=0.8)
    ax.scatter(*B_nearest_old, color=ORANGE, s=70, marker='^', zorder=4, edgecolor='white', linewidth=0.8)
    for p in B_enemies_far:
        ax.scatter(*p, color=ORANGE, s=70, marker='^', zorder=4, edgecolor='white', linewidth=0.8,
                   alpha=0.55)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(panel_label, fontsize=11)


fig, axes = plt.subplots(1, 3, figsize=(11, 4))

# --- (a) initial state ---
ax = axes[0]
draw_base(ax, '(a)')
ax.add_patch(Circle(xi, delta_old, fill=False, edgecolor=BLUE, linewidth=1.6, linestyle='-'))
ax.plot([xi[0], B_nearest_old[0]], [xi[1], B_nearest_old[1]], color=TEXT_SECONDARY,
        linewidth=1, linestyle=':')
ax.annotate(f'$\\delta_B(x_i)={delta_old:.1f}$', (6.3, 5.6), fontsize=9.5, color=TEXT_SECONDARY)

# --- (b) Friend-gain ---
ax = axes[1]
draw_base(ax, '(b)')
ax.add_patch(Circle(xi, delta_old, fill=False, edgecolor=BLUE, linewidth=1.6, linestyle='-'))
ax.scatter(*new_friend, color=BLUE, s=110, marker='*', zorder=6, edgecolor='white', linewidth=1.0)
ax.annotate('new', new_friend, textcoords='offset points', xytext=(8, 4), fontsize=9, color=BLUE)
d_nf = dist(new_friend, xi)
ax.annotate(f'joins $G_B(x_i)$\n($\\delta_B$ unchanged)', (0.3, 1.0), fontsize=9, color=TEXT_SECONDARY)

# --- (c) Enemy-shrink ---
ax = axes[2]
draw_base(ax, '(c)')
ax.add_patch(Circle(xi, delta_old, fill=False, edgecolor=GRAY_LIGHT, linewidth=1.4, linestyle='--'))
ax.add_patch(Circle(xi, delta_new, fill=False, edgecolor=ORANGE, linewidth=1.6, linestyle='-'))
ax.scatter(*new_enemy, color=ORANGE, s=110, marker='*', zorder=6, edgecolor='white', linewidth=1.0)
ax.annotate('new', new_enemy, textcoords='offset points', xytext=(8, -12), fontsize=9, color=ORANGE)
# mark (3,6) as removed
evicted = A_friends[2]
ax.scatter(*evicted, s=180, facecolors='none', edgecolors=TEXT_SECONDARY, linewidth=1.4,
           marker='o', zorder=7)
ax.annotate('evicted', evicted, textcoords='offset points', xytext=(-38, 6), fontsize=9,
            color=TEXT_SECONDARY)
ax.annotate(f'$\\delta_B(x_i)$: {delta_old:.1f}$\\to${delta_new:.2f}', (0.3, 1.0), fontsize=9,
            color=TEXT_SECONDARY)

# shared legend (color = class, marker = state)
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE, markersize=9, label='class A (existing)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=ORANGE, markersize=9, label='class B (existing)'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=TEXT_SECONDARY, markersize=11, label='newly inserted'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_mechanism_illustration.pdf'))
plt.close(fig)
print('fig_mechanism_illustration.pdf OK')
print(f'delta_old={delta_old:.3f}, delta_new={delta_new:.3f}, dist(new_friend,xi)={d_nf:.3f}, '
      f'dist(evicted,xi)={dist(evicted,xi):.3f}')
