"""Plot the executed two-step workshop's European call and grid-American put."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch08_binomial_valuation import Market, lattice, payoff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

market = Market(50, 1.2, .85, 1.03)
call, call_levels = lattice(market, 2, payoff('call', 50), history=True)
put, put_levels = lattice(market, 2, payoff('put', 50),
                          american=True, history=True)
np.testing.assert_allclose(put_levels[1][0], 7.5)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'svg.fonttype': 'none'})
for rows, cols, size, suffix in ((1, 2, (11, 4.7), ''),
                                 (2, 1, (6, 9.3), '-portrait')):
    fig, axes = plt.subplots(rows, cols, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax, levels, title in zip(axes, (call_levels, put_levels),
                                 ('European call', 'Put: exercise on grid'),
                                 strict=True):
        ax.set_facecolor('#182229')
        for level in (0, 1):
            for j in range(level + 1):
                y = j - level / 2
                for change in (0, 1):
                    ax.plot([level, level + 1],
                            [y, j + change - (level + 1) / 2],
                            color='#607681', linewidth=1)
        for level in range(3):
            for j in range(level + 1):
                y = j - level / 2
                exercise = title.startswith('Put') and level == 1 and j == 0
                edge = '#ffc66d' if exercise else '#8cdbce'
                text = (f'S = {market.node(level, j):.3f}\n'
                        f'V = {levels[level][j]:.3f}')
                if exercise:
                    text += '\nexercise'
                ax.text(level, y, text, ha='center', va='center', fontsize=10,
                        bbox={'boxstyle': 'round,pad=.35',
                              'facecolor': '#182229', 'edgecolor': edge})
        ax.set(xlim=(-.35, 2.38), ylim=(-1.35, 1.35),
               title=title, xticks=[0, 1, 2],
               xticklabels=['Time 0', 'Time 1', 'Time 2'], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('K = 50; per-step G = 1.03; pricing up-weight q = 18/35')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch08-two-step-tree{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch08-two-step-tree{suffix}.png', dpi=140)
    plt.close(fig)
