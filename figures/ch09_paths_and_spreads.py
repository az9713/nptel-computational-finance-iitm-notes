"""Show same-terminal histories and the language labs' multi-leg payoffs."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch09_paths_and_option_labs import strategies
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

grid = np.arange(60., 151.)
strangle, bull, butterfly = strategies(grid)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
colors = ['#8cdbce', '#ffc66d', '#bda6ef']
for rows, cols, size, suffix in ((1, 2, (10.5, 4.2), ''),
                                 (2, 1, (6, 8.2), '-portrait')):
    fig, axes = plt.subplots(rows, cols, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].plot([0, 1, 2], [100, 120, 96], color=colors[0], marker='o',
                 label='UD: average 105.333; maximum 120')
    axes[0].plot([0, 1, 2], [100, 80, 96], color=colors[1], marker='s',
                 label='DU: average 92; maximum 100')
    axes[0].axhline(115, color=colors[2], linestyle=':', label='Barrier H = 115')
    axes[0].set(title='Same terminal price, different contract state',
                 xlabel='Observation date', ylabel='Share price',
                 xticks=[0, 1, 2], ylim=(72, 132))
    for series, name, color in zip((strangle, bull, butterfly),
                                   ('Long strangle: puts90 + calls110',
                                    'Bull call spread: calls90 - calls110',
                                    'Butterfly: calls90 - 2calls100 + calls110'),
                                   colors, strict=True):
        axes[1].plot(grid, series, label=name, color=color)
    axes[1].set(title='Multi-leg gross payoffs, no premium subtraction',
                 xlabel='Terminal share price', ylabel='Payoff per package',
                 xlim=(60, 150), ylim=(-2, 46))
    for ax in axes:
        ax.legend(loc='upper left', facecolor='#182229',
                  edgecolor='#607681', labelcolor='#e7edf0', fontsize=8)
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch09-paths-and-spreads{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch09-paths-and-spreads{suffix}.png', dpi=140)
    plt.close(fig)
