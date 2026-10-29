"""Exact scalar portfolio section and consumer halfspace, in two layouts."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch20_constrained_portfolios import feasible_interval, long_only, bliss, COV, V
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base, lo, hi = feasible_interval(0.115)
optimum = long_only(0.115)
coordinates = np.linspace(-.02, .04, 201)
values = np.array([(base + t * V) @ COV @ (base + t * V) / 2
                   - optimum.variance / 2 for t in coordinates])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for rows, cols, size, suffix in [(1, 2, (10.8, 4.9), ''),
                                (2, 1, (6, 9.5), '-portrait')]:
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    ax = axes[0]
    ax.axvspan(0, .04, color='#8cdbce', alpha=.08,
               label='Visible part of feasible interval')
    ax.plot(coordinates, values, color='#8cdbce',
             label='Half variance minus constrained minimum')
    unrestricted = optimum.unrestricted_coordinate
    free_value = ((base + unrestricted * V) @ COV @
                  (base + unrestricted * V) / 2 - optimum.variance / 2)
    ax.scatter([unrestricted], [free_value], color='#ffc66d',
                label='Equality-only minimum: negative B weight', zorder=4)
    ax.scatter([0], [0], color='#bda6ef', marker='s',
                label='Long-only minimum: B weight zero', zorder=4)
    ax.axhline(0, color='#607681', linewidth=.8)
    ax.axvline(0, color='#607681', linewidth=.8)
    ax.set(xlabel='B weight t (zoom near the active boundary)',
            ylabel='Difference in half variance', title='Same 11.5% target; different feasible sets')
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 121)
    for radius in [1, np.sqrt(2), 2]:
        ax.plot(3 + radius * np.cos(theta), 2 + radius * np.sin(theta),
                 color='#607681', linewidth=.9)
    x = np.linspace(0, 5, 101)
    ax.plot(x, 3 - x, color='#ffc66d', label='Budget 6: x + y = 3')
    ax.fill_between(x, -1, np.minimum(3 - x, 5), color='#8cdbce', alpha=.08)
    binding = bliss([2, 2], 6)[0]
    ax.scatter([3], [2], color='#8cdbce', marker='o',
                label='Bliss (3, 2): affordable at budget 40', zorder=4)
    ax.scatter([binding[0]], [binding[1]], color='#bda6ef', marker='s',
                label='Budget 6 optimum (2, 1)', zorder=4)
    ax.set(xlabel='Quantity x', ylabel='Quantity y',
            title='Squared distance to bliss; halfspace constraint',
            xlim=(0, 5), ylim=(-1, 5), aspect='equal')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                   labelcolor='#e7edf0', fontsize=8, loc='upper left')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch20-active-constraints{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch20-active-constraints{suffix}.png', dpi=140)
    plt.close(fig)
