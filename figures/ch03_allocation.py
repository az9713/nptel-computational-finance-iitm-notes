"""Regenerate the two mathematical allocation plots in chapter 3."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rf, mean, sigma, aversion = .05, .12, .20, 4.0
w_star = (mean - rf) / (aversion * sigma**2)
assert abs(w_star - .4375) < 1e-14
print(f'Baseline risky weight={w_star:.8f}; clipping threshold={(mean-rf)/sigma**2:.8f}')
weights = np.linspace(0, 1, 401)
scores = rf + weights * (mean - rf) - aversion * sigma**2 * weights**2 / 2
levels = np.linspace(.05, 10, 500)
optimal = np.minimum(1, (mean-rf) / (levels * sigma**2))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
peak = rf + w_star*(mean-rf) - aversion*sigma**2*w_star**2/2
(ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
for rows, columns, size, suffix in ((1, 2, (9, 3.5), ''),
                                     (2, 1, (5, 7), '-portrait')):
    if rows == 2:
        plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(rows, columns, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
    axes[0].plot(100*weights, 100*scores, color='#8cdbce', linewidth=2)
    axes[0].plot(100*w_star, 100*peak, 'o', color='#ffc66d')
    axes[0].axvline(100*w_star, color='#ffc66d', linestyle=':', linewidth=1)
    axes[0].set(xlabel='Risky weight (%)',
                ylabel='Mean–variance score (×100)',
                title='Objective at penalty λ = 4', xlim=(0, 100))
    axes[1].plot(levels, 100*optimal, color='#8cdbce', linewidth=2)
    axes[1].plot(4, 100*w_star, 'o', color='#ffc66d')
    axes[1].axvline(1.75, color='#afbec8', linestyle=':', linewidth=1)
    axes[1].set(xlabel='Variance penalty λ\n(decimal-return convention)',
                ylabel='Optimal risky weight (%)',
                title='The bound changes the solution',
                xlim=(0, 10), ylim=(0, 105))
    fig.savefig(ROOT / f'assets/ch03-allocation{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch03-allocation{suffix}.png', dpi=150)
    plt.close(fig)
