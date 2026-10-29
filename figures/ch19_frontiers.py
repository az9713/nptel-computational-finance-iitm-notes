"""Computed equality frontiers; teaching inputs are not market estimates."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch19_mean_variance import (LECTURE_MU, LECTURE_COV, REVATI_MU,
                                REVATI_COV, equality_portfolio, global_minimum)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for rows, cols, size, suffix in [(1, 2, (10.8, 4.7), ''),
                                (2, 1, (6, 9), '-portrait')]:
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax, mu, cov, rf, limits, title in zip(
            axes, [LECTURE_MU, REVATI_MU], [LECTURE_COV, REVATI_COV],
            [.05, .02], [(.06, .14), (.02, .14)],
            ['Recovered lecture matrix: raw numerical units',
             'Revati: decimal simple-return units']):
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
        _, m0, v0 = global_minimum(mu, cov)
        for lo, hi, color, style, label in [
                (limits[0], m0, '#ffc66d', '--', 'Dominated lower branch'),
                (m0, limits[1], '#8cdbce', '-', 'Efficient risky branch')]:
            targets = np.linspace(lo, hi, 121)
            sds = [np.sqrt(equality_portfolio(mu, cov, m).variance) for m in targets]
            ax.plot(sds, targets, color=color, linestyle=style, label=label)
        d = np.linalg.solve(cov, mu - rf)
        w = d / np.sum(d)
        mt, st = float(mu @ w), np.sqrt(float(w @ cov @ w))
        sigma = np.linspace(0, st * 1.5, 101)
        ax.plot(sigma, rf + (mt - rf) / st * sigma, color='#bda6ef',
                label='Cash + efficient risky exposures')
        ax.scatter([np.sqrt(v0)], [m0], marker='s', color='#ffffff',
                   label='Global minimum variance', zorder=4)
        ax.scatter([st], [mt], color='#bda6ef', marker='o',
                   edgecolor='#ffffff', label='Tangency fund', zorder=4)
        targets = [.08, .09, .10, .11, .12] if rf == .05 else [.04, .08, .12]
        ax.scatter([np.sqrt(equality_portfolio(mu, cov, m).variance) for m in targets],
                   targets, color='#ffc66d', s=16, zorder=4)
        ax.set(xlabel='Portfolio standard deviation (return units)',
               ylabel='Expected one-period simple return', title=title,
               ylim=(limits[0] - .015, limits[1] + .01))
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch19-frontiers{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch19-frontiers{suffix}.png', dpi=140)
    plt.close(fig)
