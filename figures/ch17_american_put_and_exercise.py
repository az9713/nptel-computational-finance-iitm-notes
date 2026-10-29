"""Computed put values and the explicit grid's exercise/continuation bracket."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch17_finite_differences import explicit_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

eu = explicit_grid(100, .1, .3, 1, 400, 200)
am = explicit_grid(100, .1, .3, 1, 400, 200, american=True)
shown = (am.stock >= 40) & (am.stock <= 150)
bracket = am.exercise_brackets
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for r, c, size, suffix in ((1, 2, (10.8, 4.5), ''),
                           (2, 1, (6, 8.8), '-portrait')):
    fig, axes = plt.subplots(r, c, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].plot(am.stock[shown], am.values[shown], color='#8cdbce',
                 label='Grid-exercise American put')
    axes[0].plot(eu.stock[shown], eu.values[shown], color='#ffc66d',
                 label='European put, same grid')
    axes[0].plot(am.stock[shown], np.maximum(100 - am.stock[shown], 0),
                 ':', color='#bda6ef', label='Immediate exercise payoff')
    axes[0].set(xlabel='Current stock price S', ylabel='Put value / payoff',
                 title='K = 100, r = 0.1, sigma = 0.3, tau = 1')
    axes[1].step(bracket[:, 0], bracket[:, 1], where='post', color='#8cdbce',
                 label='Last positive-payoff exercise node')
    axes[1].step(bracket[:, 0], bracket[:, 2], where='post', color='#ffc66d',
                 linestyle='--', label='Next continuation node')
    axes[1].fill_between(bracket[:, 0], bracket[:, 1], bracket[:, 2],
                         step='post', color='#8cdbce', alpha=.14)
    axes[1].set(xlabel='Time remaining tau (years)', ylabel='Stock grid coordinate',
                 title=f'Stock spacing 2, {am.time_steps} time intervals')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch17-american-put-and-exercise{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch17-american-put-and-exercise{suffix}.png', dpi=140)
    plt.close(fig)
