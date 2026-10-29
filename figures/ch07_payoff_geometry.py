"""Draw original chapter-7 claims and funded profits from executed payoffs."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch07_payoff_functions import running_case
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

names, premiums, growth, wealth = running_case()
prices = np.unique(np.r_[np.linspace(70, 130, 601), 94.5, 100, 105.25, 105.5])
values = np.stack([wealth(s) for s in prices])
profits = values - growth * premiums
np.testing.assert_allclose(values[:, 3], np.maximum(prices, 100))
np.testing.assert_allclose(values[:, 4], np.minimum(prices, 100))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
colors = ['#8cdbce', '#ffc66d', '#bda6ef']
for rows, columns, size, suffix in ((1, 3, (12, 3.9), ''),
                                    (3, 1, (5.8, 11), '-portrait')):
    fig, axes = plt.subplots(rows, columns, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
        ax.axvline(100, color='#afbec8', linewidth=.8, linestyle=':')
        ax.set(xlabel='Terminal share price (money/share)', xlim=(70, 130))
    for j in range(2):
        axes[0].plot(prices, values[:, j], color=colors[j], label=names[j])
        axes[1].plot(prices, profits[:, j], color=colors[j], label=names[j])
    axes[1].plot(prices, profits[:, 2], color=colors[2], label='straddle')
    axes[1].axhline(0, color='#afbec8', linewidth=.8)
    axes[1].scatter([94.5, 105.5], [0, 0], color=colors[2], s=18, zorder=4)
    axes[0].set(title='European claims, K = 100', ylabel='Expiry payoff')
    axes[1].set(title='Premiums funded at G = 1.05', ylabel='Terminal net P&L')
    axes[2].plot(prices, prices, color='#afbec8', linestyle='--', label='share')
    axes[2].plot(prices, values[:, 3], color=colors[0], label='protective put')
    axes[2].plot(prices, values[:, 4], color=colors[1], label='covered call')
    axes[2].set(title='Floor versus cap: gross values', ylabel='Terminal wealth')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=9)
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch07-payoff-geometry{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch07-payoff-geometry{suffix}.png', dpi=140)
    plt.close(fig)
