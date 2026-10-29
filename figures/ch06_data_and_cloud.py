"""Plot the fixed synthetic price panel and a reproducible candidate cloud."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch06_factors_and_data import daily_fixture, portfolio_cloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_, _, _, _, _, returns, prices = daily_fixture()
weights, means, vols = portfolio_cloud(returns)
assert weights.shape == (10000, 3)
assert np.allclose(weights.sum(axis=1), 1)
colors = ['#8cdbce', '#ffc66d', '#bda6ef']
labels = ['Asset A', 'Asset B', 'Asset C']
print('Final rebased synthetic prices:', 100*prices[-1]/prices[0])
print('Negative sample-mean fraction:', np.mean(means < 0))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for rows, columns, size, suffix in ((1, 2, (9, 3.6), ''),
                                    (2, 1, (5.4, 7.6), '-portrait')):
    if rows == 2:
        plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(rows, columns, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    rebased = 100*prices/prices[0]
    for j in range(3):
        axes[0].plot(np.arange(len(prices)), rebased[:, j], color=colors[j],
                     marker='o', markersize=3, label=labels[j])
    axes[0].set(xlabel='Synthetic daily observation index',
                ylabel='Price rebased to 100', title='One reproducible price panel')
    axes[0].legend(facecolor='#182229', edgecolor='#607681',
                    labelcolor='#e7edf0', fontsize=9)
    axes[1].scatter(100*vols, 10000*means, s=3, alpha=.12,
                    color='#8cdbce', rasterized=True)
    for j in range(3):
        axes[1].scatter(100*returns[:, j].std(ddof=1),
                        10000*returns[:, j].mean(), color=colors[j],
                        marker='*', s=100, edgecolor='#182229')
    axes[1].axhline(0, color='#afbec8', linewidth=.8)
    axes[1].set(xlabel='Sample daily standard deviation (%)',
                ylabel='Sample daily mean (basis points)',
                title='10,000 sampled constant-weight portfolios')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch06-data-cloud{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch06-data-cloud{suffix}.png', dpi=150)
