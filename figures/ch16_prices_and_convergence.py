"""Actual Black–Scholes price curves and signed CRR errors for the workshop."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch16_black_scholes import black_scholes
from ch08_binomial_valuation import crr, lattice, payoff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

spots = np.linspace(20, 160, 281)
calls = [black_scholes(s, 90, .045, .28, .75) for s in spots]
puts = [black_scholes(s, 90, .045, .28, .75, 'put') for s in spots]
counts = np.arange(4, 201)
benchmark = black_scholes(85, 90, .045, .28, .75)
errors = [lattice(crr(85, .045, .28, .75, int(n)), int(n), payoff('call', 90))
          - benchmark for n in counts]
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
    axes[0].plot(spots, calls, color='#8cdbce', label='Call price, tau = 0.75')
    axes[0].plot(spots, puts, color='#ffc66d', label='Put price, tau = 0.75')
    axes[0].plot(spots, np.maximum(spots - 90, 0), ':', color='#8cdbce',
                 label='Call expiry payoff')
    axes[0].plot(spots, np.maximum(90 - spots, 0), ':', color='#ffc66d',
                 label='Put expiry payoff')
    axes[0].axvline(85, color='#bda6ef', linestyle='--', linewidth=.9,
                    label='Workshop spot 85')
    axes[0].set(xlabel='Current stock price S', ylabel='Option price / payoff',
                 title='Strike 90, r = 0.045, sigma = 0.28')
    axes[1].plot(counts, errors, '.-', markersize=3, linewidth=.7, color='#8cdbce',
                 label='CRR call minus Black–Scholes')
    axes[1].axhline(0, color='#ffc66d', linewidth=.9)
    axes[1].set(xlabel='CRR step count n', ylabel='Signed price error',
                 title='Spot 85, strike 90, horizon 0.75 years')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch16-prices-and-convergence{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch16-prices-and-convergence{suffix}.png', dpi=140)
    plt.close(fig)
