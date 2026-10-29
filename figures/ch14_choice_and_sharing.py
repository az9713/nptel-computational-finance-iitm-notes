"""Certainty-equivalent allocation and sharing of one common risky payoff."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch14_expected_utility import (
    RETURNS, PROBABILITIES, terminal_wealth, crra_ce, cara_ce,
    discrete_derivative, bisect_width)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

alphas = np.linspace(0, 1, 201)
ces = np.array([crra_ce([terminal_wealth(500, a, .045, r) for r in RETURNS],
                        PROBABILITIES, 3) for a in alphas])
optimum = bisect_width(discrete_derivative, 0, 1, width_tol=1e-10).point
best = crra_ce([terminal_wealth(500, optimum, .045, r) for r in RETURNS],
               PROBABILITIES, 3)
holders = np.unique(np.rint(np.logspace(0, 3, 151)).astype(int))
shared = np.array([n*cara_ce(np.array([50000, -25000])/n, [.5, .5], 1/50000)
                    for n in holders])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for rows, cols, size, suffix in ((1, 2, (10.8, 4.5), ''),
                                 (2, 1, (6, 8.8), '-portrait')):
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].plot(alphas, ces, color='#8cdbce', label='Three-state CRRA, gamma = 3')
    axes[0].axhline(522.5, color='#ffc66d', linestyle='--', label='Safe wealth')
    axes[0].scatter([optimum], [best], color='#bda6ef', zorder=3,
                     label=f'Optimal alpha = {optimum:.4f}')
    axes[0].set(xlabel='Fraction in risky asset', ylabel='Certainty equivalent (W0 = 500)',
                 title='A higher expected return need not justify full risk')
    axes[1].semilogx(holders, shared, color='#8cdbce', label='n times CE(X / n)')
    axes[1].axhline(12500, color='#ffc66d', linestyle='--', label='Expected total payoff')
    axes[1].set(xlabel='Number of identical holders n',
                 ylabel='Sum of individual certainty equivalents',
                 title='Sharing changes holders, not the total payoff')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch14-choice-and-sharing{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch14-choice-and-sharing{suffix}.png', dpi=140)
    plt.close(fig)
