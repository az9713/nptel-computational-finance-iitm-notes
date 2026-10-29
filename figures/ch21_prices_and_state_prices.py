"""Bounded LP receipts and the state-price equations for the lecture market."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch21_arbitrage import LECTURE_A, bounded_lp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fair = 50 / 7
call_prices = np.unique(np.r_[np.linspace(6, 8, 41), fair])
receipts = np.array([-bounded_lp(np.array([1, 100, c]), -LECTURE_A,
                                 np.zeros(2)).value for c in call_prices])
q_up = np.linspace(.45, .9, 101)
bond_line = 20 / 21 - q_up
stock_line = (100 - 110 * q_up) / 90
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for rows, cols, size, suffix in [(1, 2, (10.8, 4.8), ''),
                                (2, 1, (6, 9.2), '-portrait')]:
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].plot(call_prices, receipts, color='#8cdbce',
                 label='Best initial receipt; future payoffs nonnegative')
    axes[0].axvline(fair, color='#bda6ef', linestyle='--',
                    label='Fair call price = 50/7')
    axes[0].scatter([6, fair, 8], [2 / 75, 0, .02],
                    color='#ffc66d', s=25, zorder=4)
    axes[0].set(xlabel='Quoted call price', ylabel='Initial receipt (currency)',
                 title='Normalization: each holding lies in [-1, 1]')
    axes[1].plot(q_up, bond_line, color='#8cdbce',
                 label='Bond: state prices sum to 20/21')
    axes[1].plot(q_up, stock_line, color='#ffc66d',
                 label='Stock: 110 s_u + 90 s_d = 100')
    axes[1].axvline(.6, color='#d99ca6', linestyle=':',
                    label='Call price 6 would require s_u = 0.6')
    axes[1].scatter([5 / 7], [5 / 21], color='#bda6ef', s=35, zorder=4,
                    label='Only bond/stock state-price pair')
    axes[1].set(xlabel='Up-state price s_u', ylabel='Down-state price s_d',
                 title='State prices are discounted values, not probabilities',
                 xlim=(.45, .9), ylim=(.04, .57))
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                   labelcolor='#e7edf0', fontsize=8, loc='upper right')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch21-prices-and-state-prices{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch21-prices-and-state-prices{suffix}.png', dpi=140)
    plt.close(fig)
