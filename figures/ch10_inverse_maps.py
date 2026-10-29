"""Plot bond inversion and a nonidentifiable finite-tree volatility region."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch10_inverse_pricing import bisect_width, coupon_bond
from ch08_binomial_valuation import crr, lattice, payoff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

answer = bisect_width(lambda y: coupon_bond(y) - 98.4, 0, .3)
yields = np.linspace(0, .15, 301)
bond_prices = np.array([coupon_bond(y) for y in yields])
vols = np.linspace(.06, 1, 301)
call_prices = np.array([lattice(crr(100, .05, v, 1, 1), 1, payoff('call', 50))
                        for v in vols])
flat = 100 - 50 * np.exp(-.05)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for rows, cols, size, suffix in ((1, 2, (10.5, 4.1), ''),
                                 (2, 1, (6, 8.1), '-portrait')):
    fig, axes = plt.subplots(rows, cols, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].plot(100 * yields, bond_prices, color='#8cdbce')
    axes[0].axhline(98.4, color='#ffc66d', linestyle='--', label='Price = 98.40')
    axes[0].scatter([100 * answer.point], [98.4], color='#ffc66d', zorder=4)
    axes[0].set(xlabel='Nominal annual yield (%)', ylabel='Full price per 100 face',
                 title='Positive cash flows give one yield')
    axes[1].plot(100 * vols, call_prices, color='#8cdbce')
    axes[1].scatter([10, 20], [flat, flat], color='#ffc66d', zorder=4,
                     label='Same price at 10% and 20% volatility')
    axes[1].axvline(100 * np.log(2), color='#bda6ef', linestyle=':',
                    label='Down-state share price equals strike 50')
    axes[1].set(xlabel='Volatility parameter (% per sqrt(year))',
                 ylabel='One-step call price',
                 title='Finite-tree inversion can be flat')
    for j, ax in enumerate(axes):
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8,
                  loc='upper left' if j else 'upper right')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch10-inverse-maps{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch10-inverse-maps{suffix}.png', dpi=140)
    plt.close(fig)
