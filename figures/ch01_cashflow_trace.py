"""Dated cash-flow amounts, present values, and the summation invariant."""
import json
from fractions import Fraction
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

times = np.array([1., 2., 3.])
amounts = np.array([100., 100., 1100.])
discounts = np.exp(-times * math.log1p(.05))
present = amounts * discounts
prefix = np.r_[0., np.cumsum(present)]
exact_price = sum(Fraction(int(c)) / Fraction(21, 20)**int(t)
                  for c, t in zip(amounts, times))
assert math.isclose(float(prefix[-1]), float(exact_price), rel_tol=1e-13)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'path',
                     'svg.hashsalt': 'computational-finance-ch01'})
layouts = [(1, 2, (10.8, 4.8), ''), (2, 1, (6, 9.2), '-portrait')]
for rows, cols, size, suffix in layouts:
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].bar(times-.17, amounts, width=.32, color='#ffc66d',
                label='Payment at its future date')
    axes[0].bar(times+.17, present, width=.32, color='#8cdbce',
                label='Value at date zero')
    axes[0].set(xlabel='Payment date (years)', ylabel='Amount (currency)',
                title='Convert each dated amount before adding',
                xticks=times, ylim=(0, 1350))
    axes[1].step(np.arange(4), prefix, where='post', color='#8cdbce',
                 label='Accumulator after each payment')
    axes[1].scatter(np.arange(4), prefix, s=30, color='#8cdbce', zorder=4)
    axes[1].axhline(float(np.sum(amounts)), color='#ffc66d', linestyle='--',
                    label='Undiscounted total')
    axes[1].annotate(f'{prefix[-1]:.2f}', xy=(3, prefix[-1]),
                     xytext=(-8, -18), textcoords='offset points',
                     ha='right', color='#8cdbce')
    axes[1].set(xlabel='Number of payments processed',
                ylabel='Accumulator (currency)',
                title='The loop invariant retains every term',
                xticks=np.arange(4), xlim=(-.1, 3.2), ylim=(0, 1550))
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                   labelcolor='#e7edf0', fontsize=8, loc='upper left')
    fig.savefig(ROOT / f'assets/ch01-cashflow-trace{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

diagnostic = {'times': times.tolist(), 'cashflows': amounts.tolist(),
              'discounts': discounts.tolist(), 'present_values': present.tolist(),
              'prefix': prefix.tolist(), 'rate_effective_annual': .05}
(ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
(ROOT / 'work/figures/ch01-cashflow-trace.json').write_text(
    json.dumps(diagnostic, indent=2) + '\n', encoding='utf-8')
print('Present-value terms:', present)
print('Accumulator trace:', prefix)
