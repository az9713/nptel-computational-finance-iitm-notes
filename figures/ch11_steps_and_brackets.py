"""Draw Newton tangents and actual central-acceptance hybrid bracket widths."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch11_newton_secant import (
    coupon_bond, coupon_slope, newton, hybrid, workshop_bond)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def residual(y):
    return coupon_bond(y) - 98.4


def workshop_residual(y):
    return workshop_bond(y) - 101.75


def workshop_derivative(y):
    return workshop_bond(y, slope=True)


newton_result = newton(residual, coupon_slope, .05)
hybrid_result = hybrid(workshop_residual, 0, .2,
                        derivative=workshop_derivative, ftol=1e-8)
ys = np.linspace(.045, .082, 301)
widths = np.array([.2] + [r['after'][1] - r['after'][0]
                          for r in hybrid_result.trace])
steps = np.arange(len(widths))
assert np.all(widths[1:] <= .75 * widths[:-1] + 1e-15)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
for rows, cols, size, suffix in ((1, 2, (10.5, 4.3), ''),
                                 (2, 1, (6, 8.5), '-portrait')):
    fig, axes = plt.subplots(rows, cols, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].plot(100 * ys, [residual(y) for y in ys], color='#8cdbce',
                  label='Five-year bond price residual')
    axes[0].axhline(0, color='#afbec8', linewidth=.8)
    for j, color in enumerate(('#ffc66d', '#bda6ef')):
        row = newton_result.trace[j]
        x, fx = row['x'], row['f']
        next_x = newton_result.trace[j + 1]['x']
        segment = np.array([x, next_x])
        axes[0].plot(100 * segment, fx + coupon_slope(x) * (segment - x),
                      color=color, linestyle='--', label=f'Tangent at step {j}')
        axes[0].scatter([100*x, 100*next_x], [fx, 0], color=color, s=16)
    axes[0].set(xlabel='Nominal annual yield (%)', ylabel='Price residual',
                 title='Tangent zeros propose the next yield')
    axes[1].semilogy(steps, widths, color='#8cdbce', marker='.',
                      label='Actual four-year bond hybrid')
    axes[1].semilogy(steps, .2 * .75 ** steps, color='#ffc66d', linestyle=':',
                      label='Proved 3/4 contraction upper bound')
    axes[1].semilogy(steps, .2 * .5 ** steps, color='#bda6ef', linestyle='--',
                      label='Pure-halving width reference')
    axes[1].set(xlabel='Accepted updates', ylabel='Bracket width in yield units',
                 title='Transparent safeguard: not always faster')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='upper right')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch11-steps-and-brackets{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch11-steps-and-brackets{suffix}.png', dpi=140)
    plt.close(fig)
