"""Actual finite-difference errors and bond shock approximations."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch13_bond_calculus import annual_workshop, bond_measures, error_sweep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

reference = annual_workshop()
sweep = error_sweep()
shocks = np.linspace(-.04, .04, 241)
exact = np.array([bond_measures(.055+d, [6, 6, 106]).price for d in shocks])
linear = reference.price+reference.slope*shocks
quadratic = linear+.5*reference.curvature*shocks**2
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
    for j, label, color in ((1, 'Forward first derivative', '#ffc66d'),
                             (2, 'Centered first derivative', '#8cdbce'),
                             (3, 'Centered second derivative', '#bda6ef')):
        axes[0].loglog(sweep[:, 0], np.maximum(sweep[:, j], 1e-17),
                        color=color, label=label)
    axes[0].set(xlabel='Yield bump h (decimal annual yield)',
                 ylabel='Absolute relative derivative error',
                 title='Smaller steps eventually lose accuracy')
    for values, label, color, style in (
            (exact, 'Exact repricing', '#8cdbce', '-'),
            (linear, 'Duration only', '#ffc66d', '--'),
            (quadratic, 'Duration + convexity', '#bda6ef', ':')):
        axes[1].plot(100*shocks, 100*(values/reference.price-1),
                      color=color, linestyle=style, label=label)
    axes[1].set(xlabel='Yield change (percentage points)',
                 ylabel='Price change (%)', title='Three-year annual-coupon bond, y = 5.5%')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch13-sensitivity-and-steps{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch13-sensitivity-and-steps{suffix}.png', dpi=140)
    plt.close(fig)
