"""Measured elementary quadrature convergence and continuous utility allocation."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch15_quadrature import exp_convergence, canonical_eu, optimum_continuous
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows = np.array(exp_convergence())
alphas = np.linspace(0, 1, 201)
ces = np.array([(-2*canonical_eu(a).value)**(-.5) for a in alphas])
optimum = optimum_continuous().point
best = (-2*canonical_eu(optimum).value)**(-.5)
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
    axes[0].loglog(rows[:, 0], rows[:, 3], 'o-', color='#ffc66d', label='Trapezoid error')
    axes[0].loglog(rows[:, 0], rows[:, 4], 's-', color='#8cdbce', label='Simpson 1/3 error')
    axes[0].loglog(rows[:, 0], rows[0, 3]*(rows[0, 0]/rows[:, 0])**2,
                    ':', color='#bda6ef', label='n^-2 reference')
    axes[0].loglog(rows[:, 0], rows[0, 4]*(rows[0, 0]/rows[:, 0])**4,
                    '--', color='#afbec8', label='n^-4 reference')
    axes[0].set(xlabel='Number of equal intervals n', ylabel='Absolute integral error',
                 title='Integral of exp(x) on [0, 1]')
    axes[1].plot(alphas, ces, color='#8cdbce', label='Continuous lognormal-gross model')
    axes[1].axhline(105, color='#ffc66d', linestyle='--', label='Safe terminal wealth')
    axes[1].scatter([optimum], [best], color='#bda6ef', zorder=3,
                     label=f'Optimum alpha = {optimum:.6f}')
    axes[1].set(xlabel='Fraction in risky asset', ylabel='Certainty equivalent (W0 = 100)',
                 title='Specified log-return mean 0.08, sd 0.2')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch15-convergence-and-allocation{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch15-convergence-and-allocation{suffix}.png', dpi=140)
    plt.close(fig)
