"""Exact marginal scaling, rejection envelope, and survival conditioning."""
from pathlib import Path
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch22_sampling import (bridge_survival, barrier_benchmarks,  # noqa: E402
                          beta_square_moments)
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from statistics import NormalDist  # noqa: E402

NORMAL = NormalDist()
COLORS = ['#8cdbce', '#ffc66d', '#bda6ef', '#d99ca6']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'path',
                     'svg.hashsalt': 'computational-finance-ch22'})


def frame(rows, cols, size):
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    return fig, axes


def legend(ax):
    ax.legend(facecolor='#182229', edgecolor='#607681',
              labelcolor='#e7edf0', fontsize=8)


layouts = [(1, 2, (10.8, 4.8), ''), (2, 1, (6, 9.2), '-portrait')]
for rows, cols, size, suffix in layouts:
    fig, axes = frame(rows, cols, size)
    grid = np.linspace(-3.5, 3.5, 701)
    for index, steps in enumerate((4, 40, 400)):
        nodes = (2 * np.arange(steps + 1) - steps) / math.sqrt(steps)
        mass = np.array([math.comb(steps, j) * .5**steps
                         for j in range(steps + 1)])
        np.testing.assert_allclose(np.sum(mass), 1., rtol=1e-13)
        cdf = np.r_[0., np.cumsum(mass)]
        values = cdf[np.searchsorted(nodes, grid, side='right')]
        axes[0].plot(grid, values, color=COLORS[index],
                     label=f'{steps} scaled coin steps')
    axes[0].plot(grid, [NORMAL.cdf(x) for x in grid],
                 color=COLORS[3], linestyle='--', label='Standard normal CDF')
    axes[0].set(xlabel='Endpoint after total time 1', ylabel='Probability ≤ x',
                title='Marginal convergence, not a process proof',
                xlim=(-3.5, 3.5), ylim=(0, 1))
    counts = np.geomspace(100, 1_000_000, 101)
    beta_variance = float(beta_square_moments()[1])
    axes[1].loglog(counts, np.sqrt(beta_variance / counts),
                   color=COLORS[0], label='Exact IID SE for X², X ~ Beta(2,2)')
    axes[1].scatter([10_000], [math.sqrt(beta_variance / 10_000)],
                    color=COLORS[1], s=35, zorder=4)
    axes[1].set(xlabel='Accepted IID observations N', ylabel='Standard error',
                title='100 times the draws gives 10 times smaller SE')
    for ax in axes:
        legend(ax)
    fig.savefig(ROOT / f'assets/ch22-walk-and-error{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

    fig, axes = frame(rows, cols, size)
    x = np.linspace(0, 1, 501)
    target = 6 * x * (1 - x)
    assert np.max(target) <= 1.5
    axes[0].plot(x, target, color=COLORS[0], label='Target f(x) = 6x(1-x)')
    axes[0].axhline(1.5, color=COLORS[1], linestyle='--',
                    label='Smallest envelope Lq(x) = 1.5')
    axes[0].axhline(1.3, color=COLORS[3], linestyle=':',
                    label='Invalid envelope L = 1.3')
    axes[0].set(xlabel='Candidate x', ylabel='Density / envelope',
                title='The bound must hold everywhere', xlim=(0, 1),
                ylim=(0, 1.8))
    stock = np.linspace(20, 300, 1401)
    logs = np.log(stock / 100)
    density = np.exp(-.5 * ((logs - .055) / .3)**2) / (
        stock * .3 * math.sqrt(2 * math.pi))
    h = bridge_survival(stock, 100, 80, .3, 1)
    survival, _, _ = barrier_benchmarks(100, 100, .1, .3, 1, 80)
    conditional = density * h / survival
    axes[1].plot(stock, density, color=COLORS[1],
                 label='Unconditioned terminal law')
    axes[1].plot(stock, conditional, color=COLORS[0],
                 label='Terminal law given survival')
    axes[1].axvline(80, color=COLORS[3], linestyle=':', label='Barrier B = 80')
    axes[1].set(xlabel='Terminal price (currency)',
                ylabel='Probability density',
                title=f'Survival probability = {survival:.6f}',
                xlim=(20, 300), ylim=(0, .022))
    for ax in axes:
        legend(ax)
    fig.savefig(ROOT / f'assets/ch22-rejection-and-survival{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

print('Wrote four Chapter 22 SVGs from analytic distributions and moments.')
