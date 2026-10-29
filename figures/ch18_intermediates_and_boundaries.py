"""Actual overrange binomial intermediates and a controlled boundary defect."""
from pathlib import Path
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch18_option_laboratory import put_boundary_fixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

counts = np.arange(20, 1501, 10)
log_coefficients = [math.log10(math.comb(int(n), int(n)//2)) for n in counts]
float_limit = math.log10(sys.float_info.max)
correct = put_boundary_fixture()
wrong = put_boundary_fixture(wrong=True)
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
    axes[0].plot(counts, log_coefficients, color='#8cdbce',
                 label='log10 central binomial coefficient')
    axes[0].axhline(float_limit, color='#ffc66d', linestyle='--',
                    label='Largest finite binary64 log10')
    axes[0].axvline(1030, color='#bda6ef', linestyle=':',
                    label='First overflowing central coefficient')
    axes[0].set(xlabel='Binomial step count n', ylabel='Base-10 logarithm',
                 title='Large intermediates despite finite probabilities')
    axes[1].plot(correct.stock, correct.values, color='#8cdbce',
                 label='Put payoff and put boundaries')
    axes[1].plot(wrong.stock, wrong.values, color='#ffc66d',
                 label='Same put payoff, wrong call boundaries')
    axes[1].axvline(100, color='#bda6ef', linestyle=':', label='Spot = strike = 100')
    axes[1].set(xlabel='Current stock price S', ylabel='Computed value',
                 title='Same stable M = 100, N = 1000 operator')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch18-intermediates-and-boundaries{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch18-intermediates-and-boundaries{suffix}.png', dpi=140)
    plt.close(fig)
