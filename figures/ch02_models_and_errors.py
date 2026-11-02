"""Interpolation conventions and truncation versus rounded evaluation."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'path',
                     'svg.hashsalt': 'computational-finance-ch02'})
time = np.linspace(1, 3, 301)
rates = .04 + .01 * (time - 1)
discount_rate = np.exp(-time * np.log1p(rates))
log_d1, log_d3 = -math.log1p(.04), -3 * math.log1p(.06)
discount_log = np.exp(log_d1 + (time - 1) * (log_d3 - log_d1) / 2)
np.testing.assert_allclose(discount_rate[[0, -1]], discount_log[[0, -1]],
                           rtol=1e-14)
degrees = np.arange(17)
truncation, bounds, rounded = [], [], []
with localcontext() as context:
    context.prec = 100
    reference = Decimal('0.5').exp()
    exact_sum = Fraction(0)
    terms = []
    for degree in degrees:
        n = int(degree)
        term = Fraction(1, 2**n * math.factorial(n))
        exact_sum += term
        terms.append(float(term))
        exact_decimal = (Decimal(exact_sum.numerator) /
                         Decimal(exact_sum.denominator))
        error = abs(reference - exact_decimal)
        first_omitted = Fraction(1, 2**(n+1) * math.factorial(n+1))
        bound = first_omitted / (1 - Fraction(1, 2*(n+2)))
        bound_decimal = Decimal(bound.numerator) / Decimal(bound.denominator)
        assert error <= bound_decimal
        truncation.append(float(error))
        bounds.append(float(bound))
        rounded.append(float(abs(Decimal.from_float(math.fsum(terms)) -
                                 reference)))

layouts = [(1, 2, (10.8, 4.8), ''), (2, 1, (6, 9.2), '-portrait')]
for rows, cols, size, suffix in layouts:
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].plot(time, discount_rate, color='#8cdbce',
                 label='Linear annual zero rate')
    axes[0].plot(time, discount_log, color='#ffc66d', linestyle='--',
                 label='Linear log discount factor')
    axes[0].scatter([1, 3], np.exp([log_d1, log_d3]), color='#bda6ef',
                    s=30, zorder=4, label='Same observed endpoint inputs')
    axes[0].set(xlabel='Maturity (years)', ylabel='Discount factor',
                title='Interpolation is a model choice', xlim=(1, 3))
    axes[1].semilogy(degrees, truncation, color='#8cdbce',
                     label='Exact-arithmetic truncation error')
    axes[1].semilogy(degrees, bounds, color='#ffc66d', linestyle='--',
                     label='Geometric tail bound')
    axes[1].semilogy(degrees, rounded, color='#bda6ef', linestyle=':',
                     marker='.', label='Total error after rounded sum')
    axes[1].set(xlabel='Taylor polynomial degree', ylabel='Absolute error',
                title='exp(0.5): a tail bound does not bound rounding')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                   labelcolor='#e7edf0', fontsize=8)
    fig.savefig(ROOT / f'assets/ch02-models-and-errors{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

diagnostic = {'degrees': degrees.tolist(), 'truncation': truncation,
              'tail_bounds': bounds, 'rounded_error': rounded,
              'D2_rate': float(discount_rate[150]),
              'D2_log_discount': float(discount_log[150])}
(ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
(ROOT / 'work/figures/ch02-models-and-errors.json').write_text(
    json.dumps(diagnostic, indent=2) + '\n', encoding='utf-8')
print('D2 rate/log-discount:', diagnostic['D2_rate'],
      diagnostic['D2_log_discount'])
print('At degree16: exact tail/rounding error:', truncation[-1], rounded[-1])
print('All exact truncation errors lie below the geometric bound.')
