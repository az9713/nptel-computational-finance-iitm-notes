"""Compare precisely named rate curves and synthetic finite-tree IV slices."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch12_curves_and_surfaces import (  # noqa: E402
    bond_price, bisect_width, surface_rows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

times = np.array([.5, 1, 1.5, 2])
discounts = np.array([.98, .955, .925, .89])
coupons = np.array([2, 2.5, 3, 3.5])
prices = np.array([c*sum(discounts[:i+1])+100*discounts[i]
                   for i, c in enumerate(coupons)])
ytms = np.array([bisect_width(lambda y: bond_price(y, round(2*t), c)-p,
                             0, .3, width_tol=1e-10).point
                 for t, c, p in zip(times, coupons, prices, strict=True)])
zeros = -np.log(discounts)/times
pars = 2*(1-discounts)/np.cumsum(discounts)
surface = surface_rows()
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
    axes[0].plot(times, 100*zeros, 'o-', color='#8cdbce',
                  label='Continuous zero rate')
    axes[0].plot(times, 100*pars, 's--', color='#ffc66d',
                  label='Nominal semiannual par rate')
    axes[0].plot(times, 100*ytms, 'x:', color='#bda6ef',
                  label='Nominal semiannual input-bond YTM')
    axes[0].set(xlabel='Time to payment / maturity (years)', ylabel='Annual rate (%)',
                 title='One discount curve, distinct rate conventions')
    for (t, strikes, prices, vols), color in zip(
            surface, ('#8cdbce', '#ffc66d', '#bda6ef'), strict=True):
        axes[1].plot(strikes, 100*vols, '.-', color=color, label=f'T = {t:g} years')
    axes[1].set(xlabel='Strike (spot = 100)', ylabel='Annual CRR implied volatility (%)',
                 title='Synthetic mixture prices, finite-tree inversions')
    for ax in axes:
        ax.legend(facecolor='#182229', edgecolor='#607681',
                  labelcolor='#e7edf0', fontsize=8, loc='best')
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch12-curve-objects{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch12-curve-objects{suffix}.png', dpi=140)
    plt.close(fig)
