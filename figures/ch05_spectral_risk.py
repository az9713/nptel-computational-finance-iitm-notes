"""Plot trace shares and normalized loadings for the explicit chapter-5 matrix."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch05_covariance_geometry import sector_covariance, sorted_eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

covariance = sector_covariance()
values, vectors = sorted_eigh(covariance)
shares = values / values.sum()
np.testing.assert_allclose(covariance @ vectors, vectors * values, atol=1e-14)
print('Trace shares:', shares)
print('Unit eigenvector loadings:\n', vectors)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
colors = ['#8cdbce', '#ffc66d', '#bda6ef']
for rows, columns, size, suffix in ((1, 2, (9, 3.6), ''),
                                    (2, 1, (5.4, 7.6), '-portrait')):
    if rows == 2:
        plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(rows, columns, figsize=size,
                             constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(axis='x', color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].barh([2, 1, 0], 100*shares, color=colors)
    axes[0].set(yticks=[2, 1, 0], yticklabels=['Mode 1', 'Mode 2', 'Mode 3'],
                xlabel='Share of covariance trace (%)', xlim=(0, 100),
                title='Concentration of total variance')
    for j, share in enumerate(shares):
        axes[0].text(100*share+1, 2-j, f'{100*share:.2f}%', va='center')
    for j in range(3):
        axes[1].barh(np.array([2, 1, 0]) + (j-1)*.23, vectors[:, j],
                     height=.22, color=colors[j], label=f'Mode {j+1}')
    axes[1].axvline(0, color='#afbec8', linewidth=.8)
    axes[1].set(yticks=[2, 1, 0],
                yticklabels=['Iron north', 'Iron south', 'Silk'],
                xlabel='Unit eigenvector loading', xlim=(-1.05, 1.05),
                title='Common, within-sector, and silk modes')
    axes[1].legend(loc='lower left', facecolor='#182229',
                    edgecolor='#607681', labelcolor='#e7edf0', fontsize=9)
    (ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'assets/ch05-spectral-risk{suffix}.svg')
    fig.savefig(ROOT / f'work/figures/ch05-spectral-risk{suffix}.png', dpi=150)
