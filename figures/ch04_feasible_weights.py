"""Plot exact budget/target geometry, with two independently defined choices."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(.2, .6, 201)
weights = np.vstack([t-.2, 1.2-2*t, t])
assert np.allclose(weights.sum(axis=0), 1)
assert np.allclose(np.array([.05, .10, .15]) @ weights, .11)
assert np.min(weights) >= -1e-14
norm_t, risk_t = 13/30, 13/45
print('Feasible endpoints: (w1,w3)=(0,.2),(.4,.6)')
print(f'Least-norm point: ({norm_t-.2:.8f},{norm_t:.8f})')
print(f'Minimum illustrative variance: ({risk_t-.2:.8f},{risk_t:.8f})')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
fig.patch.set_facecolor('#182229')
ax.set_facecolor('#182229')
ax.fill([0, 1, 0], [0, 0, 1], color='#29424a', alpha=.8)
ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color='#76919c')
extension = np.linspace(0, .85, 100)
ax.plot(extension-.2, extension, '--', color='#8cdbce', alpha=.6)
ax.plot(weights[0], weights[2], color='#8cdbce', linewidth=3,
        label='Long-only target segment')
for point, color, label in ((norm_t, '#ffc66d', 'Least weight norm'),
                             (risk_t, '#bda6ef', 'Minimum stated variance')):
    ax.plot(point-.2, point, 'o', color=color, markersize=8, label=label)
ax.text(.49, .10, 'Budget triangle\nw₂ = 1 − w₁ − w₃ ≥ 0', fontsize=10)
ax.annotate('(0, 0.2)', (0, .2), xytext=(.02, .12), fontsize=10)
ax.annotate('(0.4, 0.6)', (.4, .6), xytext=(.46, .63), fontsize=10)
ax.set(xlabel='Weight w₁', ylabel='Weight w₃', xlim=(-.05, 1.02),
       ylim=(-.05, 1.02), title='One target leaves a segment of portfolios')
ax.set_aspect('equal')
ax.grid(color='#33444f', linewidth=.5)
ax.legend(loc='upper right', facecolor='#182229', edgecolor='#607681',
          labelcolor='#e7edf0', fontsize=9)
(ROOT / 'work/figures').mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / 'assets/ch04-feasible-weights.svg')
fig.savefig(ROOT / 'work/figures/ch04-feasible-weights.png', dpi=150)
