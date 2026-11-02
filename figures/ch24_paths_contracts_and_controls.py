"""Original Asian-path information, geometric controls and date contracts."""
from pathlib import Path
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch24_asian_workflow import (  # noqa: E402
    equal_dates, sample_payoffs, geometric_price, pilot_control,
    lognormal_call)
from ch22_sampling import estimate  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ['#8cdbce', '#ffc66d', '#bda6ef', '#d99ca6']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': '#e7edf0', 'axes.labelcolor': '#e7edf0',
                     'xtick.color': '#afbec8', 'ytick.color': '#afbec8',
                     'axes.edgecolor': '#607681', 'svg.fonttype': 'path',
                     'svg.hashsalt': 'computational-finance-ch24'})
(ROOT / 'assets').mkdir(exist_ok=True)


def frame(rows, cols, size):
    fig, axes = plt.subplots(rows, cols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor('#182229')
    for ax in axes:
        ax.set_facecolor('#182229')
        ax.grid(color='#33444f', linewidth=.6)
        ax.set_axisbelow(True)
    return fig, axes


def legend(ax, **kwargs):
    ax.legend(facecolor='#182229', edgecolor='#607681',
              labelcolor='#e7edf0', fontsize=8, **kwargs)


dates = equal_dates(250)
controls = pilot_control(dates)
paired = controls['paired_draws']
rng = np.random.Generator(np.random.PCG64(2400))
iid = estimate(sample_payoffs(rng, dates, 100_000).arithmetic)
continuous = lognormal_call(
    math.log(100.)+(.1-.5*.3**2)/2, .3**2/3, 100., math.exp(-.1))

for rows, cols, size, suffix in [(1, 2, (11.6, 4.9), ''),
                                 (2, 1, (6, 9.4), '-portrait')]:
    fig, axes = frame(rows, cols, size)
    for index, (observations, label) in enumerate([
            ([96., 78., 132., 114.], 'Supplied observations: A = 105'),
            ([120., 120., 120., 114.], 'Original history: A = 118.5')]):
        axes[0].plot(
            range(5), [100., *observations], 'o-',
            color=COLORS[index], label=label)
        axes[0].axhline(np.mean(observations), color=COLORS[index],
                        linestyle=':', linewidth=1)
    axes[0].axhline(100, color=COLORS[2], linestyle='--', label='Strike = 100')
    axes[0].set(xlabel='Observation index (declared initial value at 0)',
                ylabel='Stock price (currency)', xticks=range(5),
                title='Equal terminal prices, different averages')
    legend(axes[0], loc='lower right')
    shown = slice(None, None, 25)
    axes[1].scatter(
        paired.geometric[shown], paired.arithmetic[shown],
        s=5, color=COLORS[0], alpha=.4,
        label='Every 25th independent pair displayed')
    end = max(float(np.max(paired.geometric)),
              float(np.max(paired.arithmetic)))*1.03
    axes[1].plot(
        [0, end], [0, end], color=COLORS[1], linestyle='--',
        label='Equal arithmetic and geometric payoff')
    axes[1].set(xlabel='Paired discounted geometric payoff',
                ylabel='Paired discounted arithmetic payoff',
                title='Shared paths produce a correlated control',
                xlim=(0, end), ylim=(0, end))
    legend(axes[1], loc='lower right')
    fig.savefig(ROOT / f'assets/ch24-paths-and-controls{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

for rows, cols, size, suffix in [(1, 2, (11.6, 4.9), ''),
                                 (2, 1, (6, 9.4), '-portrait')]:
    fig, axes = frame(rows, cols, size)
    counts = np.unique(np.rint(np.geomspace(1, 500, 90)).astype(int))
    exclude = [geometric_price(equal_dates(int(m))) for m in counts]
    include = [geometric_price(equal_dates(int(m), include_initial=True))
               for m in counts]
    axes[0].semilogx(
        counts, exclude, color=COLORS[0],
        label='m observations, excluding initial value')
    axes[0].semilogx(
        counts, include, color=COLORS[1],
        label='m + 1 observations, including initial')
    axes[0].axhline(continuous, color=COLORS[2], linestyle='--',
                    label='Continuous geometric limit')
    axes[0].set(xlabel='Equal time intervals m, ending at maturity',
                ylabel='Exact geometric call price',
                title='Dates define different contracts',
                xlim=(1, 500))
    legend(axes[0])
    cases = [iid, controls['paired'], controls['unit_control'],
             controls['controlled']]
    labels = ['IID\n100k paths', 'Antithetic\n100k pairs',
              'Pairs + G\nunit coefficient', 'Pairs + G\npilot coefficient']
    for index, result in enumerate(cases):
        axes[1].errorbar(
            result.mean, index, xerr=1.96*result.se, fmt='o',
            color=COLORS[index], capsize=5)
    axes[1].set(yticks=range(4), yticklabels=labels,
                xlabel='Arithmetic call estimate (1.96 SE whiskers)',
                title="Controls reduce this run's estimated error",
                xlim=(9.04, 9.23), ylim=(-.5, 3.5))
    axes[1].invert_yaxis()
    fig.savefig(ROOT / f'assets/ch24-dates-and-precision{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

print('Wrote four Chapter24 SVGs from declared paths, laws and seeds.')
