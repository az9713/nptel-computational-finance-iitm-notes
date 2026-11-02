"""Original seeded pricing, posterior-audit and predictive-utility figures."""
from pathlib import Path
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
sys.path.insert(0, str(ROOT / 'code'))
from ch23_pricing_and_inference import (  # noqa: E402
    call_price, discounted_call_and_stock, posterior, posterior_moments,
    metropolis, student_pdf, utility_quadrature)
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
                     'svg.hashsalt': 'computational-finance-ch23'})
(ROOT / 'assets').mkdir(exist_ok=True)


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


rng = np.random.Generator(np.random.PCG64(2300))
payoff, _ = discounted_call_and_stock(rng.standard_normal(1_000_000))
exact = call_price(100, 100, .1, .3, 1)
counts = np.array([100, 1000, 10_000, 100_000, 1_000_000])
estimates = [estimate(payoff[:n]) for n in counts]
model = posterior(60, .00781, .0402)
moments = posterior_moments(model)
chains = []
for index, (offset, factor) in enumerate(((-.03, .5), (.03, 2.))):
    rng = np.random.Generator(np.random.PCG64(2303+index))
    chain, _, _ = metropolis(
        rng, model, initial_mu=model.mean+offset, variance_factor=factor)
    chains.append(chain)
quadrature = utility_quadrature(model)
refined = utility_quadrature(model, nodes=512)
for name in ('plugin_optimum', 'predictive_optimum'):
    assert abs(quadrature[name]-refined[name]) < 1e-7

for rows, cols, size, suffix in [(1, 3, (14, 4.7), ''),
                                 (3, 1, (6, 12), '-portrait')]:
    fig, axes = frame(rows, cols, size)
    axes[0].loglog(counts, [abs(q.mean-exact) for q in estimates],
                   'o-', color=COLORS[0], label='Observed absolute error')
    axes[0].loglog(counts, [q.se for q in estimates],
                   's--', color=COLORS[1], label='Estimated IID SE')
    axes[0].set(xlabel='Independent terminal draws N',
                ylabel='Currency units', title='One seeded run is nonmonotone')
    grid = np.linspace(-.013, .029, 501)
    scale = model.sample_sd/math.sqrt(model.count)
    density = student_pdf((grid-model.mean)/scale, model.count-1)/scale
    axes[1].hist(
        chains[0][:, 0], bins=65, density=True, color=COLORS[0],
        alpha=.48, label='MH retained states')
    axes[1].plot(
        grid, density, color=COLORS[1], linewidth=2, label='Exact posterior')
    axes[1].set(xlabel='Monthly log-return mean m_R',
                ylabel='Posterior density', xlim=(-.013, .029),
                title='An analytic law audits the chain')
    time = np.arange(2001, 20_001, 10)
    for index, chain in enumerate(chains):
        axes[2].plot(
            time, chain[::10, 0], color=COLORS[index],
            linewidth=.55, alpha=.7, label=f'Chain {index+1}')
    axes[2].axhline(moments['mu_mean'], color=COLORS[2], linestyle='--',
                    label='Exact posterior mean')
    axes[2].set(xlabel='Iteration (every 10th state displayed)',
                ylabel='Monthly mean m_R',
                title='A trace cannot prove convergence')
    for ax in axes:
        legend(ax)
    axes[1].legend(
        loc='upper right', facecolor='#182229',
        edgecolor='#607681', labelcolor='#e7edf0', fontsize=8)
    fig.savefig(ROOT / f'assets/ch23-estimation-and-posterior{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

for rows, cols, size, suffix in [(1, 2, (11.2, 4.8), ''),
                                 (2, 1, (6, 9.2), '-portrait')]:
    fig, axes = frame(rows, cols, size)
    returns = np.linspace(-.3, .32, 801)
    normal_z = (returns-model.mean)/model.sample_sd
    normal_density = np.exp(-normal_z**2/2) / (
        model.sample_sd*math.sqrt(2*math.pi))
    predictive_scale = quadrature['predictive_scale']
    predictive_density = student_pdf(
        (returns-model.mean)/predictive_scale,
        model.count-1)/predictive_scale
    axes[0].semilogy(
        returns, np.where(normal_density >= 1e-10, normal_density, np.nan),
        color=COLORS[1], label='Plug-in normal')
    axes[0].semilogy(
        returns, predictive_density, color=COLORS[0],
        label='Posterior predictive t, df = 59')
    axes[0].set(xlabel='Future monthly log return R',
                ylabel='Probability density (log scale)',
                title='Parameter integration changes the tails',
                xlim=(-.3, .32), ylim=(1e-10, 30))
    weights = np.linspace(0, 1, 201)
    for predictive, color, label in [
            (False, COLORS[1], 'Plug-in normal'),
            (True, COLORS[0], 'Predictive t')]:
        gain = [(quadrature['value'](w, predictive)-.008)*10_000
                for w in weights]
        axes[1].plot(weights, gain, color=color, label=label)
        name = 'predictive_optimum' if predictive else 'plugin_optimum'
        optimum = quadrature[name]
        value = (quadrature['value'](optimum, predictive)-.008)*10_000
        axes[1].scatter([optimum], [value], color=color, s=38, zorder=4)
    axes[1].axhline(
        0, color=COLORS[2], linestyle=':', label='Safe-only utility')
    axes[1].set(xlabel='Risky fraction w (long-only)',
                ylabel='Expected log-utility gain × 10,000',
                title='This example increases the preferred weight',
                xlim=(0, 1))
    for ax in axes:
        legend(ax)
    fig.savefig(ROOT / f'assets/ch23-predictive-decisions{suffix}.svg',
                metadata={'Date': None})
    plt.close(fig)

print('Wrote four Chapter23 SVGs from declared laws and original seeds.')
