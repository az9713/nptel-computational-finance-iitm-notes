"""Explicit no-dividend European / grid-exercise American option solver.

Nonnegative rates, positive volatility/horizon, uniform stock grid. The default
uses centered drift except where a one-sided drift is needed for nonnegative
off-diagonals. It rejects nonmonotone time steps. allow_unsafe is deliberately
restricted to diagnostic experiments; its result is not a reliable price.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import scalar, step_count, crr, lattice, payoff  # noqa: E402
from ch16_black_scholes import black_scholes, greeks  # noqa: E402


def coefficients(rate, volatility, space_steps, drift='monotone'):
    """Interior generator entries lower, diagonal, upper; no dt factor."""
    r = scalar(rate, 'rate')
    sig = scalar(volatility, 'volatility', positive=True)
    m = step_count(space_steps)
    if r < 0 or m < 3 or drift not in ('central', 'monotone'):
        raise ValueError('require r>=0, M>=3 and central or monotone drift')
    i = np.arange(1, m, dtype=float)
    diffusion = .5 * sig ** 2 * i ** 2
    lower = diffusion - .5 * r * i
    upper = diffusion + .5 * r * i
    if drift == 'monotone':
        fix = (lower < 0) | (upper < 0)
        lower[fix] = diffusion[fix] + np.maximum(-r * i[fix], 0)
        upper[fix] = diffusion[fix] + np.maximum(r * i[fix], 0)
    diagonal = -lower - upper - r
    if not all(np.all(np.isfinite(a)) for a in (lower, diagonal, upper)):
        raise ValueError('generator outside finite range')
    return lower, diagonal, upper


@dataclass(frozen=True)
class GridResult:
    stock: np.ndarray
    values: np.ndarray
    previous_values: np.ndarray
    dt: float
    ds: float
    time_steps: int
    monotone: bool
    exercise_brackets: np.ndarray

    def price(self, spot):
        s = scalar(spot, 'spot')
        if not 0 <= s <= self.stock[-1]:
            raise ValueError('spot outside numerical domain')
        return float(np.interp(s, self.stock, self.values))


def boundaries(kind, strike, rate, tau, smax, american):
    discount_strike = strike * math.exp(-rate * tau)
    if kind == 'call':
        return 0., smax - discount_strike
    return (strike if american else discount_strike), 0.


def explicit_grid(strike, rate, volatility, horizon, smax, space_steps,
                  time_steps=None, *, kind='put', american=False,
                  drift='monotone', allow_unsafe=False):
    k = scalar(strike, 'strike', positive=True)
    r = scalar(rate, 'rate')
    sig = scalar(volatility, 'volatility', positive=True)
    tau = scalar(horizon, 'horizon', positive=True)
    ceiling = scalar(smax, 'upper stock boundary', positive=True)
    m = step_count(space_steps)
    if kind not in ('call', 'put') or ceiling <= k:
        raise ValueError('valid kind and Smax>K required')
    if not isinstance(american, bool) or not isinstance(allow_unsafe, bool):
        raise ValueError('exercise and diagnostic flags must be boolean')
    lower, diagonal, upper = coefficients(r, sig, m, drift)
    if time_steps is None:
        n = max(1, math.ceil(tau * float(np.max(-diagonal)) / .9))
    else:
        n = step_count(time_steps)
    dt = tau / n
    a, b, c = dt * lower, 1 + dt * diagonal, dt * upper
    monotone = bool(min(np.min(a), np.min(b), np.min(c)) >= 0)
    if not monotone and not allow_unsafe:
        raise ValueError('nonmonotone stencil: change drift or increase time steps')
    stock = np.linspace(0., ceiling, m + 1)
    ds = ceiling / m
    sign = 1 if kind == 'call' else -1
    intrinsic = np.maximum(sign * (stock - k), 0.)
    values = intrinsic.copy()
    previous = values.copy()
    brackets = []
    for level in range(1, n + 1):
        previous = values
        continuation = a * previous[:-2] + b * previous[1:-1] + c * previous[2:]
        values = np.empty(m + 1)
        values[0], values[-1] = boundaries(kind, k, r, level * dt, ceiling, american)
        if american:
            values[1:-1] = np.maximum(intrinsic[1:-1], continuation)
            if kind == 'put':
                exercised = ((intrinsic[1:-1] > 0)
                             & (intrinsic[1:-1] >= continuation))
                # Only positive-payoff exercise is identified. Far zero-payoff
                # equality must not be mistaken for the put's lower boundary.
                indices = np.flatnonzero(exercised) + 1
                last = int(indices[-1]) if indices.size else 0
                if indices.size and not np.array_equal(indices, np.arange(1, last + 1)):
                    raise ValueError('noncontiguous put exercise nodes in this grid')
                brackets.append((level * dt, stock[last], stock[last + 1]))
        else:
            values[1:-1] = continuation
        if not np.all(np.isfinite(values)):
            raise ValueError('nonfinite grid update')
    return GridResult(stock, values, previous, dt, ds, n, monotone,
                      np.array(brackets).reshape(-1, 3))


def grid_greeks(result, index):
    """Spatial delta/gamma and calendar theta at an interior grid node."""
    i = step_count(index)
    if i >= len(result.stock) - 1 or not result.monotone:
        raise ValueError('interior node of a monotone calculation required')
    v, h = result.values, result.ds
    delta = (v[i + 1] - v[i - 1]) / (2 * h)
    gamma = (v[i + 1] - 2 * v[i] + v[i - 1]) / h ** 2
    theta = (result.previous_values[i] - v[i]) / result.dt
    return float(delta), float(gamma), float(theta)


def source_grid_diagnostic():
    lower, diagonal, upper = coefficients(.045, .28, 100, 'central')
    dt = .75 / 300
    a, b, c = dt * lower, 1 + dt * diagonal, dt * upper
    matrix = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    return {'ds': 2.5, 'dt': dt, 'i25': (a[24], b[24], c[24]),
            'i25_sum': a[24] + b[24] + c[24], 'i99_center': b[98],
            'minimum_monotone_N': math.ceil(.75 * np.max(-diagonal)),
            'interior_spectral_radius': radius}


def refinement():
    rows = []
    for m in (100, 200, 400):
        european = explicit_grid(100, .1, .3, 1, 400, m)
        american = explicit_grid(100, .1, .3, 1, 400, m, american=True)
        rows.append((m, european.time_steps, european.price(100), american.price(100)))
    return rows


def main():
    print('Source50 global diagnostic:', source_grid_diagnostic())
    try:
        explicit_grid(90, .045, .28, .75, 250, 100, 300, drift='central')
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe source grid was accepted')
    unsafe = explicit_grid(90, .045, .28, .75, 250, 100, 300,
                           drift='central', allow_unsafe=True)
    print('UNSAFE diagnostic only: max abs value / interpolated spot90 value',
          np.max(np.abs(unsafe.values)), unsafe.price(90))
    assert not unsafe.monotone and np.max(np.abs(unsafe.values)) > 1e8
    safe = explicit_grid(90, .045, .28, .75, 250, 100, 1000, drift='central')
    print('Safe source50 grid put at90 / BS:', safe.price(90),
          black_scholes(90, 90, .045, .28, .75, 'put'))
    assert abs(safe.price(90) - black_scholes(90, 90, .045, .28, .75, 'put')) < .02
    print('Original refinement M/N/EUput/AMput:', refinement())
    for n in (1000, 2000):
        price = lattice(crr(100, .1, .3, 1, n), n, payoff('put', 100), american=True)
        print('Independent CRR grid-American put:', n, price)
    european = explicit_grid(100, .1, .3, 1, 400, 200)
    american = explicit_grid(100, .1, .3, 1, 400, 200, american=True)
    call = explicit_grid(100, .1, .3, 1, 400, 200, kind='call')
    am_call = explicit_grid(100, .1, .3, 1, 400, 200, kind='call', american=True)
    np.testing.assert_allclose(call.values, am_call.values, atol=2e-11, rtol=0)
    assert np.all(american.values >= np.maximum(100 - american.stock, 0))
    assert np.all(american.values >= european.values - 2e-11)
    assert american.values[0] == 100
    np.testing.assert_allclose(european.values[0], 100 * math.exp(-.1))
    print('Exercise-boundary last bracket:', american.exercise_brackets[-1])
    calculated = grid_greeks(european, 50)
    exact = greeks(100, 100, .1, .3, 1, 'put')
    print('Grid EUput delta/gamma/theta:', calculated)
    print('Analytic EUput delta/gamma/theta:', exact.delta, exact.gamma, exact.theta)
    np.testing.assert_allclose(calculated, (exact.delta, exact.gamma, exact.theta),
                               atol=.015, rtol=0)
    # Two repricings, identical stock/time grids, and N safe for sigma+.001.
    high = explicit_grid(100, .1, .301, 1, 400, 200)
    low = explicit_grid(100, .1, .299, 1, 400, 200, high.time_steps)
    vega = (high.price(100) - low.price(100)) / .002
    print('Repriced EUput vega / analytic:', vega, exact.vega)
    assert abs(vega - exact.vega) < .04
    am_high = explicit_grid(100, .1, .301, 1, 400, 200,
                            high.time_steps, american=True)
    am_low = explicit_grid(100, .1, .299, 1, 400, 200,
                           high.time_steps, american=True)
    print('Grid AMput delta/gamma/theta:', grid_greeks(american, 50))
    print('Repriced AMput vega:', (am_high.price(100) - am_low.price(100)) / .002)
    common_n = explicit_grid(100, .1, .3, 1, 500, 250).time_steps
    for ceiling in (300, 400, 500):
        # Both dS=2 and dt=1/common_n fixed while changing the domain.
        result = explicit_grid(100, .1, .3, 1, ceiling, ceiling // 2,
                               common_n, american=True)
        print('Fixed-dS/dt domain check Smax/M/N/AMput:', ceiling, ceiling // 2,
              result.time_steps, result.price(100))
    for params in ({'space_steps': True}, {'time_steps': 0}, {'rate': -.01},
                   {'volatility': 0}, {'kind': 'unknown'}, {'american': 1},
                   {'allow_unsafe': 1}, {'smax': 80}, {'horizon': float('nan')}):
        arguments = dict(strike=100, rate=.1, volatility=.3, horizon=1,
                         smax=400, space_steps=100)
        arguments.update(params)
        try:
            explicit_grid(**arguments)
        except ValueError:
            continue
        raise AssertionError(f'invalid parameters accepted: {params}')
    print('All stencil, stability, exercise, benchmark and Greek checks passed.')


if __name__ == '__main__':
    main()
