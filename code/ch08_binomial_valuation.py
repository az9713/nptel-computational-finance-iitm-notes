"""Finite-state replication and valuation on moderate teaching trees.

Synthetic no-dividend prices. 'american' means exercise at the model's grid
dates, not an exact continuously exercisable market. No physical probabilities
enter the price. Default rolling storage is linear in the step count.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


def scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'invalid {name}')
    return value


def step_count(steps):
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral):
        raise ValueError('steps must be a positive integer')
    if steps < 1:
        raise ValueError('steps must be a positive integer')
    return int(steps)


@dataclass(frozen=True)
class Market:
    spot: float
    up: float
    down: float
    growth: float

    def __post_init__(self):
        for name in ('spot', 'up', 'down', 'growth'):
            value = scalar(getattr(self, name), name, positive=True)
            object.__setattr__(self, name, value)
        if not self.down < self.growth < self.up:
            raise ValueError('strict no-arbitrage condition d < G < u failed')
        if not 0 < self.q < 1:
            raise ValueError('pricing weights not representably interior')

    @property
    def q(self):
        return (self.growth - self.down) / (self.up - self.down)

    def node(self, level, ups):
        exponent = (math.log(self.spot) + ups * math.log(self.up)
                    + (level - ups) * math.log(self.down))
        try:
            value = math.exp(exponent)
        except OverflowError as exc:
            raise ValueError('node exceeds floating-point range') from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError('node outside positive floating-point range')
        return value


def payoff(kind, strike):
    strike = scalar(strike, 'strike')
    if kind not in ('call', 'put') or strike < 0:
        raise ValueError('invalid kind or nonnegative strike')
    sign = 1 if kind == 'call' else -1
    return lambda s: max(sign * (s - strike), 0.0)


def claim_value(function, stock):
    if not callable(function):
        raise ValueError('claim must be callable')
    return scalar(function(stock), 'claim output')


def replicate(market, up_value, down_value):
    """Return shares, initial bank balance, and unique one-step claim price."""
    fu, fd = scalar(up_value, 'up claim'), scalar(down_value, 'down claim')
    delta = (fu - fd) / (market.spot * (market.up - market.down))
    bank = (market.up * fd - market.down * fu)
    bank /= market.growth * (market.up - market.down)
    value = delta * market.spot + bank
    if not all(math.isfinite(x) for x in (delta, bank, value)):
        raise ValueError('replication arithmetic outside finite range')
    return delta, bank, value


def lattice(market, steps, function, *, american=False, history=False):
    """Roll back an expiry payoff; optionally retain all levels for inspection.

    function(s) is a terminal claim; in American mode it is also the immediate
    exercise payment at every grid date. Signed European claims are permitted.
    Each retained level has nodes ordered by ascending number of up moves.
    """
    steps = step_count(steps)
    if not isinstance(american, bool) or not isinstance(history, bool):
        raise ValueError('american and history must be booleans')
    values = [claim_value(function, market.node(steps, j))
              for j in range(steps + 1)]
    if american and min(values) < 0:
        raise ValueError('exercise payments must be nonnegative')
    levels = [None] * (steps + 1) if history else None
    if history:
        levels[steps] = values.copy()
    for level in range(steps - 1, -1, -1):
        for j in range(level + 1):
            continuation = ((1 - market.q) * values[j]
                            + market.q * values[j + 1]) / market.growth
            if not math.isfinite(continuation):
                raise ValueError('rollback outside finite range')
            if american:
                exercise = claim_value(function, market.node(level, j))
                if exercise < 0:
                    raise ValueError('exercise payments must be nonnegative')
                continuation = max(exercise, continuation)
            values[j] = continuation
        values.pop()
        if history:
            levels[level] = values.copy()
    return (values[0], levels) if history else values[0]


def direct_sum(market, steps, function):
    """Small-tree binomial sum; independent check, not big-n code."""
    steps = step_count(steps)
    terms = [math.comb(steps, j) * market.q ** j
             * (1 - market.q) ** (steps - j)
             * claim_value(function, market.node(steps, j))
             for j in range(steps + 1)]
    return math.fsum(terms) / market.growth ** steps


def recursive(market, steps, function):
    """Naive tiny-tree demonstration; cap at 20 to bound redundant work."""
    steps = step_count(steps)
    if steps > 20:
        raise ValueError('naive demonstration restricted to at most 20 steps')
    calls = 0

    def visit(level, ups):
        nonlocal calls
        calls += 1
        if level == steps:
            return claim_value(function, market.node(level, ups))
        return ((1 - market.q) * visit(level + 1, ups)
                + market.q * visit(level + 1, ups + 1)) / market.growth

    return visit(0, 0), calls


def crr(spot, rate, volatility, horizon, steps):
    """Continuously compounded annual rate, volatility per sqrt(year)."""
    steps = step_count(steps)
    rate = scalar(rate, 'rate')
    volatility = scalar(volatility, 'volatility', positive=True)
    horizon = scalar(horizon, 'horizon', positive=True)
    dt = horizon / steps
    if abs(rate) * math.sqrt(dt) >= volatility:
        raise ValueError('CRR time step fails strict no-arbitrage condition')
    try:
        up = math.exp(volatility * math.sqrt(dt))
        down = math.exp(-volatility * math.sqrt(dt))
        growth = math.exp(rate * dt)
    except OverflowError as exc:
        raise ValueError('CRR factor outside floating-point range') from exc
    return Market(spot, up, down, growth)


def main():
    m = Market(100, 1.2, .9, 1.1)
    for kind in ('call', 'put'):
        f = payoff(kind, 110)
        delta, bank, value = replicate(m, f(120), f(90))
        np.testing.assert_allclose([delta * 120 + bank * 1.1,
                                   delta * 90 + bank * 1.1], [f(120), f(90)],
                                   atol=2e-13)
        np.testing.assert_allclose(value, lattice(m, 1, f), atol=2e-13)
        np.testing.assert_allclose(value, 200 / 33, atol=2e-13)
        print(f'One-step {kind}: delta={delta:.9f}, '
              f'B={bank:.9f}, V={value:.9f}')
    credit = 7 - 200 / 33
    print(f'Call quote7 arbitrage credit={credit:.9f}, '
          f'terminal={credit*1.1:.9f}')
    print(f'Physical p=.8: E payoff={.8*10:.9f}, discounted={.8*10/1.1:.9f}, '
          f'expected option return={.8*10/(200/33)-1:.9f}')
    three = crr(100, .1, .3, 1, 3)
    print(f'CRR3: u={three.up:.9f}, d={three.down:.9f}, '
          f'G={three.growth:.9f}, q={three.q:.9f}')
    print('CRR3 stock nodes:', [three.node(3, j) for j in range(4)])
    price, levels = lattice(three, 3, payoff('call', 100), history=True)
    print('CRR3 call levels:', levels)
    np.testing.assert_allclose(price,
                               direct_sum(three, 3, payoff('call', 100)))
    for n in (1, 2, 3, 6, 10):
        rec, calls = recursive(three, n, payoff('call', 100))
        rolling = lattice(three, n, payoff('call', 100))
        np.testing.assert_allclose(rec, rolling)
        assert calls == 2 ** (n + 1) - 1
    # Full week-3 workshop fixtures, separate from original running example.
    print('Mixed strike call130/put115:',
          [(s, payoff('call', 130)(s), payoff('put', 115)(s))
           for s in (145, 120, 108)])
    print('Protective K85:', [(s, s + payoff('put', 85)(s))
                              for s in (70, 85, 100)])
    print('Covered K95:', [(s, s - payoff('call', 95)(s))
                           for s in (80, 95, 110)])
    print(f'Workshop parity put={4.8-78+80*.96:.9f}, package={4.8+80*.96:.9f}')
    copper = Market(40, 1.15, .85, 1.04)
    print('Copper hedge and price:', replicate(copper, 4, 0), 'q=', copper.q)
    two = Market(50, 1.2, .85, 1.03)
    european_put = lattice(two, 2, payoff('put', 50))
    for kind, american in (('call', False), ('put', False), ('put', True)):
        result, levels = lattice(two, 2, payoff(kind, 50),
                                 american=american, history=True)
        print(f'Two-step {kind} American={american}: '
              f'V={result:.9f}, levels={levels}')
        if not american:
            np.testing.assert_allclose(
                result, direct_sum(two, 2, payoff(kind, 50)))
        else:
            assert result > european_put
            np.testing.assert_allclose(levels[1][0], 7.5, atol=2e-13)
            np.testing.assert_allclose(result, (1 - two.q) * 7.5 / 1.03)
    for n in (1, 2, 4, 20):
        f = payoff('call', 50)
        np.testing.assert_allclose(lattice(two, n, f, american=True),
                                   lattice(two, n, f), atol=2e-12)
    negative_rate = Market(100, 1.1, .9, .95)
    eu = lattice(negative_rate, 1, payoff('call', 50))
    am = lattice(negative_rate, 1, payoff('call', 50), american=True)
    np.testing.assert_allclose([eu, am], [100 - 50 / .95, 50])
    print(f'Negative rate: European call={eu:.9f}, grid-American={am:.9f}')
    for growth in (.9, 1.2, .8, 1.3):
        try:
            Market(100, 1.2, .9, growth)
        except ValueError:
            continue
        raise AssertionError('arbitrage boundary should be rejected')
    for invalid in (0, -1, True, 1.5):
        try:
            lattice(m, invalid, payoff('call', 100))
        except ValueError:
            continue
        raise AssertionError('invalid step count accepted')
    np.testing.assert_allclose(lattice(two, 2, lambda s: s - 50),
                               two.spot - 50 / two.growth ** 2)
    print('All replication, rollback, counting, workshop '
          'and boundary checks passed.')


if __name__ == '__main__':
    main()
