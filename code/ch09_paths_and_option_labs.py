"""Tiny exact-history markets and reproducible paired-language lab fixtures.

Asian averaging and barriers include time zero and every grid date. Barriers
hit on equality and pay no rebate. Prices use Q, not physical path weights.
Enumeration is deliberately capped; it is an explanatory reference algorithm.
"""
from fractions import Fraction
from functools import cache
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np

# Explicit sibling import also works when the validation runner uses Python -I.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import (  # noqa: E402
    Market, lattice, payoff, recursive, scalar, step_count)


def histories(market, steps):
    """Yield (ordered move tuple, observed prices, Q probability)."""
    steps = step_count(steps)
    if steps > 12:
        raise ValueError('exhaustive enumeration is capped at 12 steps')
    for moves in product((1, 0), repeat=steps):
        ups = 0
        prices = [market.spot]
        for level, move in enumerate(moves, 1):
            ups += move
            prices.append(market.node(level, ups))
        probability = market.q ** ups * (1 - market.q) ** (steps - ups)
        yield moves, tuple(prices), probability


def path_price(market, steps, function):
    """Discount a finite path-claim expectation, streaming over histories."""
    if not callable(function):
        raise ValueError('path payoff must be callable')
    steps = step_count(steps)
    terms = (prob * scalar(function(path), 'path payoff')
             for _, path, prob in histories(market, steps))
    return math.fsum(terms) / market.growth ** steps


def path_claims(path):
    """Fixed original contracts used by the chapter-9 four-path comparison."""
    terminal = path[-1]
    maximum = max(path)
    asian = max(math.fsum(path) / len(path) - 100, 0)
    vanilla = max(terminal - 90, 0)
    hit = maximum >= 115
    out = 0.0 if hit else vanilla
    inside = vanilla if hit else 0.0
    lookback = maximum - terminal
    return asian, out, inside, lookback


def exact_asian(spot, up, down, growth, strike, steps):
    """Fraction-valued augmented-state rollback: (level, up count, price sum).

    Inputs are exact Fraction/integer quantities. Return price and cache info;
    no claim of quadratic state count for an arithmetic running average.
    """
    steps = step_count(steps)
    if steps > 12:
        raise ValueError('exact teaching recursion capped at 12 steps')
    spot, up, down, growth, strike = map(
        Fraction, (spot, up, down, growth, strike))
    if spot <= 0 or not 0 < down < growth < up or strike < 0:
        raise ValueError('invalid exact market')
    q = (growth - down) / (up - down)

    @cache
    def visit(level, ups, total):
        if level == steps:
            return max(total / (steps + 1) - strike, Fraction(0))
        next_down = spot * up ** ups * down ** (level + 1 - ups)
        next_up = spot * up ** (ups + 1) * down ** (level - ups)
        return ((1 - q) * visit(level + 1, ups, total + next_down)
                + q * visit(level + 1, ups + 1, total + next_up)) / growth

    value = visit(0, 0, spot)
    return value, visit.cache_info()


def vector_prices(prices):
    values = np.asarray(prices, dtype=object)
    if values.ndim != 1:
        raise ValueError('prices must be one-dimensional')
    values = np.array([scalar(x, 'price') for x in values], dtype=float)
    if np.any(values < 0):
        raise ValueError('prices must be nonnegative')
    return values


def strategies(prices):
    """Vector long strangle, bull spread, and equal-width butterfly claims."""
    prices = vector_prices(prices)
    c90 = np.maximum(prices - 90, 0)
    c100 = np.maximum(prices - 100, 0)
    c110 = np.maximum(prices - 110, 0)
    return (np.maximum(90 - prices, 0) + c110,
            c90 - c110, c90 - 2 * c100 + c110)


def bond_price(rate):
    """Annual cash flows 5, 5, 105 at years 1, 2, 3; effective annual yield."""
    rate = scalar(rate, 'yield')
    if rate <= -1:
        raise ValueError('effective yield must exceed -1')
    cash, years = np.array([5., 5., 105.]), np.arange(1, 4)
    return float(cash @ (1 + rate) ** (-years.astype(float)))


def main():
    market = Market(100, 1.2, .8, 1.05)
    paths = list(histories(market, 2))
    np.testing.assert_allclose(sum(p for _, _, p in paths), 1)
    for moves, path, probability in paths:
        label = ''.join('U' if m else 'D' for m in moves)
        print(label, 'path=', path, 'Q=', probability,
              'Asian/out/in/lookback=', path_claims(path))
    prices = [path_price(market, 2, lambda p, j=j: path_claims(p)[j])
              for j in range(4)]
    print('Asian, out, in, lookback prices:', prices)
    exact, info = exact_asian(100, Fraction(6, 5), Fraction(4, 5),
                              Fraction(21, 20), 100, 2)
    np.testing.assert_allclose(float(exact), prices[0], atol=2e-13)
    print('Exact Asian fraction:', exact, 'state cache:', info)
    for n in (1, 2, 3, 4):
        exact, _ = exact_asian(100, Fraction(6, 5), Fraction(4, 5),
                               Fraction(21, 20), 100, n)
        enumerated = path_price(market, n, lambda p: path_claims(p)[0])
        np.testing.assert_allclose(float(exact), enumerated, atol=2e-12)
    np.testing.assert_allclose(prices[1] + prices[2],
                               lattice(market, 2, payoff('call', 90)))
    assert path_price(market, 2, lambda p: 0 if max(p) >= 100 else 1) == 0
    grid = np.arange(50, 151, 1)
    assert len(grid) == 101 and grid[0] == 50 and grid[-1] == 150
    c = np.maximum(grid - 100, 0)
    p = np.maximum(100 - grid, 0)
    np.testing.assert_allclose(c + p, abs(grid - 100))
    np.testing.assert_allclose(grid + p, np.maximum(grid, 100))
    np.testing.assert_allclose(grid - c, np.minimum(grid, 100))
    strangle, bull, butterfly = strategies([60, 90, 95, 100, 105, 110, 150])
    print('Strategy grid60,90,95,100,105,110,150:', strangle, bull, butterfly)
    np.testing.assert_allclose(bull, [0, 0, 5, 10, 15, 20, 20])
    np.testing.assert_allclose(butterfly, [0, 0, 5, 10, 5, 0, 0])
    for rate in (.04, .05, .06):
        loop = math.fsum(c / (1 + rate) ** t
                         for c, t in zip([5, 5, 105], [1, 2, 3], strict=True))
        np.testing.assert_allclose(loop, bond_price(rate))
        print(f'Bond yield={rate:.2f}, price={bond_price(rate):.9f}')
    for growth in (1.02, math.exp(.05 / 3)):
        m = Market(100, 1.1, .9, growth)
        call = lattice(m, 3, payoff('call', 100))
        put = lattice(m, 3, payoff('put', 100))
        np.testing.assert_allclose(call - put, 100 - 100 / growth ** 3)
        rec, calls = recursive(m, 3, payoff('call', 100))
        np.testing.assert_allclose(rec, call)
        assert calls == 15
        print(f'Lab G={growth:.9f}, call={call:.9f}, put={put:.9f}')
    for spot in (90, 100, 110):
        m = Market(spot, 1.1, .9, 1.02)
        print('Spot sensitivity:', spot, lattice(m, 3, payoff('call', 100)),
              lattice(m, 3, payoff('put', 100)))
    straddle_price = path_price(market, 2, lambda p: abs(p[-1] - 100))
    funded = straddle_price * market.growth ** 2
    q_win, physical_win, physical_mean = 0.0, 0.0, 0.0
    for _, path, q_prob in paths:
        claim = abs(path[-1] - 100)
        q_win += q_prob * (claim > funded)
        physical_win += .25 * (claim > funded)
        physical_mean += .25 * claim
    print(f'Straddle fair price={straddle_price:.9f}, funded={funded:.9f}, '
          f'Q win={q_win:.9f}, physical p=.5 win={physical_win:.9f}, '
          f'physical Epay={physical_mean:.9f}, '
          f'expected funded PnL={physical_mean-funded:.9f}')
    np.testing.assert_allclose([funded, q_win, physical_win, physical_mean],
                               [24.125, .53125, .5, 22])
    for operation in (lambda: vector_prices([80, True]),
                      lambda: vector_prices([[80, 100]]),
                      lambda: vector_prices([float('nan')])):
        try:
            operation()
        except ValueError:
            continue
        raise AssertionError('invalid vector should be rejected')
    print('All history, state, vector, bond, tree '
          'and probability checks passed.')


if __name__ == '__main__':
    main()
