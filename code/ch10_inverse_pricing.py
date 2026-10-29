"""Inverse prices with conditional bracketing guarantees on declared fixtures.

The caller supplies continuity and reliable function signs. A floating computed
zero is reported as such, not as a proof of the exact mathematical root.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import (  # noqa: E402
    crr, lattice, payoff, scalar, step_count)


@dataclass(frozen=True)
class BracketResult:
    point: float
    residual: float
    left: float
    right: float
    f_left: float
    f_right: float
    iterations: int
    evaluations: int
    reason: str


def bisect_width(function, left, right, width_tol=1e-8, max_iter=100):
    """Stop on absolute bracket width; preserve computed endpoint signs.

    Conditional on continuity and correct signs, the midpoint is within half
    the reported width of some root. This is not SciPy's mixed tolerance API.
    """
    if not callable(function):
        raise ValueError('function must be callable')
    left, right = scalar(left, 'left'), scalar(right, 'right')
    width_tol = scalar(width_tol, 'width tolerance', positive=True)
    max_iter = step_count(max_iter)
    if left >= right:
        raise ValueError('left must be smaller than right')
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        return scalar(function(x), 'function value')

    fl, fr = evaluate(left), evaluate(right)
    for x, fx in ((left, fl), (right, fr)):
        if fx == 0:
            return BracketResult(x, fx, x, x, fx, fx, 0, evaluations,
                                 'evaluated_zero')
    if (fl > 0) == (fr > 0):
        raise ValueError('endpoint values must have opposite signs')
    iterations = 0
    while right - left > width_tol:
        if iterations >= max_iter:
            raise RuntimeError('iteration budget exhausted')
        middle = left / 2 + right / 2
        if not left < middle < right:
            raise ArithmeticError('no representable interior midpoint')
        fm = evaluate(middle)
        iterations += 1
        if fm == 0:
            return BracketResult(middle, fm, middle, middle, fm, fm,
                                 iterations, evaluations, 'evaluated_zero')
        if (fl > 0) != (fm > 0):
            right, fr = middle, fm
        else:
            left, fl = middle, fm
    middle = left / 2 + right / 2
    return BracketResult(middle, evaluate(middle), left, right, fl, fr,
                         iterations, evaluations, 'width_tolerance')


def coupon_bond(yield_rate):
    """Five-year face100, annual coupon7%, semiannual nominal annual yield."""
    y = scalar(yield_rate, 'yield')
    if y <= -2:
        raise ValueError('semiannual nominal yield must exceed -2')
    return math.fsum((3.5 + (100 if k == 10 else 0)) / (1 + y / 2) ** k
                     for k in range(1, 11))


def coupon_slope(yield_rate):
    y = scalar(yield_rate, 'yield')
    if y <= -2:
        raise ValueError('yield must exceed -2')
    return -math.fsum((k / 2) * (3.5 + (100 if k == 10 else 0))
                      / (1 + y / 2) ** (k + 1) for k in range(1, 11))


def lecture_call(volatility):
    """Declared source fixture, not an authenticated historical quote."""
    market = crr(24360, .065, volatility, 5 / 365, 50)
    return lattice(market, 50, payoff('call', 24400))


def main():
    quote = 98.4
    bond = bisect_width(lambda y: coupon_bond(y) - quote, 0, .3)
    vol = bisect_width(lambda sigma: lecture_call(sigma) - 125, .01, 1)
    print('Bond inversion:', bond)
    print('Volatility inversion:', vol)
    for y in (0, .3, .15, .075, .0375, .05625):
        print(f'Bond y={y:.8f}, price={coupon_bond(y):.9f}, '
              f'residual={coupon_bond(y)-quote:.9f}')
    for sigma in (.01, 1):
        print(f'IV bracket sigma={sigma:.2f}, '
              f'price={lecture_call(sigma):.9f}, '
              f'residual={lecture_call(sigma)-125:.9f}')
    np.testing.assert_allclose(coupon_bond(.07), 100, atol=2e-12)
    assert bond.left <= bond.point <= bond.right
    assert bond.right - bond.left <= 1e-8 and bond.iterations == 25
    assert vol.right - vol.left <= 1e-8 and vol.iterations == 27
    more_expensive = bisect_width(lambda y: coupon_bond(y) - 98.41, 0, .3)
    cheaper = bisect_width(lambda y: coupon_bond(y) - 98.39, 0, .3)
    slope = coupon_slope(bond.point)
    print(f'Bond local slope={slope:.9f}, dy/dP={1/slope:.12f}')
    print('Quote98.39–98.41 yield midpoint band:',
          more_expensive.point, cheaper.point,
          'width basis points=', (cheaper.point-more_expensive.point)*10000)
    h = 1e-5
    vega = (lecture_call(vol.point + h)
            - lecture_call(vol.point - h)) / (2 * h)
    print(f'IV finite-bump local price slope={vega:.9f}, '
          f'vol move for price0.1 approx={.1/vega:.12f}')
    flat_prices = []
    for sigma in (.1, .2):
        m = crr(100, .05, sigma, 1, 1)
        flat_prices.append(lattice(m, 1, payoff('call', 50)))
    np.testing.assert_allclose(flat_prices, 100 - 50 * math.exp(-.05))
    print('Flat one-step deep-ITM call for sigma.1/.2:', flat_prices)

    def signed(y):
        return 230 / (1 + y) - 132 / (1 + y) ** 2 - 100

    np.testing.assert_allclose([signed(.1), signed(.2)], 0, atol=3e-14)
    print('Signed-cashflow residual at0/.1/.2/.3:',
          [signed(y) for y in (0, .1, .2, .3)])
    # Width termination alone cannot discover a missing continuity assumption.
    jump = bisect_width(lambda x: -1 if x < 0 else 1, -1, 1)
    assert abs(jump.residual) == 1
    print('Discontinuous sign jump:', jump)
    for initial_width in (.3, .99, 1):
        required = math.ceil(math.log2(initial_width / 1e-8))
        print('Width/halvings:', initial_width, required)
    # Known closed-form inverses and endpoint termination check the solver.
    zero_coupon_yield = (100 / 95) ** (1 / 3) - 1
    zcb = bisect_width(lambda y: 100 / (1 + y) ** 3 - 95, 0, .2)
    assert (abs(zcb.point - zero_coupon_yield)
            <= (zcb.right - zcb.left) / 2 + 1e-15)
    square = bisect_width(lambda x: x*x - 2, 1, 2)
    assert abs(square.point - math.sqrt(2)) <= (square.right-square.left) / 2
    assert bisect_width(lambda x: x, 0, 1).reason == 'evaluated_zero'
    invalid = (lambda: bisect_width(lambda x: 1, 0, 1),
               lambda: bisect_width(lambda x: float('nan'), 0, 1),
               lambda: bisect_width(lambda x: x, 1, 0),
               lambda: bisect_width(lambda x: x, -1, 1, width_tol=0),
               lambda: bisect_width(lambda x: x*x - 2, 1, 2, max_iter=1),
               lambda: bisect_width(lambda x: x, -1, 1, max_iter=True))
    for function in invalid:
        try:
            function()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError('invalid root call should fail')
    print('All inverse-pricing, bracket and interpretation checks passed.')


if __name__ == '__main__':
    main()
