"""European terminal claims and financed P&L; no option-price model.

All money is per one underlying share. Synthetic prices, no dividends,
transaction costs, default, or early exercise. Run with Python 3.13/NumPy 2.2.
"""
from collections.abc import Callable
import math

import numpy as np


def nonnegative_finite(value: float, name: str) -> float:
    """Accept a real finite scalar in [0, infinity), excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be a real scalar')
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def vanilla(kind: str, strike: float) -> Callable[[float], float]:
    """Return a scalar payoff function; validate its captured strike once."""
    if kind not in ('call', 'put'):
        raise ValueError('kind must be call or put')
    strike = nonnegative_finite(strike, 'strike')

    def payoff(terminal: float) -> float:
        terminal = nonnegative_finite(terminal, 'terminal price')
        difference = terminal - strike if kind == 'call' else strike - terminal
        return max(difference, 0.0)

    return payoff


def apply_to_grid(payoff: Callable[[float], float], grid) -> np.ndarray:
    """Evaluate a scalar callback on a finite nonnegative one-dimensional grid.

    Preserve order and require one finite scalar output per input. A general
    signed portfolio payoff is allowed; input validation never clips prices.
    """
    if not callable(payoff):
        raise ValueError('payoff must be callable')
    points = np.asarray(grid, dtype=object)
    if points.ndim != 1:
        raise ValueError('grid must be one-dimensional')
    points = [nonnegative_finite(x, 'grid price') for x in points]
    result = []
    for point in points:
        value = payoff(point)
        if (isinstance(value, (bool, np.bool_, str, bytes, complex,
                               np.complexfloating)) or not np.isscalar(value)):
            raise ValueError('callback must return a real scalar')
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('callback output must be finite')
        result.append(value)
    return np.asarray(result, dtype=float)


def running_case():
    """Return funded strategy costs, an implication of parity, and payoffs."""
    spot, strike, growth, call_price = 100.0, 100.0, 1.05, 5.0
    discount = 1.0 / growth
    put_price = call_price - spot + strike * discount
    call, put = vanilla('call', strike), vanilla('put', strike)
    premiums = np.array([call_price, put_price, call_price + put_price,
                         spot + put_price, spot - call_price])
    names = ('call', 'put', 'straddle', 'protective put', 'covered call')

    def wealth(terminal):
        c, p = call(terminal), put(terminal)
        return np.array([c, p, c + p, terminal + p, terminal - c])

    return names, premiums, growth, wealth


def main():
    names, premiums, growth, wealth = running_case()
    call, put = vanilla('call', 100), vanilla('put', 100)
    grid = np.array([0, 80, 94.5, 100, 104, 105.25, 105.5, 120.0])
    claims = np.stack([wealth(s) for s in grid])
    profits = claims - growth * premiums
    np.testing.assert_allclose(premiums[1], 5 / 21, atol=2e-14)
    np.testing.assert_allclose(growth * premiums,
                               [5.25, .25, 5.5, 105.25, 99.75])
    np.testing.assert_allclose(claims[:, 2], abs(grid - 100))
    np.testing.assert_allclose(claims[:, 3], np.maximum(grid, 100))
    np.testing.assert_allclose(claims[:, 4], np.minimum(grid, 100))
    np.testing.assert_allclose(grid + claims[:, 1], claims[:, 0] + 100)
    np.testing.assert_allclose(apply_to_grid(call, grid), claims[:, 0])
    np.testing.assert_allclose(apply_to_grid(put, grid), claims[:, 1])
    np.testing.assert_allclose(apply_to_grid(lambda s: call(s) + put(s), grid),
                               claims[:, 2])
    assert profits[4, 0] == -1.25  # In the money, yet a financed loss.
    np.testing.assert_allclose(wealth(94.5)[2] - growth * premiums[2], 0,
                               atol=2e-14)
    np.testing.assert_allclose(wealth(105.5)[2] - growth * premiums[2], 0,
                               atol=2e-14)
    # At a kink, centered finite differences converge to neither side slope.
    slopes = []
    for h in (1., .1, .01):
        slopes.append((call(100 + h) - call(100 - h)) / (2 * h))
    np.testing.assert_allclose(slopes, .5, atol=3e-13)
    # Independent grid checks of convexity and the Lipschitz bound.
    for left, right in ((80, 120), (95, 98), (101, 104)):
        for f in (call, put):
            assert (f(.3 * left + .7 * right)
                    <= .3 * f(left) + .7 * f(right) + 1e-12)
            assert abs(f(left) - f(right)) <= abs(left - right)
    # Put quoted at 1: sell expensive stock+put, buy call+strike bond.
    credit = 100 + 1 - (5 + 100 / growth)
    np.testing.assert_allclose(credit, 16 / 21, atol=2e-14)
    np.testing.assert_allclose(credit * growth, .8, atol=2e-14)
    net_claim = claims[:, 0] + 100 - grid - claims[:, 1]
    np.testing.assert_allclose(net_claim, 0, atol=2e-14)
    # Contract failures are rejected rather than silently clipped/broadcast.
    invalid = (lambda: vanilla('call', -1),
               lambda: call(float('nan')), lambda: call(float('inf')),
               lambda: call(-.01), lambda: call(True),
               lambda: apply_to_grid(call, [[80, 100]]),
               lambda: apply_to_grid(lambda s: [s], [100]),
               lambda: apply_to_grid(lambda s: float('nan'), [100]),
               lambda: apply_to_grid(call, ['100']),
               lambda: apply_to_grid(call, [80, True]))
    for operation in invalid:
        try:
            operation()
        except ValueError:
            continue
        raise AssertionError('invalid input was accepted')
    print('Strategies:', ', '.join(names))
    print('Initial costs:', np.array2string(premiums, precision=9))
    print('Funded terminal costs:',
          np.array2string(growth * premiums, precision=9))
    for s, row, pnl in zip(grid, claims, profits, strict=True):
        print(f'S_T={s:6.2f} wealth={row} financed_PnL={pnl}')
    print(f'Parity violation: credit={credit:.9f}, '
          f'terminal={credit*growth:.9f}')
    print('Centered call slope at strike:', slopes)
    print('All payoff, strategy, parity, and contract checks passed.')


if __name__ == '__main__':
    main()
