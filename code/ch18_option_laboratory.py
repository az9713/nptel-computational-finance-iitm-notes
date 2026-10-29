"""Complete European option lab with controlled numerical failure experiments.

The logarithmic terminal sum avoids separately overflowing binomial factors
for tested moderate trees (through 5000 steps). It is not a general large-n
binomial probability library: log-gamma subtraction still loses precision.
R/SciPy counterparts in the chapter are documentary, not executed here.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import scalar, step_count, crr, lattice, payoff  # noqa: E402
from ch16_black_scholes import black_scholes  # noqa: E402
from ch17_finite_differences import coefficients, boundaries, explicit_grid, GridResult  # noqa: E402


def logsumexp(terms):
    """Internal finite logarithms; empty nonnegative sum has log -infinity."""
    if not terms:
        return -math.inf
    largest = max(terms)
    return largest + math.log(math.fsum(math.exp(x - largest) for x in terms))


@dataclass(frozen=True)
class TerminalResult:
    price: float
    probability_mass: float
    steps: int


def binomial_european(spot, strike, rate, volatility, horizon, steps, kind='call'):
    """O(n) terminal European expectation using log weights/payments.

    The probability mass is reported, never silently normalized to one.
    Interior positive spot/strike/volatility/horizon and strict CRR no arbitrage.
    """
    s = scalar(spot, 'spot', positive=True)
    k = scalar(strike, 'strike', positive=True)
    r = scalar(rate, 'rate')
    sig = scalar(volatility, 'volatility', positive=True)
    tau = scalar(horizon, 'horizon', positive=True)
    n = step_count(steps)
    if kind not in ('call', 'put'):
        raise ValueError('kind must be call or put')
    market = crr(s, r, sig, tau, n)
    logq, log1q = math.log(market.q), math.log1p(-market.q)
    shock = sig * math.sqrt(tau / n)
    logs, terms = [], []
    for j in range(n + 1):
        log_weight = (math.lgamma(n + 1) - math.lgamma(j + 1)
                      - math.lgamma(n - j + 1) + j * logq + (n - j) * log1q)
        logs.append(log_weight)
        x = math.log(s) - math.log(k) + (2 * j - n) * shock
        if kind == 'call' and x > 0:
            log_payment = math.log(k) + x + math.log(-math.expm1(-x))
        elif kind == 'put' and x < 0:
            log_payment = math.log(k) + math.log(-math.expm1(x))
        else:
            continue
        terms.append(log_weight + log_payment)
    try:
        price = math.exp(logsumexp(terms) - r * tau)
        mass = math.exp(logsumexp(logs))
    except OverflowError as exc:
        raise ValueError('terminal expectation outside finite range') from exc
    if not all(math.isfinite(v) for v in (price, mass)):
        raise ValueError('nonfinite terminal result')
    return TerminalResult(price, mass, n)


def naive_diagnostic(steps):
    """Deliberately bad floating-coefficient experiment on the source call.

    Overrange integer coefficients are represented by inf to reproduce the
    floating intermediate failure, not to return a trustworthy price.
    """
    n = step_count(steps)
    market = crr(100, .1, .3, 1, n)
    combs = []
    for j in range(n + 1):
        try:
            combs.append(float(math.comb(n, j)))
        except OverflowError:
            combs.append(math.inf)
    js = np.arange(n + 1)
    shock = .3 / math.sqrt(n)
    payments = np.maximum(100 * np.exp((2 * js - n) * shock) - 100, 0)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        weights = np.array(combs) * market.q ** js * (1 - market.q) ** (n - js)
        result = float(math.exp(-.1) * np.sum(weights * payments))
    return {'steps': n, 'overflowed_coefficients': int(np.sum(np.isinf(combs))),
            'finite_price': math.isfinite(result), 'diagnostic_value': result,
            'up_minus_down': market.up - market.down,
            'log10_central_coefficient': math.log10(math.comb(n, n // 2))}


def put_boundary_fixture(*, wrong=False):
    """One controlled fixture; only the boundary contract changes.

    The deliberately wrong run uses call boundaries with a put payoff.
    Holding generator, grids and all other inputs fixed isolates that defect.
    """
    if not isinstance(wrong, bool):
        raise ValueError('wrong must be boolean')
    k, r, sig, tau, ceiling, m, n = 100., .1, .3, 1., 200., 100, 1000
    lower, diagonal, upper = coefficients(r, sig, m)
    dt, ds = tau / n, ceiling / m
    a, b, c = dt * lower, 1 + dt * diagonal, dt * upper
    assert min(np.min(a), np.min(b), np.min(c)) >= 0
    stock = np.linspace(0., ceiling, m + 1)
    values = np.maximum(k - stock, 0.)
    previous = values.copy()
    for level in range(1, n + 1):
        previous = values
        values = np.empty(m + 1)
        values[1:-1] = a * previous[:-2] + b * previous[1:-1] + c * previous[2:]
        values[0], values[-1] = boundaries('call' if wrong else 'put', k, r,
                                           level * dt, ceiling, False)
    return GridResult(stock, values, previous, dt, ds, n, True, np.empty((0, 3)))


def lab_table():
    rows = []
    for n in (3, 30, 300, 1000, 1200, 1500, 5000):
        call = binomial_european(100, 100, .1, .3, 1, n)
        put = binomial_european(100, 100, .1, .3, 1, n, 'put')
        rows.append((n, call.price, put.price, call.probability_mass))
    return rows


def main():
    print('Full laboratory n/logsum call/logsum put/probabilitymass:', lab_table())
    for n in (3, 30, 300, 1000):
        market = crr(100, .1, .3, 1, n)
        for kind in ('call', 'put'):
            terminal = binomial_european(100, 100, .1, .3, 1, n, kind)
            rollback = lattice(market, n, payoff(kind, 100))
            np.testing.assert_allclose(terminal.price, rollback, atol=2e-9, rtol=0)
            assert abs(terminal.probability_mass - 1) < 2e-10
    for n, call, put, mass in lab_table():
        np.testing.assert_allclose(call - put, 100 - 100 * math.exp(-.1),
                                   atol=2e-8, rtol=0)
        assert abs(mass - 1) < 2e-10
    for n in (1000, 1020, 1029, 1030, 1200, 1500):
        diagnostic = naive_diagnostic(n)
        print('NAIVE intermediate diagnostic:', diagnostic)
        assert diagnostic['finite_price'] == (n <= 1029)
    correct = put_boundary_fixture()
    wrong = put_boundary_fixture(wrong=True)
    reference = explicit_grid(100, .1, .3, 1, 200, 100, 1000)
    np.testing.assert_allclose(correct.values, reference.values, atol=2e-12, rtol=0)
    print('Correct versus call-for-put boundary fixture prices:',
          correct.price(100), wrong.price(100))
    print('Correct versus wrong endpoints:', correct.values[[0, -1]],
          wrong.values[[0, -1]])
    assert abs(correct.values[0] - 100 * math.exp(-.1)) < 2e-13
    assert correct.values[-1] == 0
    assert wrong.values[0] == 0 and wrong.values[-1] > 100
    assert wrong.price(100) - correct.price(100) > 2
    for kind in ('call', 'put'):
        grid = explicit_grid(100, .1, .3, 1, 200, 100, 1000, kind=kind)
        analytic = black_scholes(100, 100, .1, .3, 1, kind)
        print('Source-like PDE / analytic / signed error:', kind, grid.price(100),
              analytic, grid.price(100) - analytic)
    for m, n in ((100, 1000), (200, 1000), (200, 3961)):
        try:
            grid = explicit_grid(100, .1, .3, 1, 200, m, n, kind='call')
        except ValueError as exc:
            print('PDE refinement rejected:', m, n, str(exc))
            assert (m, n) == (200, 1000)
        else:
            print('PDE refinement accepted:', m, n, grid.price(100))
    for invalid in (0, True, 2.5, -1):
        try:
            binomial_european(100, 100, .1, .3, 1, invalid)
        except ValueError:
            continue
        raise AssertionError('invalid step count accepted')
    for params in ({'strike': 0}, {'spot': True}, {'volatility': 0},
                   {'kind': 'unknown'}, {'horizon': float('nan')}):
        arguments = dict(spot=100, strike=100, rate=.1, volatility=.3,
                         horizon=1, steps=30)
        arguments.update(params)
        try:
            binomial_european(**arguments)
        except ValueError:
            continue
        raise AssertionError(f'invalid input accepted: {params}')
    print('All terminal-sum, laboratory, boundary and refinement checks passed.')


if __name__ == '__main__':
    main()
