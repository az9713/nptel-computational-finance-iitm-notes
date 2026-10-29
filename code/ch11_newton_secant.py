"""Newton, cached secant, and a transparent safeguarded teaching hybrid.

Residual tolerances have function-value units. The hybrid is not Brent's full
algorithm. Its central-half acceptance guarantees contraction of a valid
sign bracket unless a residual/zero stop occurs, subject to reliable signs.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import (  # noqa: E402
    crr, lattice, payoff, scalar, step_count)
from ch10_inverse_pricing import (  # noqa: E402
    bisect_width, coupon_bond, coupon_slope)


@dataclass
class Result:
    point: float
    residual: float
    updates: int
    function_evaluations: int
    derivative_evaluations: int
    reason: str
    trace: list
    bracket: tuple | None = None


def parameter(x, admissible):
    x = scalar(x, 'parameter')
    if admissible is not None and not admissible(x):
        raise ValueError('proposal outside the declared parameter domain')
    return x


def newton(function, derivative, start, *, ftol=1e-10, max_iter=30,
           admissible=None):
    ftol = scalar(ftol, 'residual tolerance', positive=True)
    max_iter = step_count(max_iter)
    if not callable(function) or not callable(derivative):
        raise ValueError('function and derivative must be callable')
    x = parameter(start, admissible)
    fx = scalar(function(x), 'function value')
    trace = [{'x': x, 'f': fx, 'method': 'initial'}]
    for update in range(max_iter + 1):
        if abs(fx) <= ftol:
            return Result(x, fx, update, update + 1, update,
                          'residual_tolerance', trace)
        if update == max_iter:
            break
        dfx = scalar(derivative(x), 'derivative')
        if dfx == 0:
            raise ArithmeticError('evaluated derivative is zero')
        trial = parameter(x - fx / dfx, admissible)
        if trial == x:
            raise ArithmeticError('parameter stagnated above tolerance')
        x, fx = trial, scalar(function(trial), 'function value')
        trace.append({'x': x, 'f': fx, 'method': 'newton'})
    raise RuntimeError('Newton update budget exhausted')


def secant(function, first, second, *, ftol=1e-10, max_iter=30,
           admissible=None):
    ftol = scalar(ftol, 'residual tolerance', positive=True)
    max_iter = step_count(max_iter)
    if not callable(function):
        raise ValueError('function must be callable')
    x0, x1 = parameter(first, admissible), parameter(second, admissible)
    if x0 == x1:
        raise ValueError('secant starts must differ')
    f0 = scalar(function(x0), 'function value')
    trace = [{'x': x0, 'f': f0, 'method': 'initial'}]
    if abs(f0) <= ftol:
        return Result(x0, f0, 0, 1, 0, 'residual_tolerance', trace)
    f1 = scalar(function(x1), 'function value')
    trace.append({'x': x1, 'f': f1, 'method': 'initial'})
    for update in range(max_iter + 1):
        if abs(f1) <= ftol:
            return Result(x1, f1, update, update + 2, 0,
                          'residual_tolerance', trace)
        if update == max_iter:
            break
        denominator = f1 - f0
        scale = max(abs(f0), abs(f1))
        if (not math.isfinite(denominator) or denominator == 0
                or abs(denominator) <= 8 * np.finfo(float).eps * scale):
            raise ArithmeticError('unresolved secant function difference')
        trial = parameter(x1 - (f1 / denominator) * (x1 - x0), admissible)
        if trial == x1:
            raise ArithmeticError('secant parameter stagnated')
        x0, f0 = x1, f1
        x1, f1 = trial, scalar(function(trial), 'function value')
        trace.append({'x': x1, 'f': f1, 'method': 'secant'})
    raise RuntimeError('secant update budget exhausted')


def hybrid(function, left, right, *, derivative=None, ftol=1e-10,
           width_tol=1e-10, max_iter=100):
    """Accept central-half proposals; otherwise use the midpoint."""
    left, right = scalar(left, 'left'), scalar(right, 'right')
    ftol = scalar(ftol, 'residual tolerance', positive=True)
    width_tol = scalar(width_tol, 'width tolerance', positive=True)
    max_iter = step_count(max_iter)
    if not callable(function) or (derivative is not None
                                  and not callable(derivative)):
        raise ValueError('function/derivative must be callable')
    if left >= right:
        raise ValueError('ordered endpoints required')
    fl, fr = scalar(function(left), 'function value'), scalar(
        function(right), 'function value')
    evaluations, derivatives, trace = 2, 0, []
    for x, fx in ((left, fl), (right, fr)):
        if fx == 0:
            return Result(x, fx, 0, evaluations, 0,
                          'evaluated_zero', [], (x, x))
    if (fl > 0) == (fr > 0):
        raise ValueError('opposite endpoint signs required')
    for update in range(max_iter + 1):
        if right - left <= width_tol:
            x = left / 2 + right / 2
            fx = scalar(function(x), 'function value')
            return Result(x, fx, update, evaluations + 1, derivatives,
                          'width_tolerance', trace, (left, right))
        best, fb = (left, fl) if abs(fl) < abs(fr) else (right, fr)
        if abs(fb) <= ftol:
            return Result(best, fb, update, evaluations, derivatives,
                          'residual_tolerance', trace, (left, right))
        if update == max_iter:
            break
        before = (left, right)
        width = right - left
        candidate, method = float('nan'), 'secant'
        if derivative is not None:
            derivatives += 1
            try:
                slope = scalar(derivative(best), 'derivative')
                if slope != 0:
                    candidate, method = best - fb / slope, 'newton'
            except (ValueError, ArithmeticError):
                pass  # Reject an unusable derivative proposal, not the price.
        else:
            difference = fr - fl
            if math.isfinite(difference) and difference != 0:
                candidate = right - (fr / difference) * width
        if (not math.isfinite(candidate)
                or not left + width / 4 <= candidate <= right - width / 4):
            candidate, method = left / 2 + right / 2, 'bisect'
        if not left < candidate < right:
            raise ArithmeticError('no representable interior trial')
        fx = scalar(function(candidate), 'function value')
        evaluations += 1
        if fx == 0:
            trace.append({'x': candidate, 'f': fx, 'method': method,
                          'before': before, 'after': (candidate, candidate)})
            return Result(candidate, fx, update + 1, evaluations, derivatives,
                          'evaluated_zero', trace, (candidate, candidate))
        if (fl > 0) != (fx > 0):
            right, fr = candidate, fx
        else:
            left, fl = candidate, fx
        trace.append({'x': candidate, 'f': fx, 'method': method,
                      'before': before, 'after': (left, right)})
    raise RuntimeError('hybrid update budget exhausted')


def workshop_bond(y, *, slope=False):
    y = scalar(y, 'yield')
    if y <= -2:
        raise ValueError('nominal semiannual yield must exceed -2')
    terms = []
    for k in range(1, 9):
        cash = 3 + (100 if k == 8 else 0)
        value = cash / (1 + y / 2) ** k
        terms.append(-(k / 2) * value / (1 + y / 2) if slope else value)
    return math.fsum(terms)


def workshop_call(sigma, strike=20200):
    return lattice(crr(20000, .055, sigma, 60 / 365, 40), 40,
                   payoff('call', strike))


def describe(name, result):
    print(name, 'point=', result.point, 'residual=', result.residual,
          'updates=', result.updates, 'f calls=', result.function_evaluations,
          'df calls=', result.derivative_evaluations, 'reason=', result.reason,
          'bracket=', result.bracket)
    for i, row in enumerate(result.trace[:6]):
        print(' ', i, row)


def main():
    def original(y):
        return coupon_bond(y) - 98.4

    original_result = newton(original, coupon_slope, .05,
                             admissible=lambda y: y > -2)
    describe('Five-year bond Newton', original_result)

    def bond(y):
        return workshop_bond(y) - 101.75

    def bond_slope(y):
        return workshop_bond(y, slope=True)

    bond_newton = newton(bond, bond_slope, .04, ftol=1e-8,
                         admissible=lambda y: y > -2)
    describe('Four-year workshop Newton', bond_newton)
    bond_bisect = bisect_width(bond, 0, .2)
    print('Four-year bisection:', bond_bisect)
    for y in (0, .2, .1, .05):
        print(f'Workshop bond y={y:.4f}, F={bond(y):.12f}')

    def option(sigma):
        return workshop_call(sigma) - 576.77

    sigma_min = .055 * math.sqrt(60 / 365 / 40)
    iv_secant = secant(option, .12, .25, ftol=1e-8,
                       admissible=lambda sigma: sigma > sigma_min)
    describe('Workshop IV secant', iv_secant)
    print('Workshop IV bisection:', bisect_width(option, .05, .4))
    for name, f, a, b, df in (('bond', bond, 0, .2, bond_slope),
                              ('IV', option, .05, .4, None)):
        result = hybrid(f, a, b, derivative=df, ftol=1e-8)
        describe('Hybrid '+name, result)
        for row in result.trace:
            old = row['before'][1] - row['before'][0]
            width = row['after'][1] - row['after'][0]
            assert width <= .75 * old + 1e-15
    # Generate an internally consistent paired-strike sensitivity comparison.
    for strike in (20200, 24000):
        quote = workshop_call(.18, strike)
        h = 1e-5
        slope = (workshop_call(.18+h, strike)
                 - workshop_call(.18-h, strike)) / (2*h)
        low = bisect_width(lambda s: workshop_call(s, strike) - (quote-1),
                           .05, .4)
        high = bisect_width(lambda s: workshop_call(s, strike) - (quote+1),
                            .05, .4)
        print(f'Paired K={strike}, quote={quote:.12f}, '
              f'local slope={slope:.9f}, '
              f'1/slope={1/slope:.12f}, price±1 IV midpoints={low.point:.12f},'
              f'{high.point:.12f}')
    kink = math.log(24000 / 20000) / (16 * math.sqrt(60 / 365 / 40))
    assert low.point < kink < .18
    print(f'Far-strike terminal-node kink, j28 of40: sigma={kink:.12f}')
    # Nonzero derivatives alone do not prevent a Newton two-cycle.

    def cycle(x):
        return x**3 - 2*x + 2

    cycle_values = [0.0]
    for _ in range(4):
        x = cycle_values[-1]
        cycle_values.append(x - cycle(x) / (3*x*x - 2))
    assert cycle_values == [0., 1., 0., 1., 0.]
    print('Newton cycle:', cycle_values)
    describe('Cycle function safeguarded', hybrid(cycle, -2, -1))
    repeated = newton(lambda x: x*x, lambda x: 2*x, 1, ftol=1e-12)
    print('Repeated root:', repeated.point, repeated.residual,
          repeated.updates,
          'first errors:', [row['x'] for row in repeated.trace[:5]])
    scaled = newton(lambda y: 1e-12*bond(y), lambda y: 1e-12*bond_slope(y),
                    .04, ftol=1e-20, admissible=lambda y: y > -2)
    np.testing.assert_allclose(scaled.point, bond_newton.point, atol=1e-14)
    expected_failures = (
        lambda: newton(cycle, lambda x: 3*x*x-2, 0, max_iter=6),
        lambda: newton(lambda x: x*x-1, lambda x: 2*x, 0),
        lambda: secant(lambda x: x*x-1, -2, 2),
        lambda: newton(original, coupon_slope, 2, admissible=lambda y: y > -2))
    for operation in expected_failures:
        try:
            operation()
        except (ValueError, ArithmeticError, RuntimeError) as error:
            print('Expected failure:', error)
            continue
        raise AssertionError('expected solver failure was not reported')
    for result in (newton(lambda x: x*x-2, lambda x: 2*x, 1),
                   secant(lambda x: x*x-2, 1, 2),
                   hybrid(lambda x: x*x-2, 1, 2)):
        np.testing.assert_allclose(result.point, math.sqrt(2), atol=1e-10,
                                   rtol=0)
    print('All Newton, secant, safeguarded, workshop '
          'and sensitivity checks passed.')


if __name__ == '__main__':
    main()
