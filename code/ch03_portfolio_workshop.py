"""Chapter 3: original checks for allocation, grids, and the week-one workshop."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math


def optimum(rf, mean, sigma, aversion):
    """Return one maximizing long-only weight for the teaching inputs."""
    if not all(math.isfinite(x) for x in (rf, mean, sigma, aversion)):
        raise ValueError("parameters must be finite")
    if sigma < 0 or aversion < 0:
        raise ValueError("sigma and aversion must be nonnegative")
    premium = mean - rf
    if sigma == 0 or aversion == 0:
        return 1.0 if premium > 0 else 0.0 if premium < 0 else 0.5
    return min(1.0, max(0.0, premium / (aversion * sigma**2)))


def objective(w, rf, mean, sigma, aversion):
    return rf + w * (mean - rf) - aversion * w*w * sigma*sigma / 2


def grid_weight(intervals, rf, mean, sigma, aversion):
    if not isinstance(intervals, int) or isinstance(intervals, bool):
        raise ValueError("intervals must be a positive integer")
    if intervals <= 0:
        raise ValueError("intervals must be a positive integer")
    return max((j / intervals for j in range(intervals + 1)),
               key=lambda w: objective(w, rf, mean, sigma, aversion))


def bond_price(nominal_yield):
    """Four semiannual payments: 3.5, 3.5, 3.5, and 103.5."""
    rate = F(nominal_yield) / 2
    return sum(F(7, 2) / (1 + rate)**k for k in range(1, 5)) + (
        F(100) / (1 + rate)**4)


def logarithm_polynomial(x, degree):
    if (not 0 <= x <= 1 or not isinstance(degree, int)
            or isinstance(degree, bool) or degree < 1):
        raise ValueError("need 0 <= x <= 1 and positive integer degree")
    return math.fsum((-1)**(k + 1) * x**k / k
                     for k in range(1, degree + 1))


def moments(distribution):
    mean = math.fsum(p * r for r, p in distribution)
    variance = math.fsum(p * (r - mean)**2 for r, p in distribution)
    return mean, variance


def checks():
    rf, mean, sigma, aversion = .05, .12, .20, 4.0
    exact = (F(3, 25) - F(1, 20)) / (4 * F(1, 5)**2)
    assert exact == F(7, 16)
    w = optimum(rf, mean, sigma, aversion)
    assert math.isclose(w, float(exact), abs_tol=1e-15)
    for intervals in (10, 100, 1000):
        wg = grid_weight(intervals, rf, mean, sigma, aversion)
        gap = objective(w, rf, mean, sigma, aversion) - objective(
            wg, rf, mean, sigma, aversion)
        identity = aversion * sigma**2 * (wg - w)**2 / 2
        bound = aversion * sigma**2 / (8 * intervals**2)
        assert abs(gap - identity) < 1e-15
        assert -1e-15 <= gap <= bound + 1e-15
    assert optimum(rf, .02, sigma, aversion) == 0
    assert optimum(rf, .30, sigma, aversion) == 1
    assert optimum(rf, mean, 0, aversion) == 1
    assert optimum(rf, rf, 0, aversion) == .5
    assert optimum(rf, mean, sigma, 0) == 1
    assert optimum(rf, .02, sigma, 0) == 0
    assert optimum(rf, rf, sigma, 0) == .5
    assert math.isclose(optimum(100*rf, 100*mean, 100*sigma,
                               aversion / 100), w)
    nu = rf
    assert abs(mean - aversion * sigma**2 * w - nu) < 1e-15
    assert (1 - w) + w == 1
    assert bond_price(F(9, 100)) < bond_price(F(8, 100)) < 100
    assert bond_price(F(7, 100)) == 100
    for bad_degree in (0, -1, .5, True):
        try:
            logarithm_polynomial(.5, bad_degree)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid degree accepted')
    for args in ((rf, mean, -1, aversion), (rf, mean, sigma, -1),
                 (rf, float('nan'), sigma, aversion)):
        try:
            optimum(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid parameter accepted')
    with localcontext() as context:
        context.prec = 80
        reference = Decimal('1.5').ln()
        for degree in range(1, 21):
            value = logarithm_polynomial(.5, degree)
            error = abs(Decimal.from_float(value) - reference)
            bound = Decimal('.5')**(degree+1) / (degree+1)
            assert error <= bound + Decimal('1e-16')
    print('checks: exact allocation, grid gap, bounds, units, multiplier,')
    print('        corner cases, invalid inputs, logarithm remainders: PASS')


def main():
    checks()
    rf, mean, sigma, aversion = .05, .12, .20, 4.0
    w = optimum(rf, mean, sigma, aversion)
    expected = rf + w * (mean - rf)
    variance = w*w * sigma*sigma
    print(f'baseline: risky={w:.8f}, safe={1-w:.8f}, ')
    print(f'          mean={expected:.8f}, variance={variance:.8f}, '
          f'std={w*sigma:.8f}')
    print(f'          objective={objective(w,rf,mean,sigma,aversion):.8f}')
    for intervals in (10, 100, 1000):
        wg = grid_weight(intervals, rf, mean, sigma, aversion)
        gap = objective(w, rf, mean, sigma, aversion) - objective(
            wg, rf, mean, sigma, aversion)
        bound = aversion * sigma**2 / (8 * intervals**2)
        print(f'grid M={intervals:4}: nodes={intervals+1:4}, '
              f'w={wg:.6f}, gap={gap:.9e}, bound={bound:.9e}')
    for lam in (0, .5, 1, 2, 4, 8):
        print(f'aversion={lam:3}: risky weight='
              f'{optimum(rf,mean,sigma,lam):.8f}')
    print(f'percent-return units: corrected lambda=.04 -> '
          f'w={optimum(5,12,20,.04):.8f}; unchanged lambda=4 -> '
          f'w={optimum(5,12,20,4):.8f}')
    first = [(.12-.2, .5), (.12+.2, .5)]
    second = [(.12-.4, .125), (.12, .75), (.12+.4, .125)]
    for i, distribution in enumerate((first, second), 1):
        m, v = moments(distribution)
        tail = sum(p for r, p in distribution if r < -.2)
        assert math.isclose(m, .12) and math.isclose(v, .04)
        print(f'moment-matched distribution {i}: mean={m:.8f}, '
              f'variance={v:.8f}, P(return < -.2)={tail:.8f}')
    for yield_rate in (F(8, 100), F(9, 100)):
        value = bond_price(yield_rate)
        print(f'workshop bond, nominal yield={float(yield_rate):.2%}: '
              f'price={float(value):.9f}')
    change = bond_price(F(9, 100)) - bond_price(F(8, 100))
    print(f'workshop +100bp price change={float(change):.9f}')
    components = [F(7, 2) / F(26, 25)**k for k in range(1, 5)]
    components[-1] += F(100) / F(26, 25)**4
    print('bond PV components:', ', '.join(f'{float(x):.9f}'
                                         for x in components))
    print(f'quote99 minus model={99-float(bond_price(F(8,100))):.9f}')
    for frequency in (1, 2, 12):
        amount = F(100) * (1 + F(8, 100) / frequency)**frequency
        print(f'100 at nominal8%, frequency={frequency}: '
              f'{float(amount):.9f}')
    print(f'100 at continuous8%: {100*math.exp(.08):.9f}')
    interpolated = F(6, 100) + F(1, 4) * (F(7, 100) - F(6, 100))
    print(f'workshop interpolated annual zero rate={float(interpolated):.8f}')
    for degree in (3, 4):
        value = logarithm_polynomial(.5, degree)
        reference = math.log1p(.5)
        bound = .5**(degree+1) / (degree+1)
        print(f'log(1.5), degree={degree}: value={value:.9f}, '
              f'reference={reference:.9f}, signed_error={value-reference:.9f}, '
              f'next_term_bound={bound:.9f}')
    for assets in (2, 3, 10):
        count = math.comb(100 + assets - 1, assets - 1)
        print(f'100-share simplex lattice, assets={assets}: {count} portfolios')
    print(f'interior sensitivity dw/dmean={1/(aversion*sigma**2):.9f}; '
          f'dw/dlambda={-(mean-rf)/(aversion**2*sigma**2):.9f}')
    print(f'interior lambda threshold for full risky holding='
          f'{(mean-rf)/sigma**2:.9f}')
    minimum_intervals = math.ceil(math.sqrt(aversion * sigma**2 / (8e-6)))
    print(f'grid gap target1e-6: M={minimum_intervals}, '
          f'nodes={minimum_intervals+1}, '
          f'bound={aversion*sigma**2/(8*minimum_intervals**2):.12e}')
    print(f'exercise means .02/.30: weights='
          f'{optimum(rf,.02,sigma,aversion):.8f}/'
          f'{optimum(rf,.30,sigma,aversion):.8f}')
    print(f'mean .13: weight={optimum(rf,.13,sigma,aversion):.8f}')


if __name__ == '__main__':
    main()
