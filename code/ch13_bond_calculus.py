"""Fixed-cash-flow derivatives and declared sensitivity/settlement examples."""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import scalar, step_count  # noqa: E402
from ch10_inverse_pricing import bisect_width, coupon_bond  # noqa: E402


@dataclass(frozen=True)
class BondMeasures:
    price: float
    slope: float
    curvature: float
    macaulay: float
    modified: float
    convexity: float
    dv01: float


def bond_measures(yield_rate, cashflows, frequency=1, first_fraction=1):
    """Known fixed nonnegative flows at equally spaced future coupon dates.

    first_fraction is the fraction of a coupon period until the first payment.
    Yield is nominal annual, compounded frequency times per year.
    """
    y = scalar(yield_rate, 'yield')
    m = step_count(frequency)
    fraction = scalar(first_fraction, 'first fraction', positive=True)
    raw = np.asarray(cashflows, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise ValueError('cashflows must be a nonempty one-dimensional sequence')
    cfs = [scalar(v, 'cashflow') for v in raw]
    if min(cfs) < 0 or max(cfs) <= 0 or fraction > 1 or y <= -m:
        raise ValueError('nonnegative nonzero flows and admissible timing/yield needed')
    powers = [fraction+k for k in range(len(cfs))]
    base = 1+y/m
    pvs = [c/base**k for c, k in zip(cfs, powers, strict=True)]
    price = math.fsum(pvs)
    slope = -math.fsum((k/m)*pv/base for k, pv in zip(powers, pvs, strict=True))
    curvature = math.fsum((k*(k+1)/m**2)*pv/base**2
                          for k, pv in zip(powers, pvs, strict=True))
    mac = math.fsum((k/m)*pv for k, pv in zip(powers, pvs, strict=True))/price
    values = (price, slope, curvature, mac)
    if not all(math.isfinite(v) for v in values) or price <= 0:
        raise ValueError('bond arithmetic outside positive finite range')
    return BondMeasures(price, slope, curvature, mac, -slope/price,
                        curvature/price, -slope*1e-4)


def differences(function, x, step):
    """Three first-derivative stencils and the centered second derivative."""
    x, h = scalar(x, 'x'), scalar(step, 'step', positive=True)
    if not x-h < x < x+h:
        raise ArithmeticError('step does not produce distinct representable points')
    lower, center, upper = [scalar(function(t), 'function value')
                            for t in (x-h, x, x+h)]
    values = ((upper-center)/h, (center-lower)/h,
              (upper-lower)/(2*h), ((upper-center)+(lower-center))/h**2)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('difference arithmetic outside finite range')
    return dict(zip(('forward', 'backward', 'centered', 'second'), values,
                    strict=True))


def annual_workshop():
    return bond_measures(.055, [6, 6, 106])


def callable_zero(rate):
    """Deterministic flat continuous rate; issuer may pay100 at1 instead of110at2."""
    r = scalar(rate, 'rate')
    return math.exp(-r)*min(100, 110*math.exp(-r))


def error_sweep():
    reference = annual_workshop()

    def price(y):
        return bond_measures(y, [6, 6, 106]).price

    hs = np.logspace(-1, -11, 81)
    rows = []
    for h in hs:
        d = differences(price, .055, h)
        rows.append((h, abs(d['forward']/reference.slope-1),
                     abs(d['centered']/reference.slope-1),
                     abs(d['second']/reference.curvature-1)))
    return np.array(rows)


def main():
    for x in (1.3, 1.4, 1.5, 1.6, 1.7):
        print('Quartic x/value:', x, x**4)
    for h in (.2, .1):
        d = differences(lambda x: x**4, 1.5, h)
        print('Quartic h/stencils:', h, d)
        np.testing.assert_allclose(d['forward'], 13.5+13.5*h+6*h*h+h**3)
        np.testing.assert_allclose(d['backward'], 13.5-13.5*h+6*h*h-h**3)
        np.testing.assert_allclose(d['centered'], 13.5+6*h*h)
        np.testing.assert_allclose(d['second'], 27+2*h*h)
    bond = annual_workshop()
    print('Annual workshop:', bond)
    print('Annual PVs:', [6/1.055, 6/1.055**2, 106/1.055**3])

    def price(y):
        return bond_measures(y, [6, 6, 106]).price

    bumped = differences(price, .055, 1e-4)
    print('Annual h1e-4 Pminus/Pzero/Pplus:', [price(y) for y in (.0549, .055, .0551)])
    print('Annual finite modified/convexity:', -bumped['centered']/bond.price,
          bumped['second']/bond.price)
    for dy in (-.01, .01, .04):
        exact = price(.055+dy)
        linear = bond.price+bond.slope*dy
        quadratic = linear+.5*bond.curvature*dy**2
        print('Shock dy/exact/linear/quadratic/relativequadratic:',
              dy, exact, linear, quadratic, (quadratic-bond.price)/bond.price)
        print('Absolute errors linear/quadratic:', abs(linear-exact),
              abs(quadratic-exact))
    third_bound = math.fsum(k*(k+1)*(k+2)*c/1.055**(k+3)
                            for k, c in enumerate([6, 6, 106], 1))
    print('Onepercentagepoint quadratic remainder bound:', third_bound*.01**3/6)
    error = abs(bond.price+bond.slope*.01+.5*bond.curvature*.01**2-price(.065))
    assert error <= third_bound*.01**3/6
    y = bisect_width(lambda x: coupon_bond(x)-98.4, 0, .3, width_tol=1e-12).point
    semi = bond_measures(y, [3.5]*9+[103.5], 2)
    print('Fiveyear nominal yield/measures:', y, semi)
    np.testing.assert_allclose(semi.price, coupon_bond(y))
    np.testing.assert_allclose(semi.modified, semi.macaulay/(1+y/2))
    zero = bond_measures(.055, [0, 0, 100])
    assert zero.macaulay == 3
    first = bond_measures(.04, [104])
    total = 2*first.price+bond.price
    port_slope = 2*first.slope+bond.slope
    weights = (2*first.price/total, bond.price/total)
    print('Twoasset portfolio V/weights/Dmod/C:', total, weights,
          -port_slope/total, (2*first.curvature+bond.curvature)/total)
    np.testing.assert_allclose(-port_slope/total,
                               weights[0]*first.modified+weights[1]*bond.modified)
    dirty = bond_measures(y, [3.5]*9+[103.5], 2, first_fraction=1/6)
    accrued = 3.5*5/6
    clean = dirty.price-accrued
    print('Fixedsettlement dirty/AI/clean/dirtymodified/cleannormalized:',
          dirty.price, accrued, clean, dirty.modified, -dirty.slope/clean)

    def clean_price(x):
        return bond_measures(x, [3.5]*9+[103.5], 2, 1/6).price-accrued

    np.testing.assert_allclose(differences(clean_price, y, 1e-4)['centered'],
                               differences(lambda x: bond_measures(
                                   x, [3.5]*9+[103.5], 2, 1/6).price,
                                   y, 1e-4)['centered'])
    # Nonparallel continuous-zero-rate shifts need separate dated exposures.
    pvs = np.array([6*math.exp(-.04), 106*math.exp(-.06*3)])
    times = np.array([1, 3])
    partials = -times*pvs
    print('Two dated continuous-zero-rate partials:', partials)
    print('Parallel+.01 and twist(+.01,-.01) linearchanges:',
          partials@np.array([.01, .01]), partials@np.array([.01, -.01]))
    r = math.log(1.1)
    print('Callable switch rate/price:', r, callable_zero(r))
    for h in (.001, .0001):
        d = differences(callable_zero, r, h)
        print('Callable h/effectiveD/effectiveC:', h,
              -d['centered']/callable_zero(r), d['second']/callable_zero(r))
        assert d['second'] < 0
    sweep = error_sweep()
    for j, label in enumerate(('forward', 'centered', 'second'), 1):
        row = sweep[np.argmin(sweep[:, j])]
        print('Sweep sampled minimum:', label, 'h=', row[0], 'relativeerror=', row[j])
    # Known polynomials test stencil mechanics independently of bond calculations.
    np.testing.assert_allclose(list(differences(lambda x: 3*x+2, 1, .25).values()),
                               [3, 3, 3, 0])
    np.testing.assert_allclose(differences(lambda x: x*x, 1, .25)['second'], 2)
    invalid = (lambda: differences(math.exp, 1, 1e-20),
               lambda: differences(math.exp, 1, 0),
               lambda: bond_measures(-1, [6, 106]),
               lambda: bond_measures(.05, [6, -100]),
               lambda: bond_measures(.05, [True, 100]),
               lambda: bond_measures(.05, [100], first_fraction=2))
    for function in invalid:
        try:
            function()
        except (ValueError, ArithmeticError):
            continue
        raise AssertionError('invalid call must fail')
    print('All finite-difference, bond, settlement and scenario checks passed.')


if __name__ == '__main__':
    main()
