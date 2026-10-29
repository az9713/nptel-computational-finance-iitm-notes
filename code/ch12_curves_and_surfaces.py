"""Declared curve and option-surface fixtures, not observed market quotes.

Exact-date bootstrap, log-discount interpolation, price-space shape checks,
and implied parameters from a common finite martingale mixture.
"""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import (  # noqa: E402
    Market, crr, lattice, payoff, scalar, step_count)
from ch10_inverse_pricing import bisect_width  # noqa: E402


def vector(values, name):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise ValueError(f'{name} must be a nonempty one-dimensional sequence')
    return np.array([scalar(x, name) for x in raw], dtype=float)


def bond_price(yield_rate, periods, coupon, face=100):
    y = scalar(yield_rate, 'yield')
    periods = step_count(periods)
    coupon = scalar(coupon, 'coupon')
    face = scalar(face, 'face', positive=True)
    if y <= -2 or coupon < 0:
        raise ValueError('yield must exceed -2 and coupon be nonnegative')
    return math.fsum((coupon + (face if k == periods else 0))
                     / (1 + y / 2) ** k for k in range(1, periods + 1))


def bootstrap(prices, coupons, face=100):
    """One bond per consecutive half-year; every bond pays at all prior dates."""
    prices, coupons = vector(prices, 'price'), vector(coupons, 'coupon')
    face = scalar(face, 'face', positive=True)
    if len(prices) != len(coupons) or min(prices) <= 0 or min(coupons) < 0:
        raise ValueError('positive prices and matching nonnegative coupons needed')
    discounts = []
    for price, coupon in zip(prices, coupons, strict=True):
        value = (price - coupon * math.fsum(discounts)) / (face + coupon)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('bootstrap implies a nonpositive discount factor')
        discounts.append(value)
    return np.array(discounts)


def log_discount(time, times, discounts):
    """Scalar interpolation on a strictly increasing positive-date grid only."""
    t = scalar(time, 'time')
    ts, ds = vector(times, 'times'), vector(discounts, 'discounts')
    if (len(ts) != len(ds) or ts[0] <= 0 or np.any(np.diff(ts) <= 0)
            or np.any(ds <= 0) or not 0 <= t <= ts[-1]):
        raise ValueError('ordered positive dates/discounts and in-range time needed')
    return math.exp(float(np.interp(t, np.r_[0, ts], np.r_[0, np.log(ds)])))


def call_slice_checks(strikes, prices, *, spot=100, discount=1, tol=1e-10):
    """Necessary finite-grid constraints for no-dividend European calls."""
    ks, cs = vector(strikes, 'strikes'), vector(prices, 'prices')
    spot = scalar(spot, 'spot', positive=True)
    discount = scalar(discount, 'discount', positive=True)
    tol = scalar(tol, 'tolerance')
    if (len(ks) != len(cs) or len(ks) < 3 or ks[0] < 0
            or np.any(np.diff(ks) <= 0) or tol < 0):
        raise ValueError('at least three matching ordered strikes needed')
    slopes = np.diff(cs) / np.diff(ks)
    return {
        'bounds': bool(np.all(cs >= np.maximum(spot-discount*ks, 0)-tol)
                       and np.all(cs <= spot+tol)),
        'verticals': bool(np.all(slopes >= -discount-tol)
                          and np.all(slopes <= tol)),
        'convex': bool(np.all(np.diff(slopes) >= -tol)),
        'slopes': slopes,
    }


def mixture_call(strike, months):
    months = step_count(months)
    f = payoff('call', strike)
    return (.75 * lattice(Market(100, 1.1, .9, 1), months, f)
            + .25 * lattice(Market(100, 1.2, .8, 1), months, f))


def implied_call(quote, strike, horizon, steps, *, spot=100, rate=0,
                 bracket=(.01, 2)):
    quote = scalar(quote, 'quote')
    f = payoff('call', strike)

    def residual(sigma):
        return lattice(crr(spot, rate, sigma, horizon, steps), steps, f) - quote

    return bisect_width(residual, *bracket, width_tol=1e-10)


def source_lab_table():
    years = (1, 2, 3, 5, 10)
    coupons = (3, 3.25, 3.4, 3.5, 3.75)
    declared_yields = (.06, .062, .064, .068, .075)
    rows = []
    for t, coupon, y in zip(years, coupons, declared_yields, strict=True):
        price = bond_price(y, 2*t, coupon)
        result = bisect_width(lambda x: bond_price(x, 2*t, coupon)-price,
                              0, .3, width_tol=1e-10)
        assert abs(result.point-y) <= (result.right-result.left)/2 + 2e-14
        rows.append((t, coupon, price, result.point, result.residual))
    return rows


def surface_rows():
    strikes = np.arange(70, 131, 10)
    rows = []
    for months in (6, 12, 24):
        prices = np.array([mixture_call(k, months) for k in strikes])
        checks = call_slice_checks(strikes, prices)
        assert all(checks[key] for key in ('bounds', 'verticals', 'convex'))
        vols = np.array([implied_call(c, k, months/12, months).point
                         for k, c in zip(strikes, prices, strict=True)])
        rows.append((months/12, strikes.copy(), prices, vols))
    for earlier, later in zip(rows, rows[1:]):
        assert np.all(later[2] >= earlier[2]-1e-11)
    return rows


def interpolate_surface(strike, horizon, rows):
    """Known common-strike-grid fixture: linear in price, then linear in time.

    Caller supplies validated surface_rows(), whose dates/strike grid agree.
    Extrapolation is rejected. This is not an arbitrary market-table importer.
    """
    k, t = scalar(strike, 'strike'), scalar(horizon, 'horizon')
    dates = [row[0] for row in rows]
    strikes = rows[0][1]
    if not strikes[0] <= k <= strikes[-1] or not dates[0] <= t <= dates[-1]:
        raise ValueError('surface query must stay inside the fixture rectangle')
    prices_at_k = [np.interp(k, row[1], row[2]) for row in rows]
    return float(np.interp(t, dates, prices_at_k))


def main():
    times = np.array([.5, 1, 1.5, 2])
    known = np.array([.98, .955, .925, .89])
    coupons = np.array([2, 2.5, 3, 3.5])
    prices = np.array([c*math.fsum(known[:i+1])+100*known[i]
                       for i, c in enumerate(coupons)])
    ds = bootstrap(prices, coupons)
    np.testing.assert_allclose(ds, known, atol=2e-16)
    zeros = -np.log(ds)/times
    par_rates = 2*(1-ds)/np.cumsum(ds)
    for t, c, p, d, z, par in zip(times, coupons, prices, ds, zeros, par_rates,
                                  strict=True):
        y = bisect_width(lambda x: bond_price(x, round(2*t), c)-p,
                         0, .3, width_tol=1e-10)
        print(f'Bootstrap t={t:g} coupon={c:g} price={p:.9f} D={d:.9f} '
              f'zero={z:.12f} par={par:.12f} YTM={y.point:.12f}')
    for coupon in (0, 5):
        price = coupon*math.fsum(ds)+100*ds[-1]
        y = bisect_width(lambda x: bond_price(x, 4, coupon)-price,
                         0, .3, width_tol=1e-10)
        print('Same2year coupon/price/YTM:', coupon, price, y.point)
    mid = log_discount(1.25, times, ds)
    linear_zero_d = math.exp(-1.25*float(np.interp(1.25, times, zeros)))
    print('D(1.25) log-linear/linear-zero:', mid, linear_zero_d)
    print('Forward continuously compounded1to1.5:', math.log(ds[1]/ds[2])/.5)
    assert log_discount(0, times, ds) == 1
    for t, d in zip(times, ds, strict=True):
        np.testing.assert_allclose(log_discount(t, times, ds), d)
    bad = call_slice_checks([90, 100, 110], [15, 12, 5])
    assert bad['bounds'] and bad['verticals'] and not bad['convex']
    print('Bad call row diagnostics:', bad, 'butterfly cost=', 15-2*12+5)
    print('Declared lecture-maturity lab rows:', source_lab_table())
    for quote in (5, 6):
        iv = implied_call(quote, 27000, 5/365, 50, spot=24360, rate=.065,
                          bracket=(.01, 1))
        print('Source far-strike quote/IV/residual:', quote, iv.point, iv.residual)
    surface = surface_rows()
    for t, strikes, cs, vols in surface:
        print(f'Surface T={t:g}')
        for k, c, v in zip(strikes, cs, vols, strict=True):
            print(f'  K={k:g} price={c:.9f} CRR-IV={v:.12f}')
    dense_k = np.linspace(70, 130, 61)
    previous = None
    for t in np.linspace(.5, 2, 16):
        cs = [interpolate_surface(k, t, surface) for k in dense_k]
        checks = call_slice_checks(dense_k, cs)
        assert all(checks[key] for key in ('bounds', 'verticals', 'convex'))
        if previous is not None:
            assert np.all(np.array(cs) >= previous-1e-10)
        previous = np.array(cs)
    print('Interpolated price at K105,T.75:', interpolate_surface(105, .75, surface))
    # Lower-triangular cash-flow matrix provides an independent solve check.
    matrix = np.array([[coupons[i] if j < i else
                        100+coupons[i] if j == i else 0
                        for j in range(4)] for i in range(4)])
    np.testing.assert_allclose(np.linalg.solve(matrix, prices), ds)
    np.testing.assert_allclose(matrix@ds, prices)
    # Positive discounts can rise under negative rates: do not forbid this.
    np.testing.assert_allclose(bootstrap([102], [0]), [1.02])
    invalid = (lambda: bootstrap([1, .1], [100, 100]),
               lambda: bootstrap([100], [2, 3]),
               lambda: bootstrap([True], [2]),
               lambda: log_discount(3, times, ds),
               lambda: log_discount(1, [1, 1], [.9, .8]),
               lambda: log_discount(1, times, [1, .9, -.2, .8]),
               lambda: call_slice_checks([90, 90, 100], [15, 12, 5]))
    for function in invalid:
        try:
            function()
        except ValueError:
            continue
        raise AssertionError('invalid call must fail')
    print('All curve, shape, calibration and failure checks passed.')


if __name__ == '__main__':
    main()
