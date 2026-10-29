"""Composite quadrature and complete declared calculus-laboratory fixtures.

Gamma3 canonical utility is -1/(2W**2). Its tail bound is separate from grid
and floating-point error. Library R/SciPy comparisons are documentary only.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import scalar, step_count  # noqa: E402
from ch10_inverse_pricing import bisect_width  # noqa: E402
from ch13_bond_calculus import bond_measures, differences  # noqa: E402


@dataclass(frozen=True)
class QuadratureResult:
    value: float
    evaluations: int
    intervals: int
    left: float
    right: float
    rule: str


def weights(intervals, rule):
    n = step_count(intervals)
    if rule == 'trapezoid':
        w = np.ones(n+1)
        w[[0, -1]] = .5
    elif rule == 'simpson':
        if n % 2:
            raise ValueError('Simpson one-third requires an even interval count')
        w = np.ones(n+1)
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        w /= 3
    else:
        raise ValueError('unknown quadrature rule')
    return w


def sampled_integral(values, left, right, rule='simpson'):
    a, b = scalar(left, 'left'), scalar(right, 'right')
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or len(raw) < 2 or a >= b:
        raise ValueError('ordered interval and at least two equispaced values needed')
    ys = [scalar(v, 'integrand value') for v in raw]
    n = len(ys)-1
    w = weights(n, rule)
    h = (b-a)/n
    result = h*math.fsum(float(wi)*yi for wi, yi in zip(w, ys, strict=True))
    if not math.isfinite(result):
        raise ValueError('quadrature arithmetic outside finite range')
    return result


def composite(function, left, right, intervals, rule='simpson'):
    if not callable(function):
        raise ValueError('integrand must be callable')
    a, b = scalar(left, 'left'), scalar(right, 'right')
    n = step_count(intervals)
    weights(n, rule)
    if a >= b:
        raise ValueError('left must be below right')
    xs = np.linspace(a, b, n+1)
    if not np.all(np.diff(xs) > 0):
        raise ArithmeticError('grid points are not distinguishable')
    value = sampled_integral([function(x) for x in xs], a, b, rule)
    return QuadratureResult(value, n+1, n, a, b, rule)


def utility_parameters(alpha, mu, sigma, safe_return, initial):
    a, mu = scalar(alpha, 'alpha'), scalar(mu, 'mu')
    sd = scalar(sigma, 'sigma', positive=True)
    rf = scalar(safe_return, 'safe return')
    w0 = scalar(initial, 'initial wealth', positive=True)
    if not 0 <= a <= 1 or rf <= -1:
        raise ValueError('long-only alpha and positive safe gross return needed')
    return a, mu, sd, 1+rf, w0


def canonical_eu(alpha, *, mu=.08, sigma=.2, safe_return=.05, initial=100,
                 intervals=320, limit=8, rule='simpson', derivative=False):
    """Truncated standard-normal-coordinate integral; no renormalization."""
    a, mu, sd, bank, w0 = utility_parameters(alpha, mu, sigma, safe_return, initial)
    limit = scalar(limit, 'normal truncation limit', positive=True)
    if not isinstance(derivative, bool):
        raise ValueError('derivative must be boolean')

    def integrand(z):
        gross = math.exp(mu+sd*z)
        wealth = w0*((1-a)*bank+a*gross)
        value = w0*(gross-bank)/wealth**3 if derivative else -.5/wealth**2
        return value*math.exp(-z*z/2)/math.sqrt(2*math.pi)

    return composite(integrand, -limit, limit, intervals, rule)


def canonical_eu_return(alpha, *, intervals=320, rule='simpson'):
    """Source39/44/45 parameters, simple-return interval [-.95,5]."""
    a, mu, sd, bank, w0 = utility_parameters(alpha, .08, .2, .05, 100)

    def integrand(r):
        z = (math.log1p(r)-mu)/sd
        density = math.exp(-z*z/2)/((1+r)*sd*math.sqrt(2*math.pi))
        wealth = w0*((1-a)*bank+a*(1+r))
        return -.5/wealth**2*density

    return composite(integrand, -.95, 5, intervals, rule)


def normal_cdf(x):
    return .5*math.erfc(-x/math.sqrt(2))


def utility_tail_bound(*, mu=.08, sigma=.2, safe_return=.05, initial=100, limit=8):
    """Uniform absolute omitted-tail bound for all alpha in [0,1], gamma3."""
    _, mu, sd, bank, w0 = utility_parameters(.5, mu, sigma, safe_return, initial)
    length = scalar(limit, 'normal limit', positive=True)
    normal_tail = 2*normal_cdf(-length)
    shifted_tail = normal_cdf(-length-2*sd)+normal_cdf(-length+2*sd)
    return (bank**(-2)*normal_tail
            + math.exp(-2*mu+2*sd*sd)*shifted_tail)/(2*w0*w0)


def optimum_continuous(*, intervals=320, limit=8):
    def objective_slope(a):
        return canonical_eu(a, intervals=intervals, limit=limit, derivative=True).value

    assert objective_slope(0) > 0 > objective_slope(1)
    return bisect_width(objective_slope, 0, 1, width_tol=1e-9)


def exp_convergence():
    exact = math.expm1(1)
    rows = []
    for n in (10, 20, 40, 80, 160, 320):
        t = composite(math.exp, 0, 1, n, 'trapezoid').value
        s = composite(math.exp, 0, 1, n, 'simpson').value
        assert 0 <= t-exact <= math.e/(12*n*n)
        assert 0 <= s-exact <= math.e/(180*n**4)+1e-15
        rows.append((n, t, s, t-exact, s-exact))
    return rows


def main():
    print('Exponential convergence n/T/S/errors:', exp_convergence())
    ys = [(1+x)**2 for x in np.linspace(0, 1, 5)]
    print('Original five-point table/T/S/exact:', ys,
          sampled_integral(ys, 0, 1, 'trapezoid'), sampled_integral(ys, 0, 1), 7/3)
    # Polynomial exactness checks are independent of the financial integrand.
    for degree in range(4):
        s = composite(lambda x: x**degree, 0, 1, 4).value
        np.testing.assert_allclose(s, 1/(degree+1), atol=1e-15, rtol=0)
    np.testing.assert_allclose(composite(lambda x: 2*x+3, 0, 1, 3, 'trapezoid').value, 4)

    def narrow(x):
        return max(1-((x-.375)/.02)**2, 0)**5

    assert composite(narrow, 0, 1, 4, 'trapezoid').value == 0
    assert composite(narrow, 0, 1, 4).value == 0
    print('Unseen smooth bump coarse T/S versus exact:', 0, 0, .02*512/693)
    fine = canonical_eu(.5, intervals=2048, limit=9).value
    print('Mixed utility fine-grid comparison value:', fine,
          'CE=', (-2*fine)**(-.5))
    for n in (20, 40, 80, 160, 320):
        vals = [canonical_eu(.5, intervals=n, rule=rule).value
                for rule in ('trapezoid', 'simpson')]
        rs = [canonical_eu_return(.5, intervals=n, rule=rule).value
              for rule in ('trapezoid', 'simpson')]
        print('Utility n/zT/zS/rT/rS:', n, *vals, *rs)
    for length in (3, 6, 8):
        value = canonical_eu(.5, intervals=640, limit=length).value
        bound = utility_tail_bound(limit=length)
        print('Normal limit/value/tailbound:', length, value, bound)
    for a, exact in ((0, -.5/(100*1.05)**2),
                     (1, -.5/100**2*math.exp(-2*.08+2*.2**2))):
        value = canonical_eu(a, intervals=640).value
        np.testing.assert_allclose(value, exact, atol=1e-16, rtol=0)
        print('Analytic endpoint alpha/value/exact:', a, value, exact)
    slope_zero = (math.exp(.08+.2**2/2)-1.05)/(100**2*1.05**3)
    slope_one = (math.exp(-2*.08+2*.2**2)
                 - 1.05*math.exp(-3*.08+4.5*.2**2))/100**2
    for a, exact in ((0, slope_zero), (1, slope_one)):
        numerical = canonical_eu(a, intervals=640, derivative=True).value
        np.testing.assert_allclose(numerical, exact, atol=1e-16, rtol=0)
    print('Analytic objective derivative endpoints:', slope_zero, slope_one)
    optimum = optimum_continuous()
    refined = optimum_continuous(intervals=640, limit=9)
    print('Continuous optimum and refined:', optimum, refined)
    assert abs(optimum.point-refined.point) < 2e-9
    for a in (0, .42, .43, .5, optimum.point, 1):
        eu = canonical_eu(a, intervals=640, limit=9).value
        print('Continuous alpha/EU/CE:', a, eu, (-2*eu)**(-.5))
    second = canonical_eu(.4, mu=.09, sigma=.22, safe_return=.045, initial=500,
                          intervals=640, limit=9).value
    print('Source43 separate continuous EU/CE:', second, (-2*second)**(-.5))
    h = 1e-4
    print('exp1 finite differences h1e-4:', differences(math.exp, 1, h))
    hs = np.logspace(-1, -14, 200)
    errors = np.array([[abs(differences(math.exp, 1, h)[key]-math.e)
                        for key in ('forward', 'centered')] for h in hs])
    for j, key in enumerate(('forward', 'centered')):
        i = int(np.argmin(errors[:, j]))
        print('exp1 sampled minimum:', key, hs[i], errors[i, j])
    semi = bond_measures(.0739, [3.5]*9+[103.5], 2)
    zero = bond_measures(.07, [0]*9+[100], 1)
    long = bond_measures(.07, [7]*29+[107], 1)
    print('Rounded5year yield.0739:', semi)
    print('10yearzero and30yearannualpar:', zero, long)

    def semi_price(y):
        return bond_measures(y, [3.5]*9+[103.5], 2).price

    d = differences(semi_price, .0739, 1e-4)
    print('Roundedbond numeric D/C:', -d['centered']/semi.price, d['second']/semi.price)
    days = np.arange(0, 183, 7)
    dirty = 98.4+3.5*days/182
    assert days[-1] == 182
    print('Illustrative constantclean days0/91/182 dirty:', [98.4+3.5*d/182 for d in (0, 91, 182)])
    np.testing.assert_allclose(dirty[-1], 101.9)
    base, up, down, bump = 96.2312, 94.4140, 98.0920, .0025
    print('Supplied25bp D/C:', (down-up)/(2*bump*base), (down-2*base+up)/(bump*bump*base))
    invalid = (lambda: composite(math.exp, 0, 1, 3),
               lambda: composite(math.exp, 1, 0, 4),
               lambda: composite(math.exp, 0, 1, True),
               lambda: sampled_integral([True, 2, 3], 0, 1),
               lambda: composite(lambda x: float('nan'), 0, 1, 4),
               lambda: canonical_eu(1.1),
               lambda: canonical_eu(.5, sigma=0),
               lambda: utility_tail_bound(limit=0))
    for function in invalid:
        try:
            function()
        except ValueError:
            continue
        raise AssertionError('invalid call must fail')
    print('All quadrature, tail, utility and paired-laboratory checks passed.')


if __name__ == '__main__':
    main()
