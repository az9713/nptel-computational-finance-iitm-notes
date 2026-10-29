"""Declared finite expected-utility decisions and transformed-density checks.

Continuous lognormal expected-utility quadrature is developed in Chapter15.
No market probabilities or personal risk preferences are estimated here.
"""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import scalar  # noqa: E402
from ch10_inverse_pricing import bisect_width  # noqa: E402


def finite_lottery(outcomes, probabilities):
    raw_x, raw_p = np.asarray(outcomes, dtype=object), np.asarray(probabilities, dtype=object)
    if (raw_x.ndim != 1 or raw_p.ndim != 1 or not len(raw_x)
            or len(raw_x) != len(raw_p)):
        raise ValueError('matching nonempty one-dimensional lottery vectors needed')
    xs = np.array([scalar(x, 'outcome') for x in raw_x])
    ps = np.array([scalar(p, 'probability') for p in raw_p])
    if np.any(ps < 0) or not math.isclose(math.fsum(ps), 1, rel_tol=0, abs_tol=1e-14):
        raise ValueError('nonnegative probabilities must sum to one')
    return xs, ps


def crra(wealth, gamma):
    """Normalized CRRA utility, with the actual logarithmic limit at gamma1."""
    w = scalar(wealth, 'wealth', positive=True)
    g = scalar(gamma, 'gamma')
    if g < 0:
        raise ValueError('gamma must be nonnegative in this teaching family')
    if g == 1:
        return math.log(w)
    return math.expm1((1-g)*math.log(w))/(1-g)


def crra_ce(wealths, probabilities, gamma):
    """CRRA certainty equivalent, with a centered near-logarithmic branch."""
    xs, ps = finite_lottery(wealths, probabilities)
    g = scalar(gamma, 'gamma')
    if min(xs) <= 0 or g < 0:
        raise ValueError('positive wealth and nonnegative gamma needed')
    if g == 1:
        return math.exp(math.fsum(p*math.log(x) for p, x in zip(ps, xs, strict=True)))
    power = 1-g
    if abs(power) < 1e-5:
        # Remove the near-one moment before taking a small reciprocal power.
        normalized = ps/math.fsum(ps)
        logs = [math.log(x) for x in xs]
        center = math.fsum(p*x for p, x in zip(normalized, logs, strict=True))
        offset = math.fsum(p*math.expm1(power*(x-center))
                           for p, x in zip(normalized, logs, strict=True))
        return math.exp(center+math.log1p(offset)/power)
    # Moderate fixtures only; log-sum-exp would broaden the extreme-value range.
    moment = math.fsum(p*x**(1-g) for p, x in zip(ps, xs, strict=True))
    return moment**(1/(1-g))


def terminal_wealth(initial, alpha, safe_return, risky_return):
    w0 = scalar(initial, 'initial wealth', positive=True)
    a, rf = scalar(alpha, 'alpha'), scalar(safe_return, 'safe return')
    r = scalar(risky_return, 'risky return')
    if not 0 <= a <= 1 or rf <= -1 or r <= -1:
        raise ValueError('long-only allocation and strictly positive gross returns needed')
    return w0*((1-a)*(1+rf)+a*(1+r))


RETURNS = np.array([-.12, .06, .25])
PROBABILITIES = np.array([.25, .5, .25])


def discrete_objective(alpha, gamma=3, safe_return=.045):
    a = scalar(alpha, 'alpha')
    gs = [terminal_wealth(1, a, safe_return, r) for r in RETURNS]
    return math.fsum(p*crra(g, gamma) for p, g in zip(PROBABILITIES, gs, strict=True))


def discrete_derivative(alpha, gamma=3, safe_return=.045):
    a, g = scalar(alpha, 'alpha'), scalar(gamma, 'gamma')
    if g < 0:
        raise ValueError('gamma must be nonnegative')
    gs = [terminal_wealth(1, a, safe_return, r) for r in RETURNS]
    return math.fsum(p*wealth**(-g)*(r-safe_return)
                     for p, wealth, r in zip(PROBABILITIES, gs, RETURNS, strict=True))


def cara_ce(outcomes, probabilities, aversion):
    xs, ps = finite_lottery(outcomes, probabilities)
    a = scalar(aversion, 'absolute risk aversion', positive=True)
    # Log-sum-exp avoids overflowing exp(-a*x) for a finite loss.
    terms = [math.log(p)-a*x for p, x in zip(ps, xs, strict=True) if p > 0]
    top = max(terms)
    return -(top+math.log(math.fsum(math.exp(t-top) for t in terms)))/a


def log_gross_density(x, mean=.09, sigma=.22):
    x, mu = scalar(x, 'log gross'), scalar(mean, 'mean')
    sd = scalar(sigma, 'sigma', positive=True)
    z = (x-mu)/sd
    return math.exp(-z*z/2)/(sd*math.sqrt(2*math.pi))


def petersburg_log_value(fee, initial=100, terms=256):
    """First256 exact-probability terms; omitted tail bounded in the text."""
    p = scalar(fee, 'fee')
    w0 = scalar(initial, 'initial wealth', positive=True)
    if not 0 <= p <= w0 or terms != 256:
        raise ValueError('this demonstration uses fee0..initial and256terms')
    return math.fsum(2.0**(-n)*math.log(w0-p+2.0**n) for n in range(1, terms+1))


def main():
    outcomes, probs = [5, 15], [.5, .5]
    print('5/15 lottery mean/logCE/squareEU:', 10, crra_ce(outcomes, probs, 1),
          .5*5**2+.5*15**2)
    ticket_two = 20-5*math.sqrt(5)
    np.testing.assert_allclose(.5*math.log(15-ticket_two)+.5*math.log(25-ticket_two),
                               math.log(10))
    print('Background10 two-outcome ticket reservationprice:', ticket_two)
    for g in (0, 1, 3):
        print('5/15 lottery CRRA gamma/CE:', g, crra_ce(outcomes, probs, g))
    np.testing.assert_allclose(crra(2, 1+1e-8), math.log(2), rtol=5e-9)
    for g in (np.nextafter(1., 0.), np.nextafter(1., 2.), 1-1e-6, 1+1e-6):
        # Independent two-outcome near-log expansion; next omitted term O(power^3).
        center = .5*math.log(75)
        half_gap = .5*math.log(3)
        reference = math.exp(center+(1-g)*half_gap**2/2)
        np.testing.assert_allclose(crra_ce(outcomes, probs, g), reference, rtol=1e-14)
    wealths = np.array([terminal_wealth(500, .4, .045, r) for r in RETURNS])
    expected = math.fsum(PROBABILITIES*wealths)
    source_eu = math.fsum(-p/(2*w*w) for p, w in zip(PROBABILITIES, wealths, strict=True))
    ce = crra_ce(wealths, PROBABILITIES, 3)
    variance = math.fsum(p*(w-expected)**2 for p, w in zip(PROBABILITIES, wealths, strict=True))
    print('Workshop wealths/canonical utilities:', wealths, -1/(2*wealths**2))
    print('Workshop mean/EU/u(mean)/CE/premium/localpremium:', expected, source_eu,
          -1/(2*expected**2), ce, expected-ce, 3*variance/(2*expected))
    assert min(wealths) <= ce < expected <= max(wealths)
    derivative_ends = [discrete_derivative(a) for a in (0, 1)]
    optimum = bisect_width(discrete_derivative, 0, 1, width_tol=1e-10)
    print('Discrete derivative endpoints/optimum:', derivative_ends, optimum)
    assert derivative_ends[0] > 0 > derivative_ends[1]
    relative_wealth = np.array([terminal_wealth(1, optimum.point, .045, r)
                                for r in RETURNS])
    q = PROBABILITIES*relative_wealth**(-3)
    q /= math.fsum(q)
    np.testing.assert_allclose(q@(1+RETURNS), 1.045, atol=1e-12, rtol=0)
    print('Marginalutility normalizedweights and Qmean gross:', q, q@(1+RETURNS))
    for a in (0, .4, optimum.point, 1):
        ws = [terminal_wealth(500, a, .045, r) for r in RETURNS]
        print('Allocation alpha/objective/CE:', a, discrete_objective(a),
              crra_ce(ws, PROBABILITIES, 3))
    for initial in (50, 500, 5000):
        ws = [terminal_wealth(initial, optimum.point, .045, r) for r in RETURNS]
        print('Scale check W0/CEoverW0:', initial, crra_ce(ws, PROBABILITIES, 3)/initial)
    for mu, sigma in ((.08, .2), (.09, .22)):
        print('Loggross mu/sigma/Egross/Esimple:', mu, sigma,
              math.exp(mu+sigma*sigma/2), math.expm1(mu+sigma*sigma/2))
    r = .25
    x = math.log1p(r)
    z = (x-.09)/.22
    fx = log_gross_density(x)
    fr = fx/(1+r)
    utility = -1/(2*terminal_wealth(500, .4, .045, r)**2)
    print('Densitypoint r/x/z/fX/fR/u/fXintegrand/fRintegrand/zIntegrand:',
          r, x, z, fx, fr, utility, utility*fx, utility*fr, utility*fx*.22)
    np.testing.assert_allclose(fr*(1+r), fx)
    for terms in (10, 20):
        print('Cappedgame terms/EX (no head -> zero):', terms, terms)
    ticket = bisect_width(lambda fee: petersburg_log_value(fee)-math.log(100),
                          0, 100, width_tol=1e-10)
    tail_bound = (math.log(100)+258*math.log(2))*2.0**(-256)
    print('Petersburg ElogX/CEofX/ticketfee/tailbound:', 2*math.log(2), 4,
          ticket, tail_bound)
    aversion = 1/50000
    gamble = np.array([50000, -25000])
    for share in (1, .1, .01, .001):
        eq = cara_ce(share*gamble, [.5, .5], aversion)
        print('CARA share/mean/CE/premium:', share, share*12500, eq, share*12500-eq)
    small_ce = cara_ce(.01*gamble, [.5, .5], aversion)
    print('99shares at123 cash/retainedCE/totalCE:', 99*123, small_ce, 99*123+small_ce)
    for n in (1, 100, 10000):
        sd = 200000*math.sqrt(.001*.999/n)
        print('Independent insurer pool N/expectedprofitperpolicy/SDlossperpolicy:', n, 100, sd)
    max_premium = -cara_ce([-200000, 0], [.001, .999], aversion)
    print('Fullinsurance maxpremium/expectedloss:', max_premium, 200)
    assert max_premium > 300 > 200
    print('Hiddeneffort retainedshare threshold:', 10/(100*(.8-.2)))
    # Known constants and translation/scaling invariances check utility mechanics.
    np.testing.assert_allclose(crra_ce([100, 100], [.2, .8], 3), 100)
    np.testing.assert_allclose(cara_ce(gamble+1000, [.5, .5], aversion),
                               cara_ce(gamble, [.5, .5], aversion)+1000)
    np.testing.assert_allclose(crra_ce(2*wealths, PROBABILITIES, 3), 2*ce)
    invalid = (lambda: finite_lottery([1, 2], [.5, .6]),
               lambda: finite_lottery([True, 2], [.5, .5]),
               lambda: crra(0, 3),
               lambda: crra_ce([1, -1], [.5, .5], 3),
               lambda: terminal_wealth(100, 1.1, .05, .1),
               lambda: log_gross_density(0, sigma=0),
               lambda: cara_ce([1], [1], 0))
    for function in invalid:
        try:
            function()
        except ValueError:
            continue
        raise AssertionError('invalid call must fail')
    print('All utility, portfolio, density, sharing and failure checks passed.')


if __name__ == '__main__':
    main()
