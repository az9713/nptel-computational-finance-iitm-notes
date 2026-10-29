"""No-dividend European prices, GBM moments, and the complete source workshop.

Annual continuously compounded rates and volatility per square-root year.
Finite moderate teaching inputs; this is not a production tail-price library.
The zero-time and zero-volatility cases have explicit price limits. Greeks
require strictly positive spot, strike, time, and volatility.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch08_binomial_valuation import scalar, crr, lattice, payoff  # noqa: E402


def normal_cdf(x):
    """Standard normal CDF, using erfc to retain small left-tail values."""
    x = scalar(x, 'normal coordinate')
    return .5 * math.erfc(-x / math.sqrt(2))


def normal_pdf(x):
    x = scalar(x, 'normal coordinate')
    return math.exp(-.5 * x * x) / math.sqrt(2 * math.pi)


def inputs(spot, strike, rate, volatility, horizon):
    values = [scalar(v, name) for v, name in zip(
        (spot, strike, rate, volatility, horizon),
        ('spot', 'strike', 'rate', 'volatility', 'horizon'))]
    if any(values[i] < 0 for i in (0, 1, 3, 4)):
        raise ValueError('spot, strike, volatility and horizon must be nonnegative')
    return values


def black_scholes(spot, strike, rate, volatility, horizon, kind='call'):
    """Price only; permits S=0, K=0, tau=0, sigma=0 via model limits.

    Ordinary subtraction can lose relative accuracy for very small prices.
    No silent nonnegativity clipping conceals that numerical limitation.
    """
    s, k, r, sig, tau = inputs(spot, strike, rate, volatility, horizon)
    if kind not in ('call', 'put'):
        raise ValueError('kind must be call or put')
    if tau == 0:
        return max((s - k) if kind == 'call' else (k - s), 0.)
    try:
        kd = k * math.exp(-r * tau)
        if not math.isfinite(kd):
            raise ValueError('discounted strike outside finite range')
        if s == 0 or k == 0 or sig == 0:
            return max(s - kd, 0.) if kind == 'call' else max(kd - s, 0.)
        scale = sig * math.sqrt(tau)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('normal scale outside positive finite range')
        d1 = (math.log(s / k) + r * tau) / scale + .5 * scale
        d2 = d1 - scale
        if kind == 'call':
            result = s * normal_cdf(d1) - kd * normal_cdf(d2)
        else:
            result = kd * normal_cdf(-d2) - s * normal_cdf(-d1)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError('price arithmetic outside finite range') from exc
    if not math.isfinite(result):
        raise ValueError('price outside finite range')
    return result


@dataclass(frozen=True)
class Greeks:
    d1: float
    d2: float
    delta: float
    gamma: float
    vega: float
    theta: float


def greeks(spot, strike, rate, volatility, horizon, kind='call'):
    """Vega per unit sigma, theta per year of calendar-time passage."""
    s, k, r, sig, tau = inputs(spot, strike, rate, volatility, horizon)
    if min(s, k, sig, tau) <= 0 or kind not in ('call', 'put'):
        raise ValueError('Greeks require positive S, K, sigma, tau and a valid kind')
    scale = sig * math.sqrt(tau)
    d1 = (math.log(s / k) + r * tau) / scale + .5 * scale
    d2 = d1 - scale
    phi = normal_pdf(d1)
    kd = k * math.exp(-r * tau)
    gamma = phi / (s * scale)
    vega = s * phi * math.sqrt(tau)
    common = -s * phi * sig / (2 * math.sqrt(tau))
    if kind == 'call':
        delta = normal_cdf(d1)
        theta = common - r * kd * normal_cdf(d2)
    else:
        delta = normal_cdf(d1) - 1
        theta = common + r * kd * normal_cdf(-d2)
    result = Greeks(d1, d2, delta, gamma, vega, theta)
    if not all(math.isfinite(v) for v in result.__dict__.values()):
        raise ValueError('Greek arithmetic outside finite range')
    return result


def gbm_moments(spot, drift, volatility, horizon):
    """Log-price-ratio mean/variance and stock mean/variance under named drift."""
    s = scalar(spot, 'spot', positive=True)
    mu = scalar(drift, 'drift')
    sig = scalar(volatility, 'volatility')
    tau = scalar(horizon, 'horizon')
    if sig < 0 or tau < 0:
        raise ValueError('volatility and horizon must be nonnegative')
    log_mean = (mu - .5 * sig ** 2) * tau
    log_variance = sig ** 2 * tau
    mean = s * math.exp(mu * tau)
    variance = mean ** 2 * math.expm1(log_variance)
    result = (log_mean, log_variance, mean, variance)
    if not all(math.isfinite(v) for v in result):
        raise ValueError('moment arithmetic outside finite range')
    return result


def estimate_gbm(log_returns, dt):
    """Equal-spacing moment estimates; sample variance uses n-1, not MLE n."""
    dt = scalar(dt, 'spacing', positive=True)
    raw = np.asarray(log_returns, dtype=object)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError('provide at least two one-dimensional log returns')
    data = np.array([scalar(v, 'log return') for v in raw])
    mean = float(np.mean(data))
    variance = float(np.var(data, ddof=1))
    sigma_squared = variance / dt
    mu = mean / dt + .5 * sigma_squared
    return mu, math.sqrt(sigma_squared), mean, variance


def direct_normal_price(spot, strike, rate, volatility, horizon, kind):
    """Separate Simpson payoff integral, split at strike, z in [-10,10].

    Fixed moderate fixtures only. An independent numerical comparison, not an
    error certificate or a general adaptive integrator.
    """
    s, k, r, sig, tau = inputs(spot, strike, rate, volatility, horizon)
    if min(s, k, sig, tau) <= 0 or kind not in ('call', 'put'):
        raise ValueError('positive interior parameters required')
    scale = sig * math.sqrt(tau)
    drift = (r - .5 * sig ** 2) * tau
    threshold = (math.log(k / s) - drift) / scale
    a, b = (max(-10., threshold), 10.) if kind == 'call' else (-10., min(10., threshold))
    if a >= b:
        return 0.
    n = 4000
    h = (b - a) / n
    terms = []
    sign = 1 if kind == 'call' else -1
    for i in range(n + 1):
        z = a + i * h
        stock = s * math.exp(drift + scale * z)
        weight = 1 if i in (0, n) else (4 if i % 2 else 2)
        terms.append(weight * max(sign * (stock - k), 0.) * normal_pdf(z))
    return math.exp(-r * tau) * h / 3 * math.fsum(terms)


def workshop():
    market = crr(64, .055, .32, .0625, 1)
    result = {
        'one_step': {'up': market.up, 'down': market.down, 'q': market.q,
                     'expected_stock': 64 * market.growth},
        'three_year_original': gbm_moments(64, .055, .32, 3),
        'three_year_later': gbm_moments(64, .045, .28, 3)}
    result['priya'] = {kind: black_scholes(85, 90, .045, .28, .75, kind)
                       for kind in ('call', 'put')}
    result['greeks'] = {kind: greeks(85, 90, .045, .28, .75, kind).__dict__
                        for kind in ('call', 'put')}
    result['trees'] = [(n, lattice(crr(85, .045, .28, .75, n), n, payoff('call', 90)))
                       for n in (4, 25, 100, 1000)]
    result['volatility'] = [(v, black_scholes(85, 90, .045, v, .75),
                            black_scholes(85, 90, .045, v, .75, 'put'))
                            for v in (.20, .28, .36)]
    result['boundary'] = [(v, black_scholes(250, 75, .045, v, .75),
                          black_scholes(250, 75, .045, v, .75, 'put'))
                          for v in (.28, 1.)]
    result['discounted_strike75'] = 75 * math.exp(-.045 * .75)
    returns = [.02, -.01, .03, -.02, .01, 0, .025, -.015, .018, .012, -.005, .022]
    result['estimates'] = estimate_gbm(returns, 1 / 12)
    result['mle_sigma'] = math.sqrt(np.var(returns, ddof=0) * 12)
    return result


def main():
    import json
    result = workshop()
    print(json.dumps(result, indent=2))
    for s in (60, 85, 120):
        for sig in (.12, .28, .6):
            for r in (-.03, .045):
                c = black_scholes(s, 90, r, sig, .75)
                p = black_scholes(s, 90, r, sig, .75, 'put')
                np.testing.assert_allclose(c - p, s - 90 * math.exp(-r * .75),
                                           rtol=0, atol=4e-14)
                for kind in ('call', 'put'):
                    value = black_scholes(s, 90, r, sig, .75, kind)
                    integral = direct_normal_price(s, 90, r, sig, .75, kind)
                    np.testing.assert_allclose(value, integral, rtol=0, atol=2e-8)
                    g = greeks(s, 90, r, sig, .75, kind)

                    def fn(x):
                        return black_scholes(x, 90, r, sig, .75, kind)

                    h = .005
                    delta = (fn(s + h) - fn(s - h)) / (2 * h)
                    gamma = (fn(s + h) - 2 * fn(s) + fn(s - h)) / h ** 2
                    np.testing.assert_allclose([delta, gamma], [g.delta, g.gamma],
                                               rtol=0, atol=2e-8)
                    eps = 1e-5
                    vega = (black_scholes(s, 90, r, sig + eps, .75, kind)
                            - black_scholes(s, 90, r, sig - eps, .75, kind)) / (2 * eps)
                    theta = (black_scholes(s, 90, r, sig, .75 - eps, kind)
                             - black_scholes(s, 90, r, sig, .75 + eps, kind)) / (2 * eps)
                    np.testing.assert_allclose([vega, theta], [g.vega, g.theta],
                                               rtol=0, atol=2e-7)
                    residual = g.theta + r * s * g.delta
                    residual += .5 * sig ** 2 * s ** 2 * g.gamma - r * value
                    assert abs(residual) < 2e-13
    np.testing.assert_allclose(normal_cdf(0), .5)
    np.testing.assert_allclose(black_scholes(100, 100, 0, .2, 1),
                               7.965567455405804, atol=3e-14)
    assert black_scholes(85, 90, .045, .28, 0, 'put') == 5
    assert black_scholes(0, 90, .045, .28, .75) == 0
    assert black_scholes(85, 0, .045, .28, .75) == 85
    np.testing.assert_allclose(black_scholes(85, 90, .045, 0, .75, 'put'),
                               90 * math.exp(-.045 * .75) - 85)
    for invalid in (True, float('nan'), float('inf'), -1):
        try:
            black_scholes(85, 90, .045, invalid, .75)
        except ValueError:
            continue
        raise AssertionError('invalid volatility accepted')
    for data in ([.1], [[.1, .2]], [True, .1], [.1, float('nan')]):
        try:
            estimate_gbm(data, 1 / 12)
        except ValueError:
            continue
        raise AssertionError('invalid return data accepted')
    print('All price, payoff-integral, parity, Greek, PDE and contract checks passed.')


if __name__ == '__main__':
    main()
