"""Discrete Asian contracts, geometric controls, and a complete pricing audit.

Declared synthetic examples only. No files or network access. Production paths
retain current state and running sums, not the entire observation matrix.
"""
from dataclasses import dataclass
import itertools
import math
from numbers import Real
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch22_sampling import estimate, positive_integer  # noqa: E402
from ch23_pricing_and_inference import call_price  # noqa: E402


def market(spot, strike, rate, volatility, horizon):
    values = (spot, strike, rate, volatility, horizon)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or
           not math.isfinite(x) for x in values):
        raise ValueError("market inputs must be finite real numbers")
    if min(spot, volatility, horizon) <= 0 or strike < 0:
        raise ValueError(
            "require positive spot/volatility/horizon and nonnegative strike")


def observation_dates(dates, horizon):
    raw = np.asarray(dates)
    if raw.dtype.kind not in 'fiu':
        raise ValueError("observation dates must be real numbers")
    times = np.asarray(raw, dtype=float)
    if (times.ndim != 1 or times.size == 0 or
            not np.all(np.isfinite(times)) or times[0] < 0 or
            not np.all(np.diff(times) > 0) or times[-1] != horizon):
        raise ValueError(
            "dates must increase from >= 0 through exact maturity")
    return times.copy()


def equal_dates(steps, horizon=1., include_initial=False):
    steps = positive_integer(steps, "observation intervals")
    if not isinstance(include_initial, bool):
        raise ValueError("initial-observation flag must be boolean")
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("positive finite maturity required")
    times = np.linspace(0., horizon, steps+1)
    return times if include_initial else times[1:]


def geometric_log_moments(dates, spot=100., rate=.1, volatility=.3,
                          horizon=1.):
    market(spot, 0., rate, volatility, horizon)
    times = observation_dates(dates, horizon)
    gaps = np.diff(np.r_[0., times])
    remaining = np.arange(times.size, 0, -1)/times.size
    variance = volatility**2*float(np.dot(gaps, remaining**2))
    mean = math.log(spot)+(rate-.5*volatility**2)*float(np.mean(times))
    return mean, variance


def geometric_price(dates, spot=100., strike=100., rate=.1, volatility=.3,
                    horizon=1.):
    market(spot, strike, rate, volatility, horizon)
    mean, variance = geometric_log_moments(
        dates, spot, rate, volatility, horizon)
    return lognormal_call(mean, variance, strike, math.exp(-rate*horizon))


def lognormal_call(mean, variance, strike, discount):
    """Discounted positive-part expectation for an explicitly normal log."""
    if (not all(math.isfinite(x) for x in
                (mean, variance, strike, discount)) or
            variance < 0 or strike < 0 or discount <= 0):
        raise ValueError("finite normal inputs and positive discount required")
    expected = math.exp(mean+.5*variance)
    if strike == 0:
        return discount*expected
    if variance == 0:
        return discount*max(expected-strike, 0)
    width = math.sqrt(variance)
    d2 = (mean-math.log(strike))/width
    d1 = d2+width
    # erfc retains small lower-tail probabilities lost by 1 + erf.
    cdf1 = .5*math.erfc(-d1/math.sqrt(2))
    cdf2 = .5*math.erfc(-d2/math.sqrt(2))
    result = discount*(expected*cdf1-strike*cdf2)
    if not math.isfinite(result) or result < 0:
        raise ValueError("geometric formula outside supported floating range")
    return result


def average_forward_price(dates, spot=100., rate=.1, horizon=1.):
    market(spot, 0., rate, .3, horizon)
    times = observation_dates(dates, horizon)
    return spot*float(np.mean(np.exp(rate*(times-horizon))))


def log_mgf(scale, innovation):
    if innovation == 'normal':
        return .5*scale**2
    if innovation == 'coin':
        return float(np.logaddexp(scale, -scale)-math.log(2))
    raise ValueError("only normal or compensated coin innovations supported")


@dataclass(frozen=True)
class PayoffDraws:
    arithmetic: np.ndarray
    geometric: np.ndarray
    average_forward: np.ndarray
    discounted_terminal_stock: np.ndarray
    antithetic_pairs: bool


def _stream_payoffs(columns, times, count, spot, strike, rate, volatility,
                    horizon, paired, innovation):
    """One column per observation, count independent paths or pairs."""
    signs = np.array([1., -1.]) if paired else np.array([1.])
    logs = np.full((signs.size, count), math.log(spot))
    price_sum, log_sum = np.zeros_like(logs), np.zeros_like(logs)
    gaps = np.diff(np.r_[0., times])
    for gap, column in zip(gaps, columns, strict=True):
        z = np.asarray(column, dtype=float)
        if z.shape != (count,) or not np.all(np.isfinite(z)):
            raise ValueError("finite scalar innovation column required")
        if (innovation == 'coin' and
                not np.all((z == -1) | (z == 1) | (gap == 0))):
            raise ValueError("coin innovations must be -1 or +1")
        scale = volatility*math.sqrt(gap)
        logs += rate*gap-log_mgf(scale, innovation)+scale*signs[:, None]*z
        with np.errstate(over='raise', invalid='raise'):
            try:
                stock = np.exp(logs)
            except FloatingPointError as error:
                raise ValueError("paths exceed finite range") from error
        price_sum += stock
        log_sum += logs
    arithmetic = price_sum/times.size
    geometric = np.exp(log_sum/times.size)
    tolerance = 64*np.finfo(float).eps*np.maximum(1., arithmetic)
    assert np.all(arithmetic >= geometric-tolerance)
    discount = math.exp(-rate*horizon)
    # Average payoffs AFTER positive-part evaluation in each paired direction.
    return PayoffDraws(
        np.mean(discount*np.maximum(arithmetic-strike, 0), axis=0),
        np.mean(discount*np.maximum(geometric-strike, 0), axis=0),
        np.mean(discount*arithmetic, axis=0),
        np.mean(discount*stock, axis=0), paired)


def sample_payoffs(rng, dates, count, spot=100., strike=100., rate=.1,
                   volatility=.3, horizon=1., paired=False,
                   innovation='normal'):
    market(spot, strike, rate, volatility, horizon)
    times = observation_dates(dates, horizon)
    count = positive_integer(count, "independent paths or pairs")
    if count < 2 or not isinstance(paired, bool):
        raise ValueError("at least two paths and a boolean pair flag required")
    log_mgf(0., innovation)  # Reject unsupported laws before drawing.
    gaps = np.diff(np.r_[0., times])

    def columns():
        for gap in gaps:
            if gap == 0:
                yield np.zeros(count)  # Initial observation consumes no RNG.
            elif innovation == 'normal':
                yield rng.standard_normal(count)
            else:
                yield 2*rng.integers(0, 2, size=count)-1

    return _stream_payoffs(columns(), times, count, spot, strike, rate,
                           volatility, horizon, paired, innovation)


def pilot_control(dates, seed=2401, pilot=20_000, production=100_000):
    rng = np.random.Generator(np.random.PCG64(seed))
    first = sample_payoffs(rng, dates, pilot, paired=True)
    denominator = float(np.var(first.geometric, ddof=1))
    if denominator <= 0:
        raise ValueError("pilot geometric control has no estimated variance")
    coefficient = float(np.cov(first.arithmetic, first.geometric, ddof=1)[0, 1]
                        / denominator)
    draws = sample_payoffs(rng, dates, production, paired=True)
    known = geometric_price(dates)
    adjusted = draws.arithmetic-coefficient*(draws.geometric-known)
    difference = known+(draws.arithmetic-draws.geometric)
    return {'coefficient': coefficient, 'known_geometric': known,
            'paired': estimate(draws.arithmetic),
            'controlled': estimate(adjusted),
            'unit_control': estimate(difference),
            'controlled_draws': adjusted, 'paired_draws': draws}


def coupled_monitoring(seed=2402, count=50_000, fine_steps=24):
    fine_steps = positive_integer(fine_steps, "fine observations")
    if fine_steps % 2:
        raise ValueError("fine grid must have twice the coarse observations")
    count = positive_integer(count, "coupled paths")
    rng = np.random.Generator(np.random.PCG64(seed))
    logs = np.full(count, math.log(100.))
    fine_sum = np.zeros(count)
    coarse_sum = np.zeros(count)
    fine_logsum = np.zeros(count)
    coarse_logsum = np.zeros(count)
    dt = 1/fine_steps
    for j in range(1, fine_steps+1):
        logs += (.1-.5*.3**2)*dt+.3*math.sqrt(dt)*rng.standard_normal(count)
        stock = np.exp(logs)
        fine_sum += stock
        fine_logsum += logs
        if j % 2 == 0:
            coarse_sum += stock
            coarse_logsum += logs
    fy = math.exp(-.1)*np.maximum(fine_sum/fine_steps-100, 0)
    cy = math.exp(-.1)*np.maximum(coarse_sum/(fine_steps//2)-100, 0)
    fx = math.exp(-.1)*np.maximum(np.exp(fine_logsum/fine_steps)-100, 0)
    cx = math.exp(-.1)*np.maximum(
        np.exp(coarse_logsum/(fine_steps//2))-100, 0)
    exact_difference = (geometric_price(equal_dates(fine_steps)) -
                        geometric_price(equal_dates(fine_steps//2)))
    return {'raw_difference': estimate(fy-cy),
            'unit_control_difference': estimate(
                exact_difference+(fy-fx)-(cy-cx)),
            'geometric_difference': exact_difference}


def coin_enumeration(steps=4):
    steps = positive_integer(steps, "enumerated steps")
    if steps > 12:
        raise ValueError("exact teaching enumeration limited to 12 steps")
    times = equal_dates(steps)
    matrix = np.array(list(itertools.product((-1., 1.), repeat=steps))).T
    draws = _stream_payoffs(matrix, times, matrix.shape[1], 100., 100., .1,
                            .3, 1., False, 'coin')
    return float(np.mean(draws.arithmetic)), float(np.mean(
        draws.discounted_terminal_stock))


def main():
    first = np.array([96., 78., 132., 114.])
    second = np.array([120., 120., 120., 114.])
    for label, path in [('source74', first),
                        ('original_same_terminal', second)]:
        arithmetic = float(np.mean(path))
        geometric = math.exp(float(np.mean(np.log(path))))
        print(f"{label}: A={arithmetic:.12f}, G={geometric:.12f}, "
              f"A-call={max(arithmetic-100, 0):.12f}, "
              f"G-call={max(geometric-100, 0):.12f}, "
              f"terminal-call={max(path[-1]-100, 0):.12f}")
    for m in (1, 4, 12, 50, 100, 250, 500):
        times = equal_dates(m)
        mean, variance = geometric_log_moments(times)
        closed_variance = .3**2*(m+1)*(2*m+1)/(6*m*m)
        assert math.isclose(variance, closed_variance, rel_tol=2e-14)
        covariance_variance = .3**2*float(np.mean(
            np.minimum.outer(times, times)))
        assert math.isclose(variance, covariance_variance, rel_tol=2e-14)
        print(f"m={m}: geometric={geometric_price(times):.12f}, "
              f"logmean={mean:.12f}, logvariance={variance:.12f}")
    continuous_geometric = lognormal_call(
        math.log(100.)+(.1-.5*.3**2)/2, .3**2/3, 100., math.exp(-.1))
    print(f"continuous-monitoring geometric limit={continuous_geometric:.12f}")
    assert abs(geometric_price([1.])-call_price(100, 100, .1, .3, 1)) < 1e-12
    # Independent matrix/cumulative-sum form of the paired R/Python labs.
    times = equal_dates(12)
    rng = np.random.Generator(np.random.PCG64(2407))
    matrix_z = rng.standard_normal((12, 37))
    gaps = np.diff(np.r_[0., times])
    increments = ((.1-.5*.3**2)*gaps[:, None] +
                  .3*np.sqrt(gaps[:, None])*matrix_z)
    log_matrix = math.log(100.)+np.cumsum(increments, axis=0)
    matrix_a = np.mean(np.exp(log_matrix), axis=0)
    matrix_g = np.exp(np.mean(log_matrix, axis=0))
    stream = _stream_payoffs(matrix_z, times, 37, 100., 100., .1,
                             .3, 1., False, 'normal')
    np.testing.assert_allclose(
        stream.arithmetic, math.exp(-.1)*np.maximum(matrix_a-100, 0),
        atol=3e-12, rtol=0)
    np.testing.assert_allclose(
        stream.geometric, math.exp(-.1)*np.maximum(matrix_g-100, 0),
        atol=3e-12, rtol=0)
    dates = equal_dates(250)
    rng = np.random.Generator(np.random.PCG64(2400))
    draws = sample_payoffs(rng, dates, 100_000)
    arithmetic = estimate(draws.arithmetic)
    geometric = estimate(draws.geometric)
    known = geometric_price(dates)
    assert abs(geometric.mean-known) < 6*geometric.se
    assert arithmetic.mean >= geometric.mean-1e-12
    forward = estimate(draws.average_forward)
    assert abs(forward.mean-average_forward_price(dates)) < 6*forward.se
    print(f"IID N100000 m250: arithmetic={arithmetic}; geometric={geometric}")
    print(f"discounted-average mean={forward}; "
          f"analytic={average_forward_price(dates):.12f}")
    controls = pilot_control(dates)
    for key in ('coefficient', 'known_geometric', 'paired',
                'controlled', 'unit_control'):
        print(f"control {key}={controls[key]}")
    assert controls['controlled'].se < controls['paired'].se
    assert controls['unit_control'].mean >= known
    below = int(np.sum(controls['controlled_draws'] < known))
    smallest = float(np.min(controls['controlled_draws']))
    print("fitted control observations below geometric mean / minimum:",
          below, smallest)
    assert below > 0  # Fixed teaching seed; not a universal count.
    print("pilot+production stock updates / fresh normal innovations:",
          2*(20_000+100_000)*dates.size, (20_000+100_000)*dates.size)
    print("coupled monitoring24-minus12:", coupled_monitoring())

    # Initial-S0 inclusion is a distinct contract, with coupled future paths.
    seed = 2403
    ex = sample_payoffs(np.random.Generator(np.random.PCG64(seed)),
                        equal_dates(12), 20_000)
    inc = sample_payoffs(np.random.Generator(np.random.PCG64(seed)),
                         equal_dates(12, include_initial=True), 20_000)
    np.testing.assert_allclose(inc.arithmetic, ex.arithmetic*12/13,
                               atol=3e-12, rtol=0)
    print("initial-observation arithmetic exclude/include:",
          estimate(ex.arithmetic), estimate(inc.arithmetic))
    print("initial-observation geometric exclude/include:",
          geometric_price(equal_dates(12)),
          geometric_price(equal_dates(12, include_initial=True)))
    rare = sample_payoffs(np.random.Generator(np.random.PCG64(2404)),
                          equal_dates(12), 1000, strike=300.)
    print("rare finite sample K300:", estimate(rare.arithmetic),
          "positive analytic geometric lower price:",
          geometric_price(equal_dates(12), strike=300.))
    assert np.all(rare.arithmetic == 0)
    assert geometric_price(equal_dates(12), strike=300.) > 0

    exact_coin, exact_stock = coin_enumeration()
    coin = sample_payoffs(np.random.Generator(np.random.PCG64(2405)),
                          equal_dates(4), 100_000, innovation='coin')
    coin_estimate = estimate(coin.arithmetic)
    assert abs(coin_estimate.mean-exact_coin) < 6*coin_estimate.se
    assert abs(exact_stock-100) < 1e-12
    print("compensated coin m4 exactAsian/discounted-stock:",
          exact_coin, exact_stock, "MC:", coin_estimate)
    zero = sample_payoffs(np.random.Generator(np.random.PCG64(2406)),
                          equal_dates(12), 1000, strike=0., rate=0.)
    constant = 100.+(zero.arithmetic-zero.average_forward)
    assert np.all(constant == 100.)
    print("zero-strike zero-rate identical-control estimate:",
          estimate(constant))
    print("negative-rate zero-strike arithmetic/European prices:",
          average_forward_price(equal_dates(12), rate=-.1), 100.)
    invalid = [
        lambda: sample_payoffs(np.random.default_rng(1), [1., .5], 10),
        lambda: sample_payoffs(np.random.default_rng(1), [.5], 10),
        lambda: sample_payoffs(np.random.default_rng(1), [1.], True),
        lambda: sample_payoffs(np.random.default_rng(1), [1.], 10, strike=-1),
        lambda: sample_payoffs(np.random.default_rng(1), [1.], 10,
                               innovation='student-t'),
        lambda: equal_dates(0),
        lambda: coupled_monitoring(fine_steps=3),
    ]
    for operation in invalid:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid contract accepted")
    print("All contract, sampler, control, moment and boundary checks passed.")


if __name__ == '__main__':
    main()
