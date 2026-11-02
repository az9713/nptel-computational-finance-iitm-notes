"""Sampling, exact GBM transitions, and continuously monitored barrier checks.

Synthetic moderate-scale fixtures; no market data, files, or network access.
NumPy generates pseudorandom samples. Analytic checks use independent moments,
normal probabilities, and a reflection/image calculation for the barrier.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from statistics import NormalDist

import numpy as np


NORMAL = NormalDist()


def beta_square_moments():
    """Integrate the polynomial density for E[X²] and Var(X²) exactly."""
    mean = 6 * (Fraction(1, 4) - Fraction(1, 5))
    second = 6 * (Fraction(1, 6) - Fraction(1, 7))
    return mean, second - mean**2


def positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def model_inputs(spot, drift, volatility, horizon):
    if not all(math.isfinite(x) for x in (spot, drift, volatility, horizon)):
        raise ValueError("model inputs must be finite real scalars")
    if spot <= 0 or volatility <= 0 or horizon <= 0:
        raise ValueError("spot, volatility and horizon must be positive")


@dataclass(frozen=True)
class Estimate:
    mean: float
    se: float
    count: int


def estimate(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("at least two finite scalar observations required")
    return Estimate(float(np.mean(x)), float(np.std(x, ddof=1) /
                    math.sqrt(x.size)), int(x.size))


def walk_terminal(rng, size, steps, magnitude):
    size = positive_integer(size, "size")
    steps = positive_integer(steps, "steps")
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("step magnitude must be finite and positive")
    # Counts have the terminal law of the independent fair coin steps.
    return magnitude * (2 * rng.binomial(steps, .5, size=size) - steps)


def gbm_terminal(rng, size, spot, drift, volatility, horizon):
    size = positive_integer(size, "size")
    model_inputs(spot, drift, volatility, horizon)
    z = rng.standard_normal(size)
    with np.errstate(over='raise', invalid='raise', under='raise'):
        return spot * np.exp((drift - .5 * volatility**2) * horizon +
                             volatility * math.sqrt(horizon) * z)


def gbm_path_from_normals(spot, drift, volatility, horizon, normals):
    model_inputs(spot, drift, volatility, horizon)
    z = np.asarray(normals, dtype=float)
    if z.ndim != 1 or z.size < 1 or not np.all(np.isfinite(z)):
        raise ValueError("a nonempty finite normal-draw vector is required")
    dt = horizon / z.size
    increments = ((drift - .5 * volatility**2) * dt +
                  volatility * math.sqrt(dt) * z)
    with np.errstate(over='raise', invalid='raise', under='raise'):
        return np.r_[spot, spot * np.exp(np.cumsum(increments))]


def inverse_terminal(uniforms, spot, drift, volatility, horizon):
    model_inputs(spot, drift, volatility, horizon)
    u = np.asarray(uniforms, dtype=float)
    if (u.ndim != 1 or not np.all(np.isfinite(u)) or
            np.any(u <= 0) or np.any(u >= 1)):
        raise ValueError("uniforms must lie strictly between zero and one")
    z = np.array([NORMAL.inv_cdf(float(x)) for x in u])
    with np.errstate(over='raise', invalid='raise', under='raise'):
        return spot * np.exp((drift - .5 * volatility**2) * horizon +
                             volatility * math.sqrt(horizon) * z)


def beta_rejection(rng, proposals, bound=1.5):
    """Return all acceptances from a fixed proposal count, plus its mask."""
    proposals = positive_integer(proposals, "proposals")
    if not math.isfinite(bound) or bound < 1.5:
        raise ValueError("Beta(2,2)/Uniform requires a global bound >= 1.5")
    candidate = rng.random(proposals)
    uniforms = rng.random(proposals)
    accepted = uniforms <= 6 * candidate * (1 - candidate) / bound
    return candidate[accepted], accepted


def bridge_survival(terminal, spot, barrier, volatility, horizon):
    model_inputs(spot, 0., volatility, horizon)
    if not math.isfinite(barrier) or not 0 < barrier < spot:
        raise ValueError("the positive barrier must be below the initial spot")
    x = np.asarray(terminal, dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("terminal prices must be finite and positive")
    d = math.log(spot / barrier)
    y = np.maximum(np.log(x / barrier), 0.)
    return -np.expm1(-2 * d * y / (volatility**2 * horizon))


def barrier_benchmarks(spot, strike, rate, volatility, horizon, barrier):
    """Zero-rebate image solution; require strike >= barrier."""
    model_inputs(spot, rate, volatility, horizon)
    if (not all(math.isfinite(v) for v in (strike, barrier)) or
            not 0 < barrier < spot or strike < barrier):
        raise ValueError("require 0 < barrier < spot and strike >= barrier")

    def call(s):
        scale = volatility * math.sqrt(horizon)
        d1 = (math.log(s / strike) +
              (rate + .5 * volatility**2) * horizon) / scale
        return s * NORMAL.cdf(d1) - strike * math.exp(-rate * horizon) * \
            NORMAL.cdf(d1 - scale)

    d = math.log(spot / barrier)
    a = rate - .5 * volatility**2
    scale = volatility * math.sqrt(horizon)
    factor = math.exp(-2 * a * d / volatility**2)
    survival = (NORMAL.cdf((d + a * horizon) / scale) -
                factor * NORMAL.cdf((-d + a * horizon) / scale))
    price = call(spot) - factor * call(barrier**2 / spot)
    return survival, price, call(spot)


def main():
    print("Deterministic workshop checks")
    for steps in (20, 30):
        print(f"tree {steps}: nodes={steps+1}, paths={2**steps}, "
              f"average multiplicity={2**steps/(steps+1):.6f}")
    for dimension in (2, 4, 10):
        print(f"tensor 50^{dimension} = {50**dimension}")
    print(f"SE 0.6 -> 0.15: N={2500*(.6/.15)**2:.0f}")
    print("walk900: mean=0 variance=36 sd=6")
    print("scaled T4/M1600: dt=.0025 step=.05 variance=4 sd=2")
    print("Brownian increment .36 to1: variance=.64 sd=.8")
    print("workshop GBM: logmean=.1 logvariance=.32 "
          f"E[ST]={40*math.exp(.26):.9f}")
    exact_z = np.array([-1., 0., 1., 2.])
    ideal = 80 * np.exp(.04 + .2 * exact_z)
    rounded = inverse_terminal([.1587, .5, .8413, .9772], 80, .06, .2, 1)
    print("inverse ideal-z prices:", np.array2string(ideal, precision=9))
    print("inverse rounded-U prices:", np.array2string(rounded, precision=9))
    path = gbm_path_from_normals(80, .06, .2, 1, [.9, .5])
    combined = (.9 + .5) / math.sqrt(2)
    np.testing.assert_allclose(path[-1], 80 * math.exp(.04 + .2 * combined),
                               rtol=1e-14)
    print(f"two-step: S(.5)={path[1]:.9f} S(1)={path[-1]:.9f} "
          f"Z={combined:.9f}")
    ratios = np.array([.62, .18, .91, .07, .45])
    uniforms = np.array([.33, .55, .20, .88, .41])
    assert np.array_equal(np.flatnonzero(uniforms <= ratios) + 1, [1, 3, 5])
    print("source rejection: accepted1,3,5; realized=.6; "
          "expected=.5 if valid M2")

    rng = np.random.Generator(np.random.PCG64(2201))
    walks = walk_terminal(rng, 100_000, 900, .2)
    walk = estimate(walks)
    assert abs(walk.mean) < 6 * walk.se
    assert abs(np.var(walks, ddof=1) - 36) < .8
    print(f"walk seeded: mean={walk.mean:.9f} "
          f"variance={np.var(walks, ddof=1):.9f}")
    samples, mask = beta_rejection(rng, 60_000)
    beta = estimate(samples**2)
    assert abs(beta.mean - .3) < 6 * beta.se
    assert abs(np.mean(mask) - 2/3) < .01
    print(f"beta: accepted={mask.sum()}/60000 mean(X^2)={beta.mean:.9f} "
          f"SE={beta.se:.9f} exact=.3")
    beta_exact_mean, beta_exact_variance = beta_square_moments()
    beta_variance = float(beta_exact_variance)
    assert beta_exact_mean == Fraction(3, 10)
    assert beta_exact_variance == Fraction(37, 700)
    assert abs(np.var(samples**2, ddof=1) - beta_variance) < .002
    print(f"beta exact variance(X^2)={beta_variance:.12f}; "
          f"N10000 SE={math.sqrt(beta_variance/10000):.12f}")
    print("exercise: Chebyshev N10000 eps.01="
          f"{beta_variance/10000/.01**2:.12f}; "
          f"normal95 halfwidth={1.96*math.sqrt(beta_variance/10000):.12f}")
    print(f"exercise: central20 multiplicity={math.comb(20, 10)}; "
          f"fixed .2 step M1600 variance={1600*.2**2:.9f}; "
          f"dt step variance={1600*.0025**2:.9f}")
    print(f"exercise: unnormalized beta Z={1/6:.12f}, C={1/4:.12f}, "
          f"acceptance={(1/6)/(1/4):.12f}")

    # Exact marginal prices under Q: r=.1, not a historical expected return.
    rng = np.random.Generator(np.random.PCG64(2202))
    terminal = gbm_terminal(rng, 400_000, 100, .1, .3, 1)
    h = bridge_survival(terminal, 100, 80, .3, 1)
    discounted = math.exp(-.1) * np.maximum(terminal - 100, 0)
    survival, price, vanilla = barrier_benchmarks(100, 100, .1, .3, 1, 80)
    weighted = estimate(discounted * h)
    assert abs(weighted.mean - price) < 6 * weighted.se
    assert abs(float(np.mean(h)) - survival) < .004
    survivors = rng.random(terminal.size) <= h
    naive = float(np.mean(discounted[survivors]))
    restored = naive * float(np.mean(survivors))
    np.testing.assert_allclose(restored, np.mean(discounted * survivors),
                               rtol=1e-14)
    print(f"barrier exact: p={survival:.12f} call={price:.12f} "
          f"vanilla={vanilla:.12f} conditional={price/survival:.12f}")
    print(f"barrier weighted MC: {weighted.mean:.9f} SE={weighted.se:.9f}")
    print(f"survival-accepted: naive={naive:.9f} restored={restored:.9f} "
          f"acceptance={np.mean(survivors):.9f}")
    near, _, _ = barrier_benchmarks(100, 100, .1, .3, 1, 95)
    print(f"barrier95: survival={near:.12f} "
          f"required normalized-envelope bound={1/near:.9f}")
    print(f"lab GBM mean={100*math.exp(.1):.9f} "
          f"sd={100*math.exp(.1)*math.sqrt(math.expm1(.09)):.9f}")
    print(f"P(ST>100)={NORMAL.cdf(.055/.3):.9f}; "
          f"P(ST>106)={NORMAL.cdf((.055-math.log(1.06))/.3):.9f}")
    failures = [
        lambda: walk_terminal(rng, True, 4, 1),
        lambda: gbm_terminal(rng, 2, -1, .1, .3, 1),
        lambda: gbm_path_from_normals(100, .1, .3, 1, []),
        lambda: inverse_terminal([0, .5], 100, .1, .3, 1),
        lambda: beta_rejection(rng, 10, 1.3),
        lambda: estimate([1]),
        lambda: bridge_survival([100], 100, 101, .3, 1),
        lambda: barrier_benchmarks(100, 70, .1, .3, 1, 80),
    ]
    for operation in failures:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("an invalid contract was silently accepted")
    print(f"Rejected {len(failures)} invalid-input contracts.")
    print("Checks passed. Seeded output is a teaching run, "
          "not a source trace.")


if __name__ == '__main__':
    main()
