"""European Monte Carlo, normal-return inference and admissible log utility.

Synthetic pricing inputs and a reconstructed rounded sufficient-statistic
benchmark. No market data, network access or output files.
"""
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch22_sampling import estimate, positive_integer  # noqa: E402

NORMAL = NormalDist()


def call_price(spot, strike, rate, volatility, horizon):
    if not all(math.isfinite(x) for x in
               (spot, strike, rate, volatility, horizon)):
        raise ValueError("pricing inputs must be finite")
    if min(spot, strike, volatility, horizon) <= 0:
        raise ValueError("spot, strike, volatility, horizon must be positive")
    width = volatility * math.sqrt(horizon)
    d1 = (math.log(spot/strike) +
          (rate+.5*volatility**2)*horizon)/width
    return (spot*NORMAL.cdf(d1) -
            strike*math.exp(-rate*horizon)*NORMAL.cdf(d1-width))


def discounted_call_and_stock(z, spot=100., strike=100., rate=.1,
                              volatility=.3, horizon=1.):
    z = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("normal draws must be finite")
    call_price(spot, strike, rate, volatility, horizon)
    stock = spot*np.exp((rate-.5*volatility**2)*horizon +
                        volatility*math.sqrt(horizon)*z)
    discount = math.exp(-rate*horizon)
    return discount*np.maximum(stock-strike, 0), discount*stock


def antithetic_pairs(z):
    a, stock_a = discounted_call_and_stock(z)
    b, stock_b = discounted_call_and_stock(-np.asarray(z))
    return (a+b)/2, (stock_a+stock_b)/2


def independent_pilot_control(seed=2301, pilot=20_000, production=100_000):
    pilot = positive_integer(pilot, "pilot pairs")
    production = positive_integer(production, "production pairs")
    if min(pilot, production) < 2:
        raise ValueError("at least two pilot and production pairs required")
    rng = np.random.Generator(np.random.PCG64(seed))
    y0, x0 = antithetic_pairs(rng.standard_normal(pilot))
    coefficient = float(np.cov(y0, x0, ddof=1)[0, 1] /
                        np.var(x0, ddof=1))
    y, x = antithetic_pairs(rng.standard_normal(production))
    adjusted = y-coefficient*(x-100)
    return coefficient, estimate(y), estimate(adjusted)


@dataclass(frozen=True)
class NormalPosterior:
    count: int
    mean: float
    sample_sd: float
    alpha: float
    beta: float
    sse: float


def posterior(count, mean, sample_sd):
    count = positive_integer(count, "data count")
    if count < 2 or not math.isfinite(mean):
        raise ValueError("at least two returns and a finite mean are required")
    if not math.isfinite(sample_sd) or sample_sd <= 0:
        raise ValueError("sample SD must be finite and strictly positive")
    sse = (count-1)*sample_sd**2
    return NormalPosterior(count, mean, sample_sd, (count-1)/2, sse/2, sse)


def posterior_moments(model):
    if model.alpha <= 1:
        raise ValueError("these posterior second moments require count > 3")
    expected_variance = model.beta/(model.alpha-1)
    expected_sigma = math.exp(.5*math.log(model.beta) +
                              math.lgamma(model.alpha-.5) -
                              math.lgamma(model.alpha))
    return {"mu_mean": model.mean,
            "mu_sd": math.sqrt(expected_variance/model.count),
            "sigma_mean": expected_sigma,
            "sigma_sd": math.sqrt(expected_variance-expected_sigma**2),
            "variance_mean": expected_variance}


def iid_posterior(rng, model, size):
    size = positive_integer(size, "posterior draws")
    variance = model.beta/rng.gamma(model.alpha, 1., size=size)
    mu = model.mean+np.sqrt(variance/model.count)*rng.standard_normal(size)
    return mu, variance


def log_target_mu_logv(mu, logv, model):
    """Log target w.r.t. dmu dlog(v), including its Jacobian."""
    if not math.isfinite(mu) or not math.isfinite(logv):
        return -math.inf
    try:
        residual = model.sse+model.count*(mu-model.mean)**2
        return -.5*model.count*logv-.5*residual*math.exp(-logv)
    except OverflowError:
        return -math.inf


def metropolis(rng, model, iterations=20_000, warmup=2000,
               initial_mu=None, variance_factor=1., mu_step=.006,
               logv_step=.22):
    iterations = positive_integer(iterations, "iterations")
    if (not isinstance(warmup, int) or isinstance(warmup, bool) or
            not 0 <= warmup < iterations):
        raise ValueError("warmup must be an integer below iterations")
    if not all(math.isfinite(x) and x > 0
               for x in (variance_factor, mu_step, logv_step)):
        raise ValueError("positive finite proposal scales are required")
    mu = model.mean if initial_mu is None else float(initial_mu)
    logv = math.log(model.sample_sd**2*variance_factor)
    current = log_target_mu_logv(mu, logv, model)
    if not math.isfinite(current):
        raise ValueError("initial state must have finite log target")
    states = np.empty((iterations, 2))
    accepted = np.zeros(iterations, dtype=bool)
    for i in range(iterations):
        z = rng.standard_normal(2)
        proposal_mu, proposal_logv = mu+mu_step*z[0], logv+logv_step*z[1]
        proposed = log_target_mu_logv(proposal_mu, proposal_logv, model)
        uniform = float(rng.random())
        log_uniform = -math.inf if uniform == 0 else math.log(uniform)
        if log_uniform < min(0., proposed-current):
            mu, logv, current = proposal_mu, proposal_logv, proposed
            accepted[i] = True
        states[i] = mu, logv  # Rejection repeats the current state.
    return states[warmup:], float(np.mean(accepted)), accepted


def batch_mean_se(values, block=300):
    """Nonoverlapping batch diagnostic; not a finite-run convergence proof."""
    values = np.asarray(values, dtype=float)
    block = positive_integer(block, "batch length")
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("finite scalar chain values required")
    batches = values.size//block
    if batches < 20:
        raise ValueError("at least twenty complete batches are required")
    means = values[:batches*block].reshape(batches, block).mean(axis=1)
    return float(np.std(means, ddof=1)/math.sqrt(batches))


def manual_mh_trace():
    kernel = {.008: 2.4, .010: 2., .011: 2.2, .012: 1.6, .014: .8}
    current = .010
    trace = []
    for candidate, uniform in zip((.012, .008, .014, .011),
                                  (.53, .91, .67, .24)):
        probability = min(1., kernel[candidate]/kernel[current])
        accepted = uniform <= probability
        if accepted:
            current = candidate
        trace.append((candidate, probability, uniform, accepted, current))
    return trace


def log_gross_wealth(weight, returns, safe_log_return):
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("unbounded log returns require 0 <= weight <= 1")
    returns = np.asarray(returns, dtype=float)
    if not np.all(np.isfinite(returns)) or not math.isfinite(safe_log_return):
        raise ValueError("finite log returns are required")
    if weight == 0:
        return np.full_like(returns, safe_log_return)
    if weight == 1:
        return returns.copy()
    return np.logaddexp(math.log1p(-weight)+safe_log_return,
                        math.log(weight)+returns)


def student_pdf(x, degrees):
    x = np.asarray(x, dtype=float)
    constant = math.exp(math.lgamma((degrees+1)/2) -
                        math.lgamma(degrees/2)) / math.sqrt(degrees*math.pi)
    return constant*np.exp(-.5*(degrees+1)*np.log1p(x*x/degrees))


def student_even_moment(degrees, order):
    if order <= 0 or order % 2 or degrees <= order:
        raise ValueError("finite positive even moment order below df required")
    result = 1.
    for j in range(1, order//2+1):
        result *= degrees*(2*j-1)/(degrees-2*j)
    return result


def utility_quadrature(model, safe_log_return=.008, nodes=256, cutoff=16.):
    """Interior utility benchmark; requires an interior root in [.2, .8].

    The tail bounds do not certify quadrature or floating-point errors.
    The Student-t calculation needs the finite twentieth moment.
    """
    nodes = positive_integer(nodes, "quadrature nodes")
    if not math.isfinite(safe_log_return):
        raise ValueError("safe log return must be finite")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("positive finite cutoff required")
    grid, weights = np.polynomial.legendre.leggauss(nodes)
    z = cutoff*grid
    df = model.count-1
    scale = model.sample_sd*math.sqrt(1+1/model.count)
    predictive_weights = cutoff*weights*student_pdf(z, df)
    normal_weights = cutoff*weights*np.exp(-z*z/2)/math.sqrt(2*math.pi)
    predictive_returns = model.mean+scale*z
    plugin_returns = model.mean+model.sample_sd*z
    moment = student_even_moment(df, 20)
    probability_bound = moment/cutoff**20
    absolute_first_tail = moment/cutoff**19
    utility_tail_bound = ((abs(safe_log_return)+abs(model.mean)) *
                          probability_bound + scale*absolute_first_tail)

    def value(weight, predictive=True):
        r = predictive_returns if predictive else plugin_returns
        w = predictive_weights if predictive else normal_weights
        return float(np.dot(w, log_gross_wealth(weight, r, safe_log_return)))

    def derivative(weight, predictive=True):
        r = predictive_returns if predictive else plugin_returns
        w = predictive_weights if predictive else normal_weights
        # Moderate-cutoff inputs only; public log-wealth evaluation is stable.
        relative = np.expm1(r-safe_log_return)
        return float(np.dot(w, relative/(1+weight*relative)))

    def optimum(predictive):
        # A full Student-t predictive law has infinite exponential moments;
        # truncated endpoint derivatives must not select an exact endpoint.
        low, high = .2, .8
        omitted = (probability_bound if predictive else
                   math.erfc(cutoff/math.sqrt(2)))
        if (derivative(low, predictive) <= 5*omitted or
                derivative(high, predictive) >= -5*omitted):
            raise ValueError("the declared interior bracket is unavailable")
        for _ in range(45):
            middle = (low+high)/2
            if derivative(middle, predictive) > 0:
                low = middle
            else:
                high = middle
        return (low+high)/2

    return {"value": value, "derivative": derivative,
            "predictive_optimum": optimum(True),
            "plugin_optimum": optimum(False),
            "predictive_mass": float(np.sum(predictive_weights)),
            "predictive_tail_probability_bound": probability_bound,
            "predictive_utility_tail_bound": utility_tail_bound,
            "predictive_derivative_tail_bound_on_bracket": 5*probability_bound,
            "predictive_scale": scale}


def finite_utility(weight, returns, probabilities, safe_log_return=.02):
    if not math.isfinite(weight) or not math.isfinite(safe_log_return):
        raise ValueError("weight and safe log return must be finite")
    returns = np.asarray(returns, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (returns.ndim != 1 or returns.shape != probabilities.shape or
            not np.all(np.isfinite(returns)) or
            not np.all(np.isfinite(probabilities)) or
            np.any(probabilities <= 0) or
            not math.isclose(float(np.sum(probabilities)), 1.,
                             rel_tol=0., abs_tol=1e-14)):
        raise ValueError("finite positive normalized probabilities required")
    gross = ((1-weight)*math.exp(safe_log_return) +
             weight*np.exp(returns))
    if np.any(gross <= 0) or not np.all(np.isfinite(gross)):
        raise ValueError("wealth must be positive in every possible state")
    return float(np.dot(probabilities, np.log(gross)))


def main():
    exact = call_price(100, 100, .1, .3, 1)
    rng = np.random.Generator(np.random.PCG64(2300))
    payoffs, _ = discounted_call_and_stock(rng.standard_normal(1_000_000))
    print(f"European exact={exact:.12f}")
    for n in (100, 1000, 10_000, 100_000, 1_000_000):
        result = estimate(payoffs[:n])
        print(f"IID N={n}: price={result.mean:.9f}, "
              f"SE={result.se:.9f}, error={result.mean-exact:.9f}")
    coefficient, paired, controlled = independent_pilot_control()
    assert abs(controlled.mean-exact) < 6*controlled.se
    assert controlled.se < paired.se
    print(f"antithetic100000pairs: price={paired.mean:.9f} SE={paired.se:.9f}")
    print(f"independent-pilot control b={coefficient:.9f}: "
          f"price={controlled.mean:.9f} SE={controlled.se:.9f}")
    workshop = math.exp(-.06)*np.array([0., 0., 6., 20., 44.])
    result = estimate(workshop)
    print(f"source74: discounted mean={result.mean:.9f} SE={result.se:.9f}; "
          f"raw payoff variance={np.var([0., 0., 6., 20., 44.], ddof=1):.9f}")
    print("source74 pilot plan for SE=.1: totalN="
          f"{math.ceil(np.var(workshop, ddof=1)/.1**2)}")
    workshop_exact = call_price(94, 90, .06, .3, 1)
    print(f"source74 analytic price={workshop_exact:.12f}; "
          f"difference/estimated SE="
          f"{(workshop_exact-result.mean)/result.se:.9f}")
    print("source74 MH trace:", manual_mh_trace())
    assert [row[-1] for row in manual_mh_trace()] == [.012, .008, .008, .011]

    model = posterior(60, .00781, .0402)
    analytic = posterior_moments(model)
    print("reconstructed rounded-statistic benchmark:", model)
    print("analytic posterior moments:", analytic)
    rng = np.random.Generator(np.random.PCG64(2302))
    mu, variance = iid_posterior(rng, model, 100_000)
    print(f"IID posterior: mu={np.mean(mu):.9f}, "
          f"sigma={np.mean(np.sqrt(variance)):.9f}")
    assert abs(np.mean(mu)-analytic["mu_mean"]) < 6*estimate(mu).se
    rows = []
    for index, (offset, factor) in enumerate(((-.03, .5), (.03, 2.))):
        rng = np.random.Generator(np.random.PCG64(2303+index))
        chain, acceptance, accepted = metropolis(
            rng, model, initial_mu=model.mean+offset, variance_factor=factor)
        sigmas = np.exp(chain[:, 1]/2)
        mu_se, sigma_se = batch_mean_se(chain[:, 0]), batch_mean_se(sigmas)
        assert abs(np.mean(chain[:, 0])-analytic["mu_mean"]) < 8*mu_se
        assert abs(np.mean(sigmas)-analytic["sigma_mean"]) < 8*sigma_se
        repeated = np.all(chain[1:] == chain[:-1], axis=1)
        assert np.any(repeated)
        rows.append(chain)
        print(f"MH chain{index+1}: acceptance={acceptance:.6f}, "
              f"mu={np.mean(chain[:, 0]):.9f} batchSE={mu_se:.9f}, "
              f"sigma={np.mean(sigmas):.9f} batchSE={sigma_se:.9f}")
    assert len(rows) == 2

    quadrature = utility_quadrature(model)
    refined = utility_quadrature(model, nodes=512)
    for key in ("predictive_optimum", "plugin_optimum"):
        assert abs(quadrature[key]-refined[key]) < 1e-7
    print("original monthly safe log return=.008; long-only utility")
    print(f"plugin optimum={quadrature['plugin_optimum']:.9f}; "
          f"predictive optimum={quadrature['predictive_optimum']:.9f}")
    print(f"predictive Student-t df59 scale="
          f"{quadrature['predictive_scale']:.12f}; "
          f"mass={quadrature['predictive_mass']:.15f}")
    print(f"tail bounds: probability="
          f"{quadrature['predictive_tail_probability_bound']:.3e}, "
          f"uniform utility={quadrature['predictive_utility_tail_bound']:.3e}")
    for weight in (0., quadrature["plugin_optimum"],
                   quadrature["predictive_optimum"], 1.):
        plugin_value = quadrature['value'](weight, False)
        predictive_value_at_weight = quadrature['value'](weight, True)
        print(f"w={weight:.9f}: pluginEU={plugin_value:.12f}; "
              f"predictiveEU={predictive_value_at_weight:.12f}")
    # Conditional future-return draws from each retained posterior state.
    # Correlation in the states remains; use batches, not IID SE.
    rng = np.random.Generator(np.random.PCG64(2305))
    predictions = []
    for chain in rows:
        predictions.append(chain[:, 0] + np.exp(chain[:, 1]/2) *
                           rng.standard_normal(chain.shape[0]))
    future = np.concatenate(predictions)
    utilities = log_gross_wealth(.4, future, .008)
    predictive_value = quadrature['value'](.4, True)
    utility_se = batch_mean_se(utilities)
    assert abs(np.mean(utilities)-predictive_value) < 8*utility_se
    print(f"MH predictive w=.4: EU={np.mean(utilities):.12f} "
          f"batchSE={utility_se:.12f}, quadrature={predictive_value:.12f}")
    for weight in (1., 3.):
        point = finite_utility(weight, [-.1, .3], [.5, .5])
        uncertain = finite_utility(weight, [-.25, .15, .05, .45], [.25]*4)
        print(f"finite source74 w={weight:g}: pointEU={point:.12f}, "
              f"mixtureEU={uncertain:.12f}")
    invalid = [
        lambda: posterior(1, 0., .04),
        lambda: posterior(60, 0., 0.),
        lambda: log_gross_wealth(1.1, [0.], .008),
        lambda: finite_utility(5., [-.25, .45], [.5, .5]),
        lambda: batch_mean_se([1., 2., 3.]),
        lambda: utility_quadrature(model, safe_log_return=-.2),
    ]
    for operation in invalid:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid teaching contract was accepted")
    print(f"Rejected {len(invalid)} invalid-input/unsupported-bracket cases.")
    print("All checks passed; traces/acceptance are not convergence proofs.")


if __name__ == '__main__':
    main()
