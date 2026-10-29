"""Chapter 6: reproducible factor, price-data, and portfolio laboratories."""
import math
import sys
import timeit
import numpy as np

ASSET_LABELS = ('Asset A', 'Asset B', 'Asset C')


def aligned_holdings(labels, positions):
    """Require one unique supplied quantity for every price-column label."""
    if len(labels) != len(set(labels)) or set(labels) != set(positions):
        raise ValueError('price columns and holdings labels must agree uniquely')
    quantities = np.array([positions[name] for name in labels], dtype=float)
    if not np.isfinite(quantities).all():
        raise ValueError('holdings must be finite')
    return quantities


def population_example():
    """Hypothetical common-horizon covariance, not an empirical estimate."""
    market_variance = .02
    beta = np.array([.75, .4, .5])
    residual = np.diag([.01, .02, .015])
    cross = market_variance * beta
    assets = market_variance * np.outer(beta, beta) + residual
    joint = np.block([[assets, cross[:, None]],
                      [cross[None, :], np.array([[market_variance]])]])
    return market_variance, beta, residual, assets, joint


def daily_fixture():
    """Eight original daily observations and nine prices; no corporate actions."""
    factor = .0005 + np.array([-.02, -.02, -.01, -.01,
                               .01, .01, .02, .02])
    alpha = np.array([.0002, -.001, -.0005])
    beta = np.array([.75, .4, .5])
    signs = np.array([[1, 1, 1], [-1, -1, -1],
                      [1, -1, 1], [-1, 1, -1],
                      [1, 1, -1], [-1, -1, 1],
                      [1, -1, -1], [-1, 1, 1]])
    errors = signs * np.array([.002, .003, .004])
    risk_free = .0001
    returns = risk_free + alpha + factor[:, None]*beta + errors
    initial = np.array([100., 80., 120.])
    prices = np.vstack([initial, initial*np.cumprod(1+returns, axis=0)])
    return factor, risk_free, alpha, beta, errors, returns, prices


def returns_from_prices(prices):
    """Complete positive real price panel; rows chronological, columns assets."""
    prices = np.asarray(prices, dtype=float)
    if (prices.ndim != 2 or prices.shape[0] < 2 or prices.shape[1] == 0
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        raise ValueError('need at least two complete positive price rows')
    return prices[1:] / prices[:-1] - 1


def fit_linear(design, observations):
    """Finite real design with numerical full column rank; one/many responses."""
    design = np.asarray(design, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if (design.ndim != 2 or observations.ndim not in (1, 2)
            or design.shape[0] != observations.shape[0]
            or design.shape[1] == 0 or not np.isfinite(design).all()
            or not np.isfinite(observations).all()):
        raise ValueError('incompatible or nonfinite regression data')
    coefficients, _, rank, singular = np.linalg.lstsq(
        design, observations, rcond=None)
    if rank < design.shape[1]:
        raise ValueError('design is rank-deficient at the numerical cutoff')
    residual = observations - design @ coefficients
    return coefficients, residual, singular


def portfolio_cloud(returns, count=10000, seed=42):
    """Long-only constant-weight sample; daily simple-return moments."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(returns.shape[1]), size=count)
    mean = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    means = weights @ mean
    variances = np.einsum('ij,jk,ik->i', weights, covariance, weights)
    return weights, means, np.sqrt(variances)


def benchmark():
    """Optional same-input arithmetic-kernel comparison, excluding printing."""
    u = np.arange(1000, dtype=float)
    v = u[::-1].copy()
    loop = lambda: math.fsum(float(a*b) for a, b in zip(u, v, strict=True))
    native = lambda: np.dot(u, v)
    assert math.isclose(loop(), native(), rel_tol=1e-14)
    for label, operation in (('Python scalar loop', loop), ('NumPy dot', native)):
        times = timeit.repeat(operation, number=1000, repeat=5)
        print(label, 'microseconds per call:',
              ', '.join(f'{1e6*t/1000:.6f}' for t in times))
    print('scope: prepared contiguous1000-element float arrays; timings are local')


def checks():
    variance, beta, residual, covariance, joint = population_example()
    assert np.linalg.eigvalsh(joint).min() > 0
    np.testing.assert_allclose(joint[:3, 3]/variance, beta, atol=1e-14)
    np.testing.assert_allclose(covariance-np.outer(joint[:3, 3],
                                                 joint[:3, 3])/variance,
                               residual, atol=1e-14)
    # Conditional MV pricing example in the augmented universe, not a claim
    # that a three-stock proxy is the all-assets market portfolio.
    mu = np.r_[.05 + .06*beta, .11]
    market_weights = np.array([0., 0., 0., 1.])
    np.testing.assert_allclose(mu-.05, 3*joint@market_weights, atol=1e-14)
    factor, rf, alpha, beta, errors, returns, prices = daily_fixture()
    np.testing.assert_allclose(returns_from_prices(prices), returns, atol=1e-14)
    design = np.column_stack([np.ones(len(factor)), factor])
    fitted, error, _ = fit_linear(design, returns-rf)
    np.testing.assert_allclose(fitted, np.vstack([alpha, beta]), atol=1e-14)
    np.testing.assert_allclose(error, errors, atol=1e-14)
    np.testing.assert_allclose(design.T @ error, 0, atol=1e-14)
    modeled = np.var(factor, ddof=1)*np.outer(beta, beta)
    modeled += errors.T @ errors / (len(factor)-1)
    np.testing.assert_allclose(np.cov(returns, rowvar=False), modeled,
                               atol=1e-14)
    np.testing.assert_allclose(np.sum(np.log1p(returns), axis=0),
                               np.log(prices[-1]/prices[0]), atol=1e-14)
    positions = {'Asset A': 50., 'Asset B': 100., 'Asset C': 30.}
    holdings = aligned_holdings(ASSET_LABELS, positions)
    order = [2, 0, 1]
    reordered = aligned_holdings([ASSET_LABELS[i] for i in order], positions)
    np.testing.assert_allclose(prices @ holdings, prices[:, order] @ reordered)
    w = np.array([1/3]*3)
    exact_log = np.log1p(returns @ w)
    weighted_log = np.log1p(returns) @ w
    assert (exact_log >= weighted_log-1e-15).all()
    weights, means, volatilities = portfolio_cloud(returns)
    assert weights.shape == (10000, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-14)
    assert (weights >= 0).all() and (volatilities > 0).all()
    selected = 37
    series = returns @ weights[selected]
    assert math.isclose(series.mean(), means[selected], abs_tol=1e-14)
    assert math.isclose(series.std(ddof=1), volatilities[selected], abs_tol=1e-14)
    covariance = np.cov(returns, rowvar=False)
    assert (covariance[1, [0, 2]] > covariance[1, 1]).all()
    first, second = np.array([.6, .4, 0.]), np.array([.5, .25, .25])
    assert math.isclose(returns.mean(axis=0)@first,
                        returns.mean(axis=0)@second, abs_tol=1e-14)
    assert first@covariance@first > second@covariance@second
    for invalid in ([[100., 80.], [0., 81.]], [[100., np.nan], [101., 81.]]):
        try:
            returns_from_prices(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('bad prices accepted')
    duplicate = np.column_stack([np.ones(8), factor, factor])
    try:
        fit_linear(duplicate, returns-rf)
    except ValueError:
        pass
    else:
        raise AssertionError('unidentified coefficients accepted')
    print('checks: joint PSD, beta/residual covariance, OLS identification,')
    print('        price/return identities, Jensen gap, seeded cloud: PASS')


def main():
    checks()
    np.set_printoptions(precision=9, suppress=True)
    variance, beta, residual, covariance, joint = population_example()
    print('population beta:', beta, 'market variance:', variance)
    print('asset covariance:\n', covariance)
    print('asset volatilities:', np.sqrt(np.diag(covariance)),
          'market volatility:', math.sqrt(variance))
    print('hypothetical CAPM means, rf5%/market11%:', .05+.06*beta)
    print('invalid covariance block determinant:', .04*2-1.5**2)
    print('minimum compatible asset variance for covariance1.5/marketvariance2:',
          1.5**2/2)
    print('covariance storage, n1000/k3:', 1000*1001//2, 1000*3+6+1000)
    print('dot example:', np.dot([1, 2, 3], [4, 5, 6]))
    print('matvec example:', np.array([[1, 2, 3], [4, 5, 6]]) @ [7, 8, 9])
    cash, dates = np.array([25., 25., 25., 25., 1025.]), np.arange(1, 6)
    for rate in (.02, .05, .01):
        print(f'bond dot at annual{rate:.0%}: {cash@(1+rate)**(-dates):.9f}')
    factor, rf, alpha, beta, errors, returns, prices = daily_fixture()
    design = np.column_stack([np.ones(len(factor)), factor])
    fit, error, singular = fit_linear(design, returns-rf)
    print('daily OLS coefficients, intercept then beta:\n', fit)
    print('design singular values:', singular)
    print('daily sample simple-return means:', returns.mean(axis=0))
    print('daily sample covariance:\n', np.cov(returns, rowvar=False))
    holdings = aligned_holdings(ASSET_LABELS,
                                {'Asset A': 50., 'Asset B': 100., 'Asset C': 30.})
    values = prices @ holdings
    print(f'fixed holdings initial/final: {values[0]:.9f}/{values[-1]:.9f}')
    print('final rebased prices:', 100*prices[-1]/prices[0])
    print('initial holding weights:', holdings*prices[0]/values[0])
    print('final holding weights:', holdings*prices[-1]/values[-1])
    w = np.array([1/3]*3)
    exact = np.log1p(returns@w)
    approximate = np.log1p(returns)@w
    print(f'first-day portfolio log={exact[0]:.12f}, '
          f'weighted asset logs={approximate[0]:.12f}, '
          f'gap={exact[0]-approximate[0]:.12f}')
    flagged = returns.copy()
    flagged[0, 0] = .30
    print('observations flagged above25% simple return:',
          np.argwhere(np.abs(flagged)>.25).tolist(), '(retained, not deleted)')
    weights, means, vols = portfolio_cloud(returns)
    print('cloud seed42/count10000; sampled minimum volatility weight:',
          weights[np.argmin(vols)])
    print(f'sampled minimum daily volatility={vols.min():.12f}, '
          f'mean there={means[np.argmin(vols)]:.12f}')
    print(f'cloud negative sample-mean fraction={np.mean(means<0):.9f}')
    covariance = np.cov(returns, rowvar=False)
    print(f'assetB exact minimum daily variance={covariance[1,1]:.12f}, '
          f'volatility={math.sqrt(covariance[1,1]):.12f}')
    for w in (np.array([.6, .4, 0.]), np.array([.5, .25, .25])):
        print('same-positive-mean candidate:', w,
              f'mean={returns.mean(axis=0)@w:.12f}, '
              f'variance={w@covariance@w:.12f}')
    print('scope: random candidates, not an optimality or forecasting certificate')
    if '--benchmark' in sys.argv:
        benchmark()


if __name__ == '__main__':
    main()
