"""Equality-only one-period portfolios; educational finite-dimensional model.

Inputs are finite decimal expected simple returns and a symmetric strictly
positive-definite covariance matrix in matching return-squared units. Negative
weights are allowed. No estimates, files, orders, or market data are obtained.
Run this file to reproduce the chapter's numerical cases and checks.
"""

from dataclasses import dataclass
from numbers import Real
import math

import numpy as np


def scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def inputs(mu, covariance):
    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mu.ndim != 1 or mu.size < 2:
        raise ValueError("mu must be a vector with at least two assets")
    if covariance.shape != (mu.size, mu.size):
        raise ValueError("covariance shape must match mu")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(covariance)):
        raise ValueError("inputs must be finite")
    scale = float(np.max(np.abs(covariance)))
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12 * scale):
        raise ValueError("covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("this implementation requires positive definiteness") from exc
    return mu, covariance


@dataclass
class Portfolio:
    weights: np.ndarray
    mean: float
    variance: float
    lambda_mean: float
    lambda_budget: float
    constraint_residual: float
    stationarity_residual: float


def equality_portfolio(mu, covariance, target, budget=1.0):
    """Minimize HALF variance with exact mean and budget constraints.

Center/scale the mean constraint before a two-right-hand-side covariance
solve and a two-dimensional Schur solve. Transform multipliers back to the
original constraints. Reject constant means; redundant-constraint and PSD
cases require a different mathematical contract. Return residual diagnostics.
"""
    mu, covariance = inputs(mu, covariance)
    target, budget = scalar(target, "target"), scalar(budget, "budget")
    center = float(np.mean(mu))
    spread = float(np.max(np.abs(mu - center)))
    if spread == 0:
        raise ValueError("mean and budget constraints must be independent")
    one = np.ones(mu.size)
    e = (mu - center) / spread
    constraints = np.column_stack((e, one))
    rhs = np.array([(target - center * budget) / spread, budget])
    y = np.linalg.solve(covariance, constraints)
    gram = constraints.T @ y
    gamma = np.linalg.solve(gram, rhs)
    weights = y @ gamma
    lambda_mean = float(gamma[0] / spread)
    lambda_budget = float(gamma[1] - gamma[0] * center / spread)
    constraint_residual = max(abs(float(mu @ weights) - target),
                              abs(float(one @ weights) - budget))
    stationarity_residual = float(np.max(np.abs(
        covariance @ weights - lambda_mean * mu - lambda_budget * one)))
    if not np.all(np.isfinite(weights)):
        raise ArithmeticError("nonfinite numerical solution")
    return Portfolio(weights, float(mu @ weights),
                     float(weights @ covariance @ weights),
                     lambda_mean, lambda_budget,
                     constraint_residual, stationarity_residual)


def direct_kkt(mu, covariance, target, budget=1.0):
    """Independent unscaled block solve for comparison on ordinary cases."""
    mu, covariance = inputs(mu, covariance)
    e = np.column_stack((mu, np.ones(mu.size)))
    block = np.block([[covariance, -e], [e.T, np.zeros((2, 2))]])
    rhs = np.concatenate((np.zeros(mu.size),
                          [scalar(target, "target"), scalar(budget, "budget")]))
    return np.linalg.solve(block, rhs)


def global_minimum(mu, covariance):
    mu, covariance = inputs(mu, covariance)
    direction = np.linalg.solve(covariance, np.ones(mu.size))
    weights = direction / np.sum(direction)
    return weights, float(mu @ weights), float(weights @ covariance @ weights)


def capital_market(mu, covariance, risk_free, volatility):
    """Efficient risky exposures and cash for a nonnegative target SD.

Allows shorting and borrowing/lending at the SAME one-period simple return.
Does not normalize the direction into a fund, so zero net risky budget is
permitted. Return risky exposures, cash, expected return, and optimal Sharpe.
"""
    mu, covariance = inputs(mu, covariance)
    risk_free = scalar(risk_free, "risk_free")
    volatility = scalar(volatility, "volatility")
    if volatility < 0:
        raise ValueError("volatility must be nonnegative")
    excess = mu - risk_free
    direction = np.linalg.solve(covariance, excess)
    squared_sharpe = float(excess @ direction)
    if squared_sharpe <= 0:
        if volatility == 0:
            return np.zeros(mu.size), 1.0, risk_free, 0.0
        raise ValueError("nonzero excess-return direction required")
    sharpe = math.sqrt(squared_sharpe)
    risky = direction * (volatility / sharpe)
    cash = 1.0 - float(np.sum(risky))
    return risky, cash, risk_free + volatility * sharpe, sharpe


LECTURE_MU = np.array([0.10, 0.08, 0.12])
LECTURE_COV = np.array([[4.0, 1.0, 0.5],
                        [1.0, 2.0, 0.3],
                        [0.5, 0.3, 3.0]])
REVATI_MU = np.array([0.04, 0.08, 0.12])
REVATI_COV = np.diag([0.01, 0.02, 0.04])


def run_checks():
    close = np.testing.assert_allclose
    p = equality_portfolio(LECTURE_MU, LECTURE_COV, 0.10)
    close(p.weights, [1 / 6, 5 / 12, 5 / 12], atol=1e-13)
    close([p.variance, p.lambda_mean, p.lambda_budget],
          [31 / 24, 25 / 3, 11 / 24], atol=1e-12)
    for target in [0.08, 0.09, 0.10, 0.11, 0.12]:
        p = equality_portfolio(LECTURE_MU, LECTURE_COV, target)
        block = direct_kkt(LECTURE_MU, LECTURE_COV, target)
        close(np.r_[p.weights, p.lambda_mean, p.lambda_budget], block,
              rtol=1e-11, atol=1e-11)
        assert p.constraint_residual < 1e-13
        assert p.stationarity_residual < 1e-12
        h = np.array([-2.0, 1.0, 1.0]) * 0.07
        change = ((p.weights + h) @ LECTURE_COV @ (p.weights + h)
                  - p.variance) / 2
        close(change, h @ LECTURE_COV @ h / 2, atol=1e-14)
    p = equality_portfolio(REVATI_MU, REVATI_COV, 0.08)
    close(p.weights, np.array([4, 5, 4]) / 13, atol=1e-14)
    close(p.variance, 1 / 130, atol=1e-14)
    for name, expected in [('target', p.lambda_mean),
                           ('budget', p.lambda_budget)]:
        step = 1e-5
        if name == 'target':
            high = equality_portfolio(REVATI_MU, REVATI_COV, 0.08 + step)
            low = equality_portfolio(REVATI_MU, REVATI_COV, 0.08 - step)
        else:
            high = equality_portfolio(REVATI_MU, REVATI_COV, 0.08, 1 + step)
            low = equality_portfolio(REVATI_MU, REVATI_COV, 0.08, 1 - step)
        close((high.variance - low.variance) / (4 * step), expected,
              rtol=1e-8, atol=1e-12)
    low = equality_portfolio(REVATI_MU, REVATI_COV, 0.04)
    high = equality_portfolio(REVATI_MU, REVATI_COV, 0.12)
    close((low.weights + high.weights) / 2, p.weights, atol=1e-14)
    nik_mu = [0.07, 0.13]
    nik_cov = [[0.01, 0.005], [0.005, 0.04]]
    close(global_minimum(nik_mu, nik_cov)[0], [7 / 8, 1 / 8])
    close(equality_portfolio(nik_mu, nik_cov, 0.10).weights, [0.5, 0.5])
    x, cash, mean, sharpe = capital_market(REVATI_MU, REVATI_COV, 0.02, 0.06)
    close(x @ REVATI_COV @ x, 0.06 ** 2, atol=1e-15)
    close(cash + np.sum(x), 1.0, atol=1e-15)
    close(mean, 0.02 + 0.06 * math.sqrt(0.47), atol=1e-14)
    close(sharpe, math.sqrt(0.47))
    x, cash, mean, sharpe = capital_market([0.01, 0.03], np.eye(2), 0.02, 0.1)
    close(np.sum(x), 0, atol=1e-14)
    close(cash, 1.0)
    assert mean > 0.02
    nearly_equal = 0.10 + 1e-5 * np.array([-1.0, 0, 1.0])
    leveraged = equality_portfolio(nearly_equal, 0.04 * np.eye(3), 0.11)
    close(leveraged.weights, [1 / 3 - 500, 1 / 3, 1 / 3 + 500], atol=1e-8)
    for mu, cov, target in [([0.1, 0.1], np.eye(2), 0.1),
                            ([0.1, 0.2], [[1, 2], [2, 1]], 0.15),
                            ([0.1, 0.2], [[1, 0], [0, 0]], 0.15),
                            ([0.1, 0.2], [[1, 0.2], [0, 1]], 0.15),
                            ([0.1, math.nan], np.eye(2), 0.15),
                            ([0.1, 0.2], np.eye(2), True)]:
        try:
            equality_portfolio(mu, cov, target)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid contract was accepted")
    print("Checks passed: known answers, independent KKT, feasible perturbations,")
    print("half-variance sensitivities, two funds, cash risk, zero-net direction,")
    print("near-dependent means, and rejected inputs.")


def main():
    np.set_printoptions(precision=12, suppress=True)
    for label, mu, cov in [('LECTURE', LECTURE_MU, LECTURE_COV),
                           ('REVATI', REVATI_MU, REVATI_COV)]:
        print(label)
        targets = [0.08, 0.09, 0.10, 0.11, 0.12] if label == 'LECTURE' else [0.04, 0.08, 0.12]
        for target in targets:
            p = equality_portfolio(mu, cov, target)
            print('target', target, 'weights', p.weights, 'variance', p.variance,
                  'sd', math.sqrt(p.variance), 'lambda/nu',
                  p.lambda_mean, p.lambda_budget)
        print('GMV', global_minimum(mu, cov))
        rf = 0.05 if label == 'LECTURE' else 0.02
        direction = np.linalg.solve(cov, mu - rf)
        fund = direction / np.sum(direction)
        fund_sd = math.sqrt(float(fund @ cov @ fund))
        print('direction', direction, 'sum', np.sum(direction), 'tangent', fund,
              'mean', mu @ fund, 'sd', fund_sd,
              'Sharpe', (mu @ fund - rf) / fund_sd)
        if label == 'REVATI':
            print('client SD .06', capital_market(mu, cov, rf, 0.06),
                  'fund fraction', 0.06 / fund_sd)
    p = equality_portfolio(0.10 + 1e-5 * np.array([-1.0, 0, 1.0]),
                           0.04 * np.eye(3), 0.11)
    print('NEAR CONSTANT', p, 'gross', np.sum(np.abs(p.weights)))
    run_checks()


if __name__ == '__main__':
    main()
