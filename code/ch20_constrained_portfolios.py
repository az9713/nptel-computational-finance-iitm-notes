"""Small, auditable inequality examples; not a general production QP solver.

The long_only/barrier functions use the fixed three-asset case from Chapter 19;
only their target changes. sara_case is a separate fixed workshop fixture.
The log-barrier solver uses analytic derivatives and feasible Newton steps;
no R or external QP runtime is implied.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch19_mean_variance import (LECTURE_MU as MU, LECTURE_COV as COV,  # noqa: E402
                                scalar, equality_portfolio)


V = np.array([-2.0, 1.0, 1.0])
E = np.column_stack(((MU - 0.10) / 0.02, np.ones(3)))
SARA_MU = np.array([0.05, 0.09, 0.15])
SARA_COV = np.array([[0.04, 0.008, 0.004],
                     [0.008, 0.0625, 0.01],
                     [0.004, 0.01, 0.1225]])


def sara_case():
    """Independent workshop fixture: candidate active face plus KKT witness."""
    unrestricted = equality_portfolio(SARA_MU, SARA_COV, 0.13)
    weights = np.array([0.0, 1 / 3, 2 / 3])
    constraints = np.column_stack((SARA_MU, np.ones(3)))
    gradient = SARA_COV @ weights
    eta = np.linalg.solve(constraints[1:], gradient[1:])
    z = gradient - constraints @ eta
    return unrestricted, weights, float(weights @ SARA_COV @ weights), eta, z


def feasible_interval(target):
    target = scalar(target, "target")
    if target < 0.08 or target > 0.12:
        raise ValueError("long-only target must lie in [0.08, 0.12]")
    k = float(np.clip((target - 0.10) / 0.02, -1.0, 1.0))
    if target == 0.08:
        k = -1.0
    if target == 0.12:
        k = 1.0
    base = np.array([1.0 - k, 0.0, k])
    return base, max(0.0, -k), (1.0 - k) / 2.0


@dataclass
class LongOnly:
    target: float
    weights: np.ndarray
    variance: float
    coordinate: float
    unrestricted_coordinate: float
    lower: float
    upper: float
    directional_gradient: float
    scalar_certificate: bool


def long_only(target):
    """Exact scalar reduction of this particular three-asset convex QP."""
    target = scalar(target, "target")
    base, lower, upper = feasible_interval(target)
    curvature = float(V @ COV @ V)
    unrestricted = -float(V @ COV @ base) / curvature
    coordinate = float(np.clip(unrestricted, lower, upper))
    weights = base + coordinate * V
    gradient = float(V @ COV @ weights)
    tolerance = 1e-11
    if upper == lower:
        certified = True  # A single feasible point, not a derivative test.
    elif abs(coordinate - lower) < tolerance:
        certified = gradient >= -tolerance
    elif abs(coordinate - upper) < tolerance:
        certified = gradient <= tolerance
    else:
        certified = abs(gradient) <= tolerance
    return LongOnly(target, weights, float(weights @ COV @ weights),
                    coordinate, unrestricted, lower, upper, gradient, certified)


def multipliers(weights):
    """Recover mean/budget and lower-bound multipliers at ordinary QP optima.

    At a singleton endpoint the equality multipliers are nonunique. Return
    None instead of presenting a minimum-norm choice as a valid dual witness.
    The scalar feasible-set certificate remains available there.
    """
    weights = np.asarray(weights, dtype=float)
    positive = np.flatnonzero(weights > 1e-9)
    if positive.size < 2:
        return None
    constraints = np.column_stack((MU, np.ones(3)))
    eta, _, _, _ = np.linalg.lstsq(constraints[positive],
                                 (COV @ weights)[positive], rcond=None)
    z = COV @ weights - constraints @ eta
    return eta, z


@dataclass
class BarrierPoint:
    tau: float
    weights: np.ndarray
    variance: float
    inequality_multipliers: np.ndarray
    mean_multiplier: float
    budget_multiplier: float
    primal_residual: float
    stationarity_residual: float
    complementarity: float
    dual_lower_bound: float
    numerical_gap: float
    central_gap_bound: float
    iterations: int


def barrier_path(target, taus=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)):
    """Strictly feasible Newton continuation for the fixed portfolio case.

    Inputs: target strictly inside (0.08, 0.12), positive nonincreasing finite
    tau values. Output: numerical primal/dual diagnostics at each tau. Raises
    on failed convergence/line search; endpoints have an empty barrier domain.
    These floating diagnostics are not interval-certified error bounds.
    """
    target = scalar(target, "target")
    base, lower, upper = feasible_interval(target)
    if not 0.08 < target < 0.12:
        raise ValueError("log barrier needs a strictly interior target")
    taus = tuple(scalar(t, "tau") for t in taus)
    if not taus or any(t <= 0 for t in taus):
        raise ValueError("tau sequence must be nonempty and positive")
    if any(b > a for a, b in zip(taus, taus[1:])):
        raise ValueError("tau sequence must be nonincreasing")
    weights = base + (lower + upper) / 2 * V
    rhs = np.array([(target - 0.10) / 0.02, 1.0])
    results = []
    for tau in taus:
        def objective(w):
            return float(w @ COV @ w / 2 - tau * np.log(w).sum())

        for iteration in range(100):
            gradient = COV @ weights - tau / weights
            projected = abs(float(V @ gradient)) / np.linalg.norm(V)
            if projected < 1e-9:
                break
            hessian = COV + np.diag(tau / weights ** 2)
            # The feasible-direction space is exactly span(V). This is the
            # null-space reduction of the equality Newton KKT system; avoid
            # losing a tiny feasible direction in a nearly converged block solve.
            derivative = float(V @ gradient)
            curvature = float(V @ hessian @ V)
            step = V * (-derivative / curvature)
            falling = step < 0
            alpha = 1.0
            if np.any(falling):
                alpha = min(alpha, 0.99 * float(np.min(
                    -weights[falling] / step[falling])))
            before = objective(weights)
            slope = -derivative ** 2 / curvature
            if slope >= 0:
                raise ArithmeticError("Newton direction is not descending")
            for _ in range(60):
                trial = weights + alpha * step
                if (np.min(trial) > 0 and
                        objective(trial) <= before + 0.01 * alpha * slope):
                    weights = trial
                    break
                alpha *= 0.5
            else:
                raise ArithmeticError("barrier line search failed")
        else:
            raise ArithmeticError("barrier Newton iteration limit")
        z = tau / weights
        gradient = COV @ weights - z
        gamma = np.linalg.lstsq(E, gradient, rcond=None)[0]
        eta = np.array([gamma[0] / 0.02,
                        gamma[1] - gamma[0] * 0.10 / 0.02])
        original_e = np.column_stack((MU, np.ones(3)))
        stationarity = COV @ weights - original_e @ eta - z
        q = original_e @ eta + z
        dual = float(eta @ [target, 1.0] - q @ np.linalg.solve(COV, q) / 2)
        value = float(weights @ COV @ weights / 2)
        results.append(BarrierPoint(
            tau, weights.copy(), 2 * value, z, float(eta[0]), float(eta[1]),
            float(np.max(np.abs(E.T @ weights - rhs))),
            float(np.max(np.abs(stationarity))),
            float(np.max(np.abs(weights * z))), dual, value - dual,
            3 * tau, iteration))
    return results


def bliss(prices, income):
    """Maximize -(x-3)^2-(y-2)^2 on a halfspace in R^2 (no goods bounds)."""
    prices = np.asarray(prices, dtype=float)
    income = scalar(income, "income")
    if prices.shape != (2,) or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("two finite strictly positive prices required")
    center = np.array([3.0, 2.0])
    shortage = max(float(prices @ center) - income, 0.0)
    multiplier = 2 * shortage / float(prices @ prices)
    choice = center - multiplier * prices / 2
    value = -float(np.sum((choice - center) ** 2))
    return choice, multiplier, value, income - float(prices @ choice)


def run_checks():
    close = np.testing.assert_allclose
    p = long_only(0.115)
    close(p.weights, [0.25, 0.0, 0.75], atol=1e-14)
    close(p.variance, 2.125, atol=1e-14)
    eta, z = multipliers(p.weights)
    close(eta, [50, -3.625], atol=1e-12)
    close(z, [0, 0.1, 0], atol=1e-12)
    close(p.unrestricted_coordinate, -1 / 156, atol=1e-14)
    close(long_only(0.10).weights, [1 / 6, 5 / 12, 5 / 12], atol=1e-14)
    close(long_only(0.08).weights, [0, 1, 0], atol=1e-14)
    close(long_only(0.12).weights, [0, 0, 1], atol=1e-14)
    free_sara, sara, sara_variance, sara_eta, sara_z = sara_case()
    assert free_sara.weights[0] < 0
    close(SARA_MU @ sara, 0.13, atol=1e-14)
    close(np.sum(sara), 1.0, atol=1e-14)
    close(sara_variance, 79 / 1200, atol=1e-14)
    close(sara_eta, [23 / 24, -47 / 800], atol=1e-14)
    close(sara_z, [97 / 6000, 0, 0], atol=1e-14)
    close(sara_z * sara, 0, atol=1e-14)
    assert sara_variance > free_sara.variance
    base, lo, hi = feasible_interval(0.115)
    initial = base + (lo + hi) / 2 * V
    gradient = COV @ initial - 0.01 / initial
    hessian = COV + np.diag(0.01 / initial ** 2)
    block = np.block([[hessian, -E], [E.T, np.zeros((2, 2))]])
    full_step = np.linalg.solve(block, np.r_[-gradient, [0, 0]])[:3]
    reduced_step = -V * float(V @ gradient) / float(V @ hessian @ V)
    close(full_step, reduced_step, atol=1e-14)
    for target in np.linspace(0.08, 0.12, 101):
        result = long_only(target)
        assert result.scalar_certificate
        assert np.min(result.weights) >= -1e-12
        close(MU @ result.weights, target, atol=1e-13)
        close(np.sum(result.weights), 1, atol=1e-13)
        base, lo, hi = feasible_interval(target)
        for t in np.linspace(lo, hi, 21):
            candidate = base + t * V
            assert candidate @ COV @ candidate >= result.variance - 1e-12
    for target in [0.10, 0.115]:
        optimum = long_only(target)
        for point in barrier_path(target):
            assert point.primal_residual < 1e-12
            assert point.stationarity_residual < 1e-8
            assert np.min(point.weights) > 0
            assert point.variance >= optimum.variance - 1e-12
            assert (point.variance - optimum.variance) / 2 <= 3 * point.tau + 1e-10
            assert point.dual_lower_bound <= optimum.variance / 2 + 1e-10
            close(point.numerical_gap, 3 * point.tau, atol=1e-11, rtol=1e-6)
    for prices, income, answer, lam in [([2, 2], 40, [3, 2], 0),
                                       ([2, 2], 6, [2, 1], 1),
                                       ([2, 2], 10, [3, 2], 0),
                                       ([5, 10], 25, [2.6, 1.2], 0.16),
                                       ([5, 10], 50, [3, 2], 0)]:
        choice, multiplier, value, slack = bliss(prices, income)
        close(choice, answer, atol=1e-14)
        close(multiplier, lam, atol=1e-14)
        assert slack >= -1e-12
        close(multiplier * slack, 0, atol=1e-12)
    for target in [0.079, 0.121, math.nan, True]:
        try:
            long_only(target)
        except ValueError:
            pass
        else:
            raise AssertionError("infeasible target accepted")
    for target, tau in [(0.08, [0.01]), (0.12, [0.01]),
                        (0.1, [0]), (0.1, [0.01, 0.1])]:
        try:
            barrier_path(target, tau)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid barrier domain accepted")
    print("Checks passed: scalar global certificates, 101 feasible targets,")
    print("KKT witness, barrier continuation/dual bounds, degeneracy and rejections.")


def main():
    np.set_printoptions(precision=12, suppress=True)
    for target in [0.08, 0.10, 0.115, 0.12]:
        p = long_only(target)
        print("LONG ONLY", p)
        print("multipliers", multipliers(p.weights))
    for p in barrier_path(0.115):
        print("BARRIER", p)
    print("SARA", sara_case())
    for prices, income in [([2, 2], 40), ([2, 2], 6), ([2, 2], 10),
                            ([5, 10], 25), ([5, 10], 50)]:
        print("BLISS", prices, income, bliss(prices, income))
    print("LOG GOODS", 180 / (3 * np.array([5, 6, 10])), "multiplier", 3 / (2 * 180))
    print("EMISSIONS", [2, 8], "abatement", [8, 2],
          "cost", 8 ** 2 + 4 * 2 ** 2, "cap multiplier", 16)
    run_checks()


if __name__ == "__main__":
    main()
