"""Finite-state arbitrage: small bounded LPs and independently checked cashflows.

NumPy vertex enumeration is deliberately restricted to small educational
problems. It is not a production LP solver or a market/trading connection.
Numerical tolerances do not certify exact no-arbitrage for arbitrary inputs.
"""
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch19_mean_variance import scalar  # noqa: E402


@dataclass
class LPResult:
    holdings: np.ndarray
    value: float
    bases_tested: int
    feasible_bases: int
    scaled_violation: float


def bounded_lp(objective, lhs, rhs, bound=1.0):
    """Minimize c@x with H@x<=h and |x_i|<=bound by small-basis enumeration.

    Finite arrays, 1..5 variables, <=100000 possible active bases. Rows are
    scaled before solving; feasibility uses absolute scaled tolerance 1e-10.
    Rank-deficient numerical bases are skipped. Raises on no detected feasible
    vertex or excessive work. Intended for the well-resolved chapter fixtures.
    """
    objective = np.asarray(objective, dtype=float)
    lhs, rhs = np.asarray(lhs, dtype=float), np.asarray(rhs, dtype=float)
    bound = scalar(bound, "bound")
    if (objective.ndim != 1 or not 1 <= objective.size <= 5 or
            lhs.ndim != 2 or lhs.shape[1] != objective.size or
            rhs.shape != (lhs.shape[0],) or bound <= 0):
        raise ValueError("invalid finite LP dimensions or bound")
    if not all(np.all(np.isfinite(a)) for a in (objective, lhs, rhs)):
        raise ValueError("finite LP coefficients required")
    n = objective.size
    matrix = np.vstack((lhs, np.eye(n), -np.eye(n)))
    right = np.r_[rhs, np.full(2 * n, bound)]
    scales = np.maximum(1.0, np.max(np.abs(matrix), axis=1))
    matrix, right = matrix / scales[:, None], right / scales
    work = math.comb(matrix.shape[0], n)
    if work > 100000:
        raise ValueError("basis enumeration work limit exceeded")
    best, best_value, feasible = None, math.inf, 0
    for active in combinations(range(matrix.shape[0]), n):
        block = matrix[list(active)]
        if np.linalg.matrix_rank(block, tol=1e-12) < n:
            continue
        point = np.linalg.solve(block, right[list(active)])
        if np.max(matrix @ point - right) > 1e-10:
            continue
        feasible += 1
        value = float(objective @ point)
        if value < best_value:
            best, best_value = point, value
    if best is None:
        raise ArithmeticError("no feasible numerical vertex found")
    violation = max(0.0, float(np.max(matrix @ best - right)))
    return LPResult(best, best_value, work, feasible, violation)


def market_inputs(payoffs, prices):
    payoffs, prices = np.asarray(payoffs, dtype=float), np.asarray(prices, dtype=float)
    if (payoffs.ndim != 2 or min(payoffs.shape) < 1 or
            prices.shape != (payoffs.shape[1],)):
        raise ValueError("payoff rows are states; columns must match prices")
    if not np.all(np.isfinite(payoffs)) or not np.all(np.isfinite(prices)):
        raise ValueError("finite market inputs required")
    return payoffs, prices


def detect(payoffs, prices, tolerance=1e-8):
    """Two-phase bounded test; return raw cashflows and a numerical decision.

    Phase one minimizes initial cost with nonnegative future payoffs.
    Phase two maximizes total future payoff with nonpositive initial cost.
    Holdings are asset units, not wealth fractions. Equal unit bounds merely
    normalize homogeneous arbitrage directions. No prices/orders are fetched.
    """
    payoffs, prices = market_inputs(payoffs, prices)
    tolerance = scalar(tolerance, "tolerance")
    if tolerance <= 0:
        raise ValueError("positive currency-unit tolerance required")
    zeros = np.zeros(payoffs.shape[0])
    phase_one = bounded_lp(prices, -payoffs, zeros)
    phase_two = bounded_lp(-payoffs.sum(axis=0),
                           np.vstack((-payoffs, prices)),
                           np.r_[zeros, 0.0])
    candidate = None
    source = None
    if (phase_one.value < -tolerance and
            np.min(payoffs @ phase_one.holdings) >= -tolerance):
        candidate, source = phase_one.holdings, "negative initial cost"
    elif (-phase_two.value > tolerance and
          float(prices @ phase_two.holdings) <= tolerance and
          np.min(payoffs @ phase_two.holdings) >= -tolerance):
        candidate, source = phase_two.holdings, "positive future payoff"
    return {
        "phase_one": phase_one, "phase_two": phase_two,
        "decision": "numerical arbitrage candidate" if candidate is not None
        else "none detected at tolerance",
        "source": source, "holdings": candidate,
        "cost": None if candidate is None else float(prices @ candidate),
        "payoffs": None if candidate is None else payoffs @ candidate,
        "tolerance": tolerance,
    }


def complete_state_prices(payoffs, prices):
    """Unique linear-fit state prices when payoff row rank is full.

    Return the candidate and its raw pricing residual. Strict positivity plus
    exact consistency is the mathematical certificate. A floating fit is a
    diagnostic; inconsistent prices are not silently replaced by the fit.
    """
    payoffs, prices = market_inputs(payoffs, prices)
    if np.linalg.matrix_rank(payoffs) != payoffs.shape[0]:
        raise ValueError("this helper requires a complete payoff span")
    state_prices = np.linalg.lstsq(payoffs.T, prices, rcond=None)[0]
    return state_prices, float(np.max(np.abs(payoffs.T @ state_prices - prices)))


LECTURE_A = np.array([[1.05, 110.0, 10.0], [1.05, 90.0, 0.0]])
AMRIT_A = np.array([[1.0, 100.0, 20.0], [1.0, 60.0, 0.0]])


def run_checks():
    close = np.testing.assert_allclose
    p = np.array([1.0, 100.0, 6.0])
    result = detect(LECTURE_A, p)
    close(result["phase_one"].holdings, [1, -7 / 600, 7 / 300], atol=1e-12)
    close(result["cost"], -2 / 75, atol=1e-12)
    close(result["payoffs"], 0, atol=1e-12)
    scaled = np.array([300 / 7, -0.5, 1])
    close(LECTURE_A @ scaled, 0, atol=1e-12)
    close(p @ scaled, -8 / 7, atol=1e-12)
    fair = np.array([1.0, 100.0, 50 / 7])
    noarb = detect(LECTURE_A, fair)
    assert noarb["holdings"] is None
    state_prices, residual = complete_state_prices(LECTURE_A, fair)
    close(state_prices, [5 / 7, 5 / 21], atol=1e-12)
    assert residual < 1e-11
    assert np.min(state_prices) > 0
    close(state_prices / state_prices.sum(), [0.75, 0.25], atol=1e-12)
    physical = np.array([0.5, 0.5])
    sdf = state_prices / physical
    close(sdf, [10 / 7, 10 / 21], atol=1e-12)
    close(physical @ sdf, 20 / 21, atol=1e-12)
    overpriced = detect(LECTURE_A, [1, 100, 8])
    assert overpriced["holdings"] is not None
    reverse = -scaled
    close(LECTURE_A @ reverse, 0, atol=1e-12)
    close(np.array([1, 100, 8]) @ reverse, -6 / 7, atol=1e-12)
    amrit = detect(AMRIT_A, [0.92, 80, 11.5])
    close(amrit["cost"], -0.03, atol=1e-12)
    close(amrit["payoffs"], 0, atol=1e-12)
    amrit_scaled = np.array([30, -0.5, 1])
    close(AMRIT_A @ amrit_scaled, 0, atol=1e-12)
    close(np.array([0.92, 80, 11.5]) @ amrit_scaled, -0.9, atol=1e-12)
    amrit_prices, residual = complete_state_prices(AMRIT_A, [0.92, 80, 12.4])
    close(amrit_prices, [0.62, 0.30], atol=1e-12)
    close(amrit_prices / 0.92, [31 / 46, 15 / 46], atol=1e-12)
    assert residual < 1e-11
    hidden = detect(np.eye(2), [1, 0])
    close(hidden["phase_one"].value, 0, atol=1e-12)
    assert hidden["source"] == "positive future payoff"
    close(hidden["holdings"], [0, 1], atol=1e-12)
    incomplete_a = np.array([[1, 0], [1, 1], [1, 2]])
    for alpha in [0.1, 0.25, 0.4]:
        s = np.array([alpha, 1 - 2 * alpha, alpha])
        close(incomplete_a.T @ s, [1, 1], atol=1e-12)
        assert np.min(s) > 0
        extended = np.column_stack((incomplete_a, [0, 0, 1]))
        assert detect(extended, [1, 1, alpha])["holdings"] is None
    for edge in [0, 0.5]:
        assert detect(extended, [1, 1, edge])["holdings"] is not None
    for currency_scale in [0.01, 100.0]:
        scaled_result = detect(LECTURE_A * currency_scale, p * currency_scale)
        assert scaled_result["holdings"] is not None
        close(scaled_result["holdings"], result["holdings"], atol=1e-11)
    for payoffs, prices in [([[1, 2]], [1]), ([[math.nan]], [1])]:
        try:
            detect(payoffs, prices)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid market shape/values accepted")
    print("Checks passed: two source markets, under/fair/overpricing, zero-cost")
    print("counterexample, state prices, incomplete interval, currency scaling,")
    print("raw cashflow witnesses and input rejections.")


def main():
    np.set_printoptions(precision=12, suppress=True)
    for label, payoffs, prices in [
            ("lecture cheap", LECTURE_A, [1, 100, 6]),
            ("lecture fair", LECTURE_A, [1, 100, 50 / 7]),
            ("lecture dear", LECTURE_A, [1, 100, 8]),
            ("Amrit cheap", AMRIT_A, [0.92, 80, 11.5]),
            ("Amrit fair", AMRIT_A, [0.92, 80, 12.4]),
            ("zero-cost counterexample", np.eye(2), [1, 0])]:
        print(label, detect(payoffs, prices))
    for a, p in [(LECTURE_A, [1, 100, 50 / 7]), (AMRIT_A, [0.92, 80, 12.4])]:
        s, residual = complete_state_prices(a, p)
        print("STATE PRICES", s, "discount", s.sum(),
              "Q", s / s.sum(), "pricing residual", residual)
    print("ILLUSTRATIVE equal-physical-probability SDF",
          np.array([5 / 7, 5 / 21]) / np.array([0.5, 0.5]))
    run_checks()


if __name__ == "__main__":
    main()
