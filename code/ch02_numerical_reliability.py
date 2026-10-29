"""Chapter 2: original synthetic examples, standard-library execution checks."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import sys


def exp_small(x, abs_tol=1e-14, max_terms=100):
    """For Python float x in [0, 0.5], return sum, tail estimate, degree.

    The tail estimate bounds exact-arithmetic truncation. It is evaluated in
    binary64 and does not certify the total floating-point evaluation error.
    """
    if not math.isfinite(x) or not 0 <= x <= 0.5:
        raise ValueError("x must be finite and in [0, 0.5]")
    if not math.isfinite(abs_tol) or abs_tol <= 0:
        raise ValueError("abs_tol must be finite and positive")
    if not isinstance(max_terms, int) or isinstance(max_terms, bool):
        raise ValueError("max_terms must be a positive integer")
    if max_terms <= 0:
        raise ValueError("max_terms must be a positive integer")
    terms = [1.0]
    term = 1.0
    for degree in range(max_terms):
        first_omitted = term * x / (degree + 1)
        ratio_bound = x / (degree + 2)
        tail = first_omitted / (1 - ratio_bound)
        if tail <= abs_tol:
            return math.fsum(terms), tail, degree
        term = first_omitted
        terms.append(term)
    raise RuntimeError("truncation target not reached within max_terms")


def annual_discount(rate, maturity):
    if not math.isfinite(rate) or rate <= -1:
        raise ValueError("annual rate must be finite and greater than -1")
    if not math.isfinite(maturity) or maturity < 0:
        raise ValueError("maturity must be finite and nonnegative")
    return math.exp(-maturity * math.log1p(rate))


def linear_value(t, a, fa, b, fb):
    if not all(math.isfinite(v) for v in (t, a, fa, b, fb)):
        raise ValueError("interpolation values must be finite")
    if not a < b or not a <= t <= b:
        raise ValueError("need ordered nodes and an interior query")
    weight = (t - a) / (b - a)
    return (1 - weight) * fa + weight * fb


def flat_price(rate):
    return math.fsum(c * math.exp(-rate * t)
                     for c, t in zip((4, 4, 104), (1, 2, 3)))


def checks():
    for x in (0.0, 0.001, 0.08, 0.5):
        value, tail, degree = exp_small(x)
        with localcontext() as ctx:
            ctx.prec = 70
            exact = Decimal.from_float(x).exp()
            error = abs(Decimal.from_float(value) - exact)
            assert error <= Decimal.from_float(tail) + Decimal("1e-15")
        assert 0 <= degree < 100 and tail <= 1e-14
    invalid = ((-0.1, 1e-14, 100), (0.6, 1e-14, 100),
               (float("nan"), 1e-14, 100), (0.1, 0, 100),
               (0.1, float("inf"), 100), (0.1, 1e-14, 0),
               (0.1, 1e-14, 1.5))
    for args in invalid:
        try:
            exp_small(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    try:
        exp_small(0.5, 1e-30, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("insufficient iteration budget accepted")
    assert linear_value(1, 1, 0.04, 3, 0.06) == 0.04
    assert linear_value(3, 1, 0.04, 3, 0.06) == 0.06
    assert math.fsum([1e16, 1, -1e16]) == 1
    assert (Fraction(101, 100) - 1) / Fraction(1, 100) == 1
    assert (Fraction(102, 100) - 1) / Fraction(1, 100) == 2
    for freq in (1, 2, 12, 365):
        nominal = freq * math.expm1(math.log1p(0.08) / freq)
        equivalent = (1 + nominal / freq) ** freq
        assert math.isclose(equivalent, 1.08, rel_tol=5e-14)
    print("checks: recurrence, Decimal reference, invalid inputs, exact system,")
    print("        interpolation endpoints, equivalent conventions: PASS")


def main():
    checks()
    d1, d3 = annual_discount(0.04, 1), annual_discount(0.06, 3)
    z2 = linear_value(2, 1, 0.04, 3, 0.06)
    d2_rate = annual_discount(z2, 2)
    d2_log = math.exp(linear_value(2, 1, math.log(d1), 3, math.log(d3)))
    p_rate = math.fsum((4 * d1, 4 * d2_rate, 104 * d3))
    p_log = math.fsum((4 * d1, 4 * d2_log, 104 * d3))
    print(f"discounts D1={d1:.9f}, D2(rate)={d2_rate:.9f}, D3={d3:.9f}")
    print(f"price: rate-linear={p_rate:.8f}, log-D-linear={p_log:.8f},")
    print(f"       difference={p_log - p_rate:.8f}")
    print(f"D2(log-D)={d2_log:.9f}; forward1-3="
          f"{-(math.log(d3) - math.log(d1)) / 2:.9f}")
    for freq in (1, 2, 12, 365):
        amount = 100 * math.exp(freq * math.log1p(0.08 / freq))
        nominal = freq * math.expm1(math.log1p(0.08) / freq)
        print(f"8% fixed nominal, m={freq:3}: {amount:.8f}; "
              f"nominal equivalent to 8% effective={nominal:.10f}")
    print(f"8% continuous: {100 * math.exp(0.08):.8f}; "
          f"continuous equivalent to 8% effective={math.log1p(0.08):.10f}")
    for x in (0.08, 0.5):
        value, tail, degree = exp_small(x)
        print(f"exp_small({x}): {value:.16f}, degree={degree}, "
              f"tail={tail:.3e}, libm_difference={abs(value-math.exp(x)):.3e}")
    for x, tolerance in ((0.001, 1e-14), (0.08, 1e-14),
                         (0.08, 1e-16)):
        value, tail, degree = exp_small(x, tolerance)
        with localcontext() as context:
            context.prec = 80
            reference = Decimal.from_float(x).exp()
            error = abs(Decimal.from_float(value) - reference)
        print(f"reference check x={x}, tolerance={tolerance:.0e}: "
              f"degree={degree}, tail={tail:.3e}, error={error:.3e}")
    for t in (1, 10):
        x = 0.08 * t
        exact = math.exp(-x)
        print(f"discount linear, T={t}: exact={exact:.9f}, "
              f"linear={1-x:.9f}, relative_error={(1-x)/exact-1:.6%}")
    r, shock = 0.05, 0.001
    p0 = flat_price(r)
    first = -math.fsum(t * c * math.exp(-r * t)
                       for c, t in zip((4, 4, 104), (1, 2, 3)))
    second = math.fsum(t*t * c * math.exp(-r * t)
                       for c, t in zip((4, 4, 104), (1, 2, 3)))
    approx = p0 + first * shock + second * shock**2 / 2
    print(f"flat-rate price={p0:.9f}, derivative={first:.9f}, "
          f"second={second:.9f}")
    print(f"+10bp exact price={flat_price(r+shock):.9f}, "
          f"quadratic={approx:.9f}")
    print(f"0.1+0.2 == 0.3: {0.1 + 0.2 == 0.3}; "
          f"naive sum={sum([1e16, 1, -1e16]):.1f}, "
          f"fsum={math.fsum([1e16, 1, -1e16]):.1f}")
    print(f"epsilon={sys.float_info.epsilon:.17e}, "
          f"unit_roundoff={sys.float_info.epsilon / 2:.17e}")
    print(f"tiny x=1e-16: log(1+x)={math.log(1+1e-16):.17e}, "
          f"log1p(x)={math.log1p(1e-16):.17e}")
    print(f"tiny x=1e-16: exp(x)-1={math.exp(1e-16)-1:.17e}, "
          f"expm1(x)={math.expm1(1e-16):.17e}")
    for rhs in (Fraction(101, 100), Fraction(102, 100)):
        y = (rhs - 1) / Fraction(1, 100)
        print(f"exact system RHS2={float(rhs):.2f}: x={1+y}, y={y}")
    # Exercise outputs use the same conventions as their problem statements.
    print(f"exercise: interpolation M=.004, gap=2: rate bound={.004*4/8:.6f}")
    low, high = 104 / 1.052**3, 104 / 1.048**3
    print(f"exercise: 104 at T=3, annual z=.05+/- .002: "
          f"price interval=[{low:.8f}, {high:.8f}]")
    print(f"exercise: 1bp on 1,000,000 at r=.05,T=10: "
          f"linear={-10*1e6*math.exp(-.5)*.0001:.8f}, "
          f"exact={1e6*(math.exp(-.501)-math.exp(-.5)):.8f}")


if __name__ == "__main__":
    main()
