"""Chapter 1: inspectable cash-flow valuation and reproducibility checks."""
from fractions import Fraction
import math
import platform
import sys
import timeit


def present_value(cashflows, times, annual_rate):
    """Value finite real inputs with dates in years and an effective rate.

    The teaching contract accepts lists/tuples of ordinary real numbers.
    Return a float, do not mutate inputs, and perform no I/O. Empty flows
    have zero value. Extreme underflow is not range-certified here.
    """
    if len(cashflows) != len(times):
        raise ValueError("cash flows and dates must have equal lengths")
    if not math.isfinite(annual_rate) or annual_rate <= -1:
        raise ValueError("effective annual rate must be finite and > -1")
    if not all(math.isfinite(c) for c in cashflows):
        raise ValueError("cash flows must be finite")
    if not all(math.isfinite(t) and t >= 0 for t in times):
        raise ValueError("dates must be finite and nonnegative")
    exponent = math.log1p(annual_rate)
    terms = [c * math.exp(-t * exponent)
             for c, t in zip(cashflows, times, strict=True)]
    if not all(math.isfinite(term) for term in terms):
        raise ArithmeticError("present values exceed the finite range")
    return math.fsum(terms)


def checks():
    cash, times, rate = [100, 100, 1100], [1, 2, 3], .05
    original = cash.copy(), times.copy()
    value = present_value(cash, times, rate)
    exact = sum(Fraction(c) / Fraction(21, 20)**t
                for c, t in zip(cash, times, strict=True))
    assert math.isclose(value, float(exact), rel_tol=1e-14)
    assert (cash, times) == original
    assert present_value(cash, times, 0) == 1300
    assert present_value([], [], rate) == 0
    assert present_value([17], [0], rate) == 17
    assert math.isclose(present_value([200, 200, 2200], times, rate),
                        2 * value, rel_tol=1e-14)
    assert math.isclose(present_value(cash[::-1], times[::-1], rate),
                        value, rel_tol=1e-14)
    assert math.isclose(present_value([100, 100, 500, 600], [1, 2, 3, 3],
                                     rate), value, rel_tol=1e-14)
    discounts = [(1+rate)**(-t) for t in times]
    assert min(discounts) * sum(cash) <= value <= sum(cash)
    assert present_value(cash, times, .02) > value
    assert present_value(cash, times, .07) < value
    assert present_value([100], [1], -.01) > 100
    for args in (([100, 1100], [1], rate), ([100], [-1], rate),
                 ([float('nan')], [1], rate), ([100], [1], -1),
                 ([100], [float('inf')], rate)):
        try:
            present_value(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid domain was accepted")
    try:
        exec('amount = principal * 1.08', {})
    except NameError:
        pass
    else:
        raise AssertionError("missing notebook dependency went unnoticed")
    print('checks: exact rational reference, zero rate, scaling, permutation,')
    print('        split payments, bounds, monotonicity, domains, state: PASS')


def benchmark():
    """Optional dated timing observation, not a portable speed guarantee."""
    cash, dates, rate = [100, 100, 1100], [1, 2, 3], .05
    calls, repetitions = 10000, 5
    samples = timeit.repeat(lambda: present_value(cash, dates, rate),
                            number=calls, repeat=repetitions)
    print(f'benchmark: Python{platform.python_version()}, '
          f'{platform.system()}, calls={calls}, repeats={repetitions}')
    print('microseconds per validated price:',
          ', '.join(f'{1e6 * sample / calls:.6f}' for sample in samples))
    print('scope: includes validation and function-call overhead; '
          'not comparable to a single exponent-only call')


def main():
    checks()
    for cash, dates, label in (([100, 100, 1100], [1, 2, 3], 'A'),
                               ([25, 25, 25, 25, 1025],
                                [1, 2, 3, 4, 5], 'B')):
        for rate in (.01, .02, .05, .07, .08):
            print(f'lab{label}: annual_rate={rate:.2%}, '
                  f'price={present_value(cash,dates,rate):.9f}')
    total = 0.0
    for index, (cash, date) in enumerate(zip([100, 100, 1100], [1, 2, 3]), 1):
        discount = 1.05**(-date)
        term = cash * discount
        total += term
        print(f'trace{index}: D={discount:.9f}, PV={term:.9f}, '
              f'total={total:.9f}')
    good = present_value([100, 1100], [1, 2], .05)
    truncated = sum(c / 1.05**t for c, t in zip([100, 1100], [1]))
    print(f'shape bug: default zip={truncated:.9f}; complete={good:.9f}')
    print(f'negative1%: future100 in year1 costs='
          f'{present_value([100],[1],-.01):.9f}')
    print(f'two discounted units at years1and2='
          f'{present_value([1,1],[1,2],.05):.9f}')
    for frequency in (1, 2, 12, 365):
        print(f'compounding8%, frequency={frequency}: '
              f'{100*(1+.08/frequency)**frequency:.9f}')
    print(f'continuous8%: {100*math.exp(.08):.9f}')
    print(f'float equality .1+.2==.3: {.1+.2==.3}; '
          f'isclose: {math.isclose(.1+.2,.3,rel_tol=1e-15)}')
    print(f'library exp(2)={math.exp(2):.9f}')
    for degree in (10, 20):
        polynomial = math.fsum(2**k / math.factorial(k)
                              for k in range(degree + 1))
        print(f'exp(2) partial degree{degree}: {polynomial:.12f}, '
              f'library discrepancy={polynomial-math.exp(2):.9e}')
    if '--benchmark' in sys.argv:
        benchmark()


if __name__ == '__main__':
    main()
