"""Chapter 4: exact financial constraints and NumPy linear-solver checks."""
from fractions import Fraction as F
import math
import numpy as np


def dot(u, v):
    if len(u) != len(v):
        raise ValueError('dot-product dimensions must agree')
    return math.fsum(a * b for a, b in zip(u, v, strict=True))


def target_family(t):
    """Exact family for means (.05,.10,.15) and target .11."""
    t = F(t)
    return [t - F(1, 5), F(6, 5) - 2*t, t]


def solve_square(a, b):
    """Moderate finite real inputs, one square system and one RHS vector."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError('a must be a nonempty square matrix')
    if b.shape != (a.shape[0],):
        raise ValueError('b must be a matching one-dimensional vector')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('data must be finite')
    return np.linalg.solve(a, b)


def diagnostics(a, b, w):
    """Return computed absolute residual and 2-norm condition estimate."""
    return np.linalg.norm(a @ w - b), np.linalg.cond(a)


def checks():
    means = [F(1, 20), F(1, 10), F(3, 20)]
    for t in (F(0), F(1, 5), F(1, 3), F(3, 5), F(1)):
        w = target_family(t)
        assert sum(w) == 1
        assert sum(a*b for a, b in zip(w, means)) == F(11, 100)
        assert all(x >= 0 for x in w) == (F(1, 5) <= t <= F(3, 5))
    a = np.array([[1., 1.], [.05, .10]])
    b = np.array([1., .08])
    np.testing.assert_allclose(solve_square(a, b), [.4, .6], atol=1e-14)
    lower = np.array([[1., 0.], [.05, 1.]])
    upper = np.array([[1., 1.], [0., .05]])
    np.testing.assert_allclose(lower @ upper, a, atol=1e-15)
    wide = np.array([[1., 1., 1.], [.05, .10, .15]])
    rhs = np.array([1., .11])
    least, _, rank, _ = np.linalg.lstsq(wide, rhs, rcond=None)
    np.testing.assert_allclose(least, [7/30, 1/3, 13/30], atol=1e-14)
    np.testing.assert_allclose(wide @ least, rhs, atol=1e-14)
    assert rank == 2
    risk_t = F(13, 45)
    risk_w = target_family(risk_t)
    variance = lambda w: (w[0]**2 + w[1]**2 + 4*w[2]**2) / 100
    norm_w = target_family(F(13, 30))
    assert risk_w == [F(4, 45), F(28, 45), F(13, 45)]
    assert variance(risk_w) < variance(norm_w)
    try:
        np.linalg.solve(wide, rhs)
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError('square solver unexpectedly accepted 2x3')
    try:
        solve_square([[1., 1.], [2., 2.]], [1., 2.])
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError('exactly singular example was accepted')
    for a_bad, b_bad in (([[1., 1., 1.]], [1.]),
                         ([[1., 0.], [0., 1.]], [[1.], [2.]]),
                         ([[1., 0.], [0., float('nan')]], [1., 2.])):
        try:
            solve_square(a_bad, b_bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid shape/data accepted')
    # Tall but consistent full-column-rank matrix has a unique exact solution.
    tall = np.array([[1., 0.], [0., 1.], [1., 1.]])
    np.testing.assert_allclose(np.linalg.lstsq(tall, [2., 3., 5.],
                                             rcond=None)[0], [2., 3.])
    print('checks: exact family, feasibility, LU, square/tall/wide systems,')
    print('        least-norm selection, singular and invalid inputs: PASS')


def main():
    checks()
    print('NumPy version:', np.__version__)
    means = [.05, .10, .15]
    for w in ([1/3]*3, [.25, .25, .5], [.5, .25, .25]):
        print(f'mean for {w}: {dot(w,means):.9f}')
    for t in (F(1, 5), F(13, 30), F(3, 5)):
        w = target_family(t)
        print('target .11 family t=', t, 'weights=',
              ','.join(str(x) for x in w),
              f'squared norm={float(sum(x*x for x in w)):.9f}')
    for label, t in (('leastnorm', F(13, 30)), ('minimumvariance', F(13, 45))):
        w = target_family(t)
        risk = (w[0]**2 + w[1]**2 + 4*w[2]**2) / 100
        print(f'{label}: t={t}, weights={w}, variance={float(risk):.12f}')
    wide = np.array([[1., 1., 1.], [.05, .10, .15]])
    rhs = np.array([1., .11])
    least, residuals, rank, singular = np.linalg.lstsq(wide, rhs, rcond=None)
    print('wide least-squares:', least, 'rank=', rank,
          'residual-array size=', residuals.size)
    print('wide explicitly computed residual=', np.linalg.norm(wide @ least-rhs))
    print('wide singular values=', singular)
    a = np.array([[1., 1.], [.05, .10]])
    b = np.array([1., .08])
    w = solve_square(a, b)
    print('two-asset target .08:', w, 'residual/kappa:', diagnostics(a,b,w))
    near = np.array([[1., 1.], [.1, .1001]])
    for target in (.10005, .10006):
        rhs = np.array([1., target])
        value = solve_square(near, rhs)
        residual, kappa = diagnostics(near, rhs, value)
        print(f'near means, target={target:.8f}: w={value}, '
              f'residual={residual:.9e}, kappa2={kappa:.9f}')
    scaled = np.diag([1., 100.]) @ near
    print(f'same near system, return row in percentage units: '
          f'kappa2={np.linalg.cond(scaled):.9f}')
    nonnormal = np.array([[1., 100.], [0., 1.]])
    print('triangular example eigenvalues:', np.linalg.eigvals(nonnormal))
    print(f'triangular example kappa2={np.linalg.cond(nonnormal):.9f}')
    inverse = np.array([[1., -100.], [0., 1.]])
    np.testing.assert_allclose(nonnormal @ inverse, np.eye(2))
    e2 = np.array([0., 1.])
    lower_bound = np.linalg.norm(nonnormal @ e2) * np.linalg.norm(inverse @ e2)
    print(f'triangular condition lower bound from one vector={lower_bound:.9f}')
    # Exact sensitivity comparison, independent of the numerical solve.
    gap = F(1001, 10000) - F(1, 10)
    for target in (F(2001, 20000), F(5003, 50000)):
        risky_second = (target - F(1, 10)) / gap
        print('exact near-system weights:', 1-risky_second, risky_second)
    print(f'weight sensitivity per unit target={float(1/gap):.1f}')
    print(f'target uncertainty allowed for weight error .01='
          f'{float(gap*F(1,100)):.9f}')


if __name__ == '__main__':
    main()
