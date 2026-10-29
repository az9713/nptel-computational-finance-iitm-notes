"""Chapter 5: covariance identities, spectral geometry, and model boundaries."""
from fractions import Fraction as F
import math
import numpy as np


def sector_covariance():
    """Original two-sector teaching matrix, not an estimated market matrix."""
    return np.array([[.04, .03, .004], [.03, .04, .004], [.004, .004, .01]])


def sorted_eigh(matrix):
    """Finite real symmetric teaching inputs; return descending unit modes.

    Symmetry is checked to an absolute tolerance of 1e-14. This does not certify
    positive semidefiniteness; negative eigenvalues remain visible.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0 or not np.isfinite(matrix).all()):
        raise ValueError('need a nonempty finite real square matrix')
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-14):
        raise ValueError('matrix must be symmetric to the stated tolerance')
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    for j in range(len(values)):
        pivot = np.argmax(np.abs(vectors[:, j]))
        if vectors[pivot, j] < 0:
            vectors[:, j] *= -1
    return values, vectors


def sample_covariance(rows):
    """Complete observations in rows; sample covariance with denominator M-1."""
    rows = np.asarray(rows, dtype=float)
    if (rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] == 0
            or not np.isfinite(rows).all()):
        raise ValueError('need at least two complete finite observations')
    centered = rows - rows.mean(axis=0)
    return centered.T @ centered / (len(rows)-1)


def pairwise_missing_covariance(rows):
    """Explicit diagnostic example, not a recommended covariance estimator."""
    n = rows.shape[1]
    result = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            selected = rows[np.isfinite(rows[:, [i, j]]).all(axis=1)]
            x, y = selected[:, i], selected[:, j]
            result[i, j] = np.dot(x-x.mean(), y-y.mean()) / (len(x)-1)
    return result


def checks():
    two = np.array([[.04, .01], [.01, .02]])
    w = np.array([.25, .75])
    exact = F(1, 4)**2 * F(4, 100) + F(3, 4)**2 * F(2, 100)
    exact += 2 * F(1, 4) * F(3, 4) * F(1, 100)
    assert exact == F(7, 400)
    assert math.isclose(w @ two @ w, float(exact), abs_tol=1e-15)
    covariance = sector_covariance()
    values, vectors = sorted_eigh(covariance)
    expected = [.04 + .002*math.sqrt(233), .01,
                .04 - .002*math.sqrt(233)]
    np.testing.assert_allclose(values, expected, atol=1e-14)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(covariance @ vectors, vectors * values,
                               atol=1e-14)
    np.testing.assert_allclose(vectors @ np.diag(values) @ vectors.T,
                               covariance, atol=1e-14)
    assert values[-1] > 0
    assert math.isclose(values.sum(), np.trace(covariance), abs_tol=1e-14)
    assert math.isclose(np.linalg.cond(covariance), values[0]/values[-1])
    for weights in ([1., 0., 0.], [1/3]*3, [1., -1., 0.]):
        z = vectors.T @ weights
        direct = np.asarray(weights) @ covariance @ weights
        assert math.isclose(direct, np.dot(values, z*z), abs_tol=1e-14)
    data = np.array([[.01, .02, -.01], [.03, -.01, .04],
                     [-.02, .01, 0.], [.02, .03, -.02]])
    sample = sample_covariance(data)
    np.testing.assert_allclose(sample, np.cov(data, rowvar=False, ddof=1),
                               atol=1e-15)
    assert np.linalg.eigvalsh(sample).min() >= -1e-14
    short_sample = sample_covariance(data[:2])
    assert np.linalg.matrix_rank(short_sample) <= 1
    bad = np.array([[1., .9, .9], [.9, 1., -.9], [.9, -.9, 1.]])
    assert sorted_eigh(bad)[0][-1] < -.7
    for invalid in ([[1., 2.], [0., 1.]], [[float('nan')]], [[1., 2., 3.]]):
        try:
            sorted_eigh(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid eigensystem input accepted')
    print('checks: exact quadratic, spectral residuals, orthogonality, trace,')
    print('        covariance Gram, rank, invalid symmetry/data: PASS')


def main():
    checks()
    np.set_printoptions(precision=9, suppress=True)
    two = np.array([[.04, .01], [.01, .02]])
    correlation = .01 / math.sqrt(.04*.02)
    print(f'two-asset correlation={correlation:.9f}')
    for first in (.5, .25):
        w = np.array([first, 1-first])
        variance = w @ two @ w
        print(f'two-asset w={w}, variance={variance:.12f}, '
              f'volatility={math.sqrt(variance):.12f}')
    print('two-asset eigenvalues:', sorted_eigh(two)[0])
    covariance = sector_covariance()
    values, vectors = sorted_eigh(covariance)
    print('sector covariance:\n', covariance)
    print('sector eigenvalues:', values)
    print('sector eigenvalue shares:', values/values.sum())
    print('sector unit eigenvectors, in columns:\n', vectors)
    print(f'sector trace={np.trace(covariance):.9f}, '
          f'kappa2={np.linalg.cond(covariance):.9f}')
    weights = np.array([1/3]*3)
    z = vectors.T @ weights
    terms = values * z*z
    print('equal-weight spectral risk terms:', terms)
    print('equal-weight portfolio risk shares:', terms/terms.sum())
    print(f'equal-weight variance={terms.sum():.12f}, '
          f'volatility={math.sqrt(terms.sum()):.12f}')
    contrast = np.array([1., -1., 0.]) / math.sqrt(2)
    print(f'unit within-sector contrast weight sum={contrast.sum():.9f}, '
          f'variance={contrast@covariance@contrast:.9f}')
    for rho in (.99, 1.):
        correlated = .04 * np.array([[1., rho], [rho, 1.]])
        ev, _ = sorted_eigh(correlated)
        print(f'correlation={rho}: eigenvalues={ev}, '
              f'computed condition={np.linalg.cond(correlated):.9f}')
    alpha = .001
    shifted = covariance + alpha*np.eye(3)
    print(f'ridge alpha={alpha}: shifted spectrum={values+alpha}, '
          f'condition={np.linalg.cond(shifted):.9f}')
    retained = values[0] * z[0]**2
    print(f'one-mode equal-weight risk={retained:.12f}, '
          f'omitted={terms[1:].sum():.12f}, '
          f'upper bound={values[1]*np.dot(weights,weights):.12f}')
    x, y = np.array([-.1, .1, 0.]), np.array([-.1, -.2, .3])
    print(f'negative covariance={np.mean((x-x.mean())*(y-y.mean())):.12f}, '
          f'joint loss probability={np.mean((x<0)&(y<0)):.9f}')
    x = np.array([-1., 0., 1.])
    y = x*x
    print(f'uncorrelated dependent: meanX={x.mean():.9f}, '
          f'meanY={y.mean():.9f}, '
          f'covariance={np.mean((x-x.mean())*(y-y.mean())):.9f}')
    bad = np.array([[1., .9, .9], [.9, 1., -.9], [.9, -.9, 1.]])
    direction = np.array([-1., 1., 1.])
    print('invalid pairwise correlations eigenvalues:', sorted_eigh(bad)[0])
    print(f'invalid direction quadratic={direction@bad@direction:.9f}')
    missing = np.array([[1., 1., np.nan], [-1., -1., np.nan],
                        [1., np.nan, 1.], [-1., np.nan, -1.],
                        [np.nan, 1., -1.], [np.nan, -1., 1.]])
    pairwise = pairwise_missing_covariance(missing)
    print('separate-pair missing-data covariance:\n', pairwise)
    print('separate-pair eigenvalues:', sorted_eigh(pairwise)[0])
    print('percent-unit covariance factor:', 100**2)
    print('variance .04 in squared percentage-point units:', .04 * 100**2)


if __name__ == '__main__':
    main()
