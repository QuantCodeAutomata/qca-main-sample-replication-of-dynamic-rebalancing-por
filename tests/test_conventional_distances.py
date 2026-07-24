"""pytest tests for src/conventional_distances.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.conventional_distances import (
    correlation_distance,
    euclidean_distance,
    compute_distance_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def array_a(rng):
    """Random (T=100, d=3) feature array for one asset."""
    return rng.normal(0, 1, (100, 3))


@pytest.fixture
def array_b(rng):
    """Another random (T=100, d=3) feature array, independent of array_a."""
    return rng.normal(0.5, 1.5, (100, 3))


@pytest.fixture
def feature_dict_6(rng):
    """Feature dict for 6 tickers, each (T=80, d=3)."""
    tickers = [f'T{i}' for i in range(6)]
    return {t: rng.standard_normal((80, 3)) for t in tickers}, tickers


@pytest.fixture
def feature_dict_small(rng):
    """Feature dict for 5 tickers, each (T=60, d=3)."""
    tickers = [f'S{i}' for i in range(5)]
    return {t: rng.standard_normal((60, 3)) for t in tickers}, tickers


# ---------------------------------------------------------------------------
# correlation_distance
# ---------------------------------------------------------------------------
class TestCorrelationDistance:
    def test_correlation_distance_identical(self, array_a):
        """Distance from an array to itself must be ~0 (within floating-point tolerance)."""
        d = correlation_distance(array_a, array_a)
        assert np.isclose(d, 0.0, atol=1e-6), (
            f"correlation_distance(Z, Z) should be 0; got {d}."
        )

    def test_correlation_distance_range(self, array_a, array_b):
        """Correlation distance must lie in [0, 2]."""
        d = correlation_distance(array_a, array_b)
        assert 0.0 - 1e-8 <= d <= 2.0 + 1e-8, (
            f"correlation_distance must be in [0, 2]; got {d}."
        )

    def test_correlation_distance_perfectly_negative(self, array_a):
        """Perfectly anti-correlated arrays → distance ≈ 2."""
        d = correlation_distance(array_a, -array_a)
        assert np.isclose(d, 2.0, atol=1e-6), (
            f"Anti-correlated arrays should give distance=2; got {d}."
        )

    def test_correlation_distance_symmetric(self, array_a, array_b):
        assert np.isclose(
            correlation_distance(array_a, array_b),
            correlation_distance(array_b, array_a),
            atol=1e-8,
        ), "correlation_distance must be symmetric."

    def test_correlation_distance_nonneg(self, array_a, array_b):
        d = correlation_distance(array_a, array_b)
        assert d >= 0.0, "correlation_distance must be non-negative."

    def test_correlation_distance_returns_float(self, array_a, array_b):
        d = correlation_distance(array_a, array_b)
        assert isinstance(d, (float, np.floating))

    def test_correlation_distance_1d_input(self, rng):
        """1-D inputs should also be accepted (auto-reshaped to (T, 1))."""
        z = rng.normal(0, 1, 100)
        d = correlation_distance(z, z)
        assert np.isclose(d, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------
class TestEuclideanDistance:
    def test_euclidean_distance_identical(self, array_a):
        """Distance from an array to itself must be 0."""
        d = euclidean_distance(array_a, array_a)
        assert np.isclose(d, 0.0, atol=1e-8), (
            f"euclidean_distance(Z, Z) should be 0; got {d}."
        )

    def test_euclidean_distance_nonneg(self, array_a, array_b):
        d = euclidean_distance(array_a, array_b)
        assert d >= 0.0, "Euclidean distance must be non-negative."

    def test_euclidean_distance_symmetric(self, array_a, array_b):
        assert np.isclose(
            euclidean_distance(array_a, array_b),
            euclidean_distance(array_b, array_a),
            atol=1e-8,
        ), "Euclidean distance must be symmetric."

    def test_euclidean_distance_triangle_inequality(self, rng):
        """d(a, c) <= d(a, b) + d(b, c) (triangle inequality)."""
        a = rng.normal(0, 1, (50, 3))
        b = rng.normal(0, 1, (50, 3))
        c = rng.normal(0, 1, (50, 3))
        assert (
            euclidean_distance(a, c)
            <= euclidean_distance(a, b) + euclidean_distance(b, c) + 1e-9
        )

    def test_euclidean_distance_known_value(self):
        """Frobenius norm of difference: [[3, 4]] - [[0, 0]] → 5."""
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        assert np.isclose(euclidean_distance(a, b), 5.0)

    def test_euclidean_distance_returns_float(self, array_a, array_b):
        d = euclidean_distance(array_a, array_b)
        assert isinstance(d, (float, np.floating))


# ---------------------------------------------------------------------------
# compute_distance_matrix
# ---------------------------------------------------------------------------
class TestComputeDistanceMatrix:
    def test_distance_matrix_symmetry(self, feature_dict_6):
        fd, tickers = feature_dict_6
        D = compute_distance_matrix(fd, tickers, method='correlation')
        assert np.allclose(D, D.T, atol=1e-8), "Distance matrix must be symmetric."

    def test_distance_matrix_zero_diagonal(self, feature_dict_6):
        fd, tickers = feature_dict_6
        D = compute_distance_matrix(fd, tickers, method='correlation')
        assert np.allclose(np.diag(D), 0.0, atol=1e-6), "Diagonal must be 0."

    def test_distance_matrix_method_correlation(self, feature_dict_6):
        """method='correlation' must run without error and return correct shape."""
        fd, tickers = feature_dict_6
        n = len(tickers)
        D = compute_distance_matrix(fd, tickers, method='correlation')
        assert D.shape == (n, n), f"Expected ({n}, {n}), got {D.shape}."

    def test_distance_matrix_method_euclidean(self, feature_dict_6):
        """method='euclidean' must run without error and return correct shape."""
        fd, tickers = feature_dict_6
        n = len(tickers)
        D = compute_distance_matrix(fd, tickers, method='euclidean')
        assert D.shape == (n, n), f"Expected ({n}, {n}), got {D.shape}."

    def test_distance_matrix_nonneg_correlation(self, feature_dict_6):
        fd, tickers = feature_dict_6
        D = compute_distance_matrix(fd, tickers, method='correlation')
        assert (D >= -1e-8).all(), "All correlation distance entries must be >= 0."

    def test_distance_matrix_nonneg_euclidean(self, feature_dict_6):
        fd, tickers = feature_dict_6
        D = compute_distance_matrix(fd, tickers, method='euclidean')
        assert (D >= -1e-8).all(), "All euclidean distance entries must be >= 0."

    def test_distance_matrix_symmetry_euclidean(self, feature_dict_6):
        fd, tickers = feature_dict_6
        D = compute_distance_matrix(fd, tickers, method='euclidean')
        assert np.allclose(D, D.T, atol=1e-8), "Euclidean distance matrix must be symmetric."

    def test_distance_matrix_zero_diagonal_euclidean(self, feature_dict_6):
        fd, tickers = feature_dict_6
        D = compute_distance_matrix(fd, tickers, method='euclidean')
        assert np.allclose(np.diag(D), 0.0, atol=1e-8), "Diagonal must be 0."

    def test_distance_matrix_identical_assets(self, rng):
        """Two identical asset arrays must yield distance 0 between them."""
        base = rng.standard_normal((80, 3))
        tickers = ['A', 'B', 'C']
        fd = {'A': base.copy(), 'B': base.copy(), 'C': rng.standard_normal((80, 3))}
        D = compute_distance_matrix(fd, tickers, method='correlation')
        assert np.isclose(D[0, 1], 0.0, atol=1e-6), (
            "Distance between identical asset arrays must be 0."
        )

    def test_distance_matrix_returns_ndarray(self, feature_dict_small):
        fd, tickers = feature_dict_small
        D = compute_distance_matrix(fd, tickers, method='correlation')
        assert isinstance(D, np.ndarray)
