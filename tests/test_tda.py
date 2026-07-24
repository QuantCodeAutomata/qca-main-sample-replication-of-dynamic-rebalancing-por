"""pytest tests for src/tda_distances.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.tda_distances import TDADistanceComputer


# ---------------------------------------------------------------------------
# Shared computer instance (L=21, step=1 per paper)
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def computer():
    return TDADistanceComputer(L=21, step=1, max_dim=1, p_norm=2.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def point_cloud_small():
    """Small point cloud (30 time-steps × 3 features) — fits one sub-window."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 3))


@pytest.fixture
def point_cloud_medium():
    """Larger point cloud (63 time-steps × 3 features) — multiple sub-windows."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((63, 3))


@pytest.fixture
def diagram_small(computer, point_cloud_small):
    """Single sub-window diagram from a small point cloud."""
    return computer._compute_persistence_diagram(point_cloud_small[:21])


@pytest.fixture
def diagram_medium(computer, point_cloud_medium):
    return computer._compute_persistence_diagram(point_cloud_medium[:21])


@pytest.fixture
def feature_dict_3assets():
    """Feature dict for 3 assets, each (63, 3)."""
    rng = np.random.default_rng(7)
    tickers = ['A', 'B', 'C']
    return {t: rng.standard_normal((63, 3)) for t in tickers}, tickers


# ---------------------------------------------------------------------------
# Persistence diagram
# ---------------------------------------------------------------------------
class TestPersistenceDiagram:
    def test_persistence_diagram_not_empty(self, diagram_small):
        """H0 homology must be present and be a 2-D array."""
        assert 'H0' in diagram_small, "Diagram must contain 'H0' key."
        assert isinstance(diagram_small['H0'], np.ndarray), "'H0' must be an ndarray."
        assert diagram_small['H0'].ndim == 2, "'H0' must be a 2-D array (n_bars × 2)."
        assert diagram_small['H0'].shape[1] == 2, (
            "'H0' columns must be [birth, death]."
        )

    def test_persistence_diagram_birth_le_death(self, diagram_small):
        """For every finite bar, birth <= death."""
        h0 = diagram_small['H0']
        if len(h0) > 0:
            assert (h0[:, 0] <= h0[:, 1]).all(), "Birth must be <= death."

    def test_persistence_diagram_returns_dict(self, computer, point_cloud_small):
        result = computer._compute_persistence_diagram(point_cloud_small[:21])
        assert isinstance(result, dict)

    def test_persistence_diagram_h1_key_present(self, diagram_small):
        """H1 key must exist (may contain zero rows when no loops detected)."""
        assert 'H1' in diagram_small
        assert diagram_small['H1'].ndim == 2

    def test_compute_all_diagrams_returns_list(self, computer, point_cloud_medium):
        diagrams = computer.compute_all_diagrams(point_cloud_medium)
        assert isinstance(diagrams, list)
        assert len(diagrams) > 0
        assert isinstance(diagrams[0], dict)
        assert 'H0' in diagrams[0]


# ---------------------------------------------------------------------------
# Distance matrix (pairwise AWD over assets)
# ---------------------------------------------------------------------------
class TestDistanceMatrix:
    def test_distance_matrix_symmetry(self, computer, feature_dict_3assets):
        fd, tickers = feature_dict_3assets
        D = computer.compute_distance_matrix_awd(fd, tickers)
        assert np.allclose(D, D.T, atol=1e-8), "Distance matrix must be symmetric."

    def test_distance_matrix_zero_diagonal(self, computer, feature_dict_3assets):
        fd, tickers = feature_dict_3assets
        D = computer.compute_distance_matrix_awd(fd, tickers)
        assert np.allclose(np.diag(D), 0.0, atol=1e-8), (
            "Diagonal entries of distance matrix must be 0."
        )

    def test_distance_matrix_nonneg(self, computer, feature_dict_3assets):
        fd, tickers = feature_dict_3assets
        D = computer.compute_distance_matrix_awd(fd, tickers)
        assert (D >= -1e-10).all(), "All distance-matrix entries must be non-negative."

    def test_distance_matrix_shape(self, computer, feature_dict_3assets):
        fd, tickers = feature_dict_3assets
        n = len(tickers)
        D = computer.compute_distance_matrix_awd(fd, tickers)
        assert D.shape == (n, n), f"Expected ({n}, {n}), got {D.shape}."


# ---------------------------------------------------------------------------
# Average Wasserstein Distance (AWD)
# ---------------------------------------------------------------------------
class TestAWD:
    def test_awd_identical_assets(self, computer):
        """AWD between a point cloud and itself must be 0."""
        rng = np.random.default_rng(2)
        Z = rng.standard_normal((63, 3))
        diags = computer.compute_all_diagrams(Z)
        awd = computer.compute_awd(diags, diags)
        assert np.isclose(awd, 0.0, atol=1e-6), (
            f"AWD of identical point clouds must be 0; got {awd}"
        )

    def test_awd_nonneg(self, computer):
        rng = np.random.default_rng(3)
        Z1 = rng.standard_normal((63, 3))
        Z2 = rng.standard_normal((63, 3))
        d1 = computer.compute_all_diagrams(Z1)
        d2 = computer.compute_all_diagrams(Z2)
        awd = computer.compute_awd(d1, d2)
        assert awd >= 0.0, "AWD must be non-negative."

    def test_awd_symmetric(self, computer):
        rng = np.random.default_rng(4)
        Z1 = rng.standard_normal((63, 3))
        Z2 = rng.standard_normal((63, 3))
        d1 = computer.compute_all_diagrams(Z1)
        d2 = computer.compute_all_diagrams(Z2)
        assert np.isclose(computer.compute_awd(d1, d2),
                          computer.compute_awd(d2, d1), atol=1e-6), (
            "AWD must be symmetric: d(Z1,Z2) == d(Z2,Z1)."
        )

    def test_awd_returns_float(self, computer):
        rng = np.random.default_rng(5)
        Z1 = rng.standard_normal((63, 3))
        Z2 = rng.standard_normal((63, 3))
        d1 = computer.compute_all_diagrams(Z1)
        d2 = computer.compute_all_diagrams(Z2)
        result = computer.compute_awd(d1, d2)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Persistence landscape vector
# ---------------------------------------------------------------------------
class TestLandscapeVector:
    def test_landscape_vector_shape(self, computer, diagram_small):
        """Landscape must return a 1-D (flat) array of length num_fns * resolution."""
        num_fns, resolution = 5, 100
        vec = computer._persistence_landscape(
            diagram_small['H0'], num_fns=num_fns, resolution=resolution
        )
        assert isinstance(vec, np.ndarray), "Landscape vector must be an ndarray."
        assert vec.ndim == 1, f"Landscape vector must be 1-D; got shape {vec.shape}."
        assert vec.shape[0] == num_fns * resolution, (
            f"Expected length {num_fns * resolution}, got {vec.shape[0]}."
        )

    def test_landscape_vector_finite(self, computer, diagram_small):
        vec = computer._persistence_landscape(diagram_small['H0'])
        assert np.isfinite(vec).all(), "Landscape vector must contain only finite values."

    def test_landscape_vector_length_consistent(self, computer, diagram_small, diagram_medium):
        """Two diagrams with the same resolution must give equal-length vectors."""
        num_fns, resolution = 5, 100
        v1 = computer._persistence_landscape(
            diagram_small['H0'], num_fns=num_fns, resolution=resolution
        )
        v2 = computer._persistence_landscape(
            diagram_medium['H0'], num_fns=num_fns, resolution=resolution
        )
        assert v1.shape == v2.shape, (
            "Landscape vectors must have the same length for fixed resolution."
        )

    def test_landscape_vector_nonneg(self, computer, diagram_small):
        """Persistence landscape values are non-negative."""
        vec = computer._persistence_landscape(diagram_small['H0'])
        assert (vec >= 0.0).all(), "Landscape values must be >= 0."


# ---------------------------------------------------------------------------
# Subwindow count formula
# ---------------------------------------------------------------------------
class TestSubwindowCount:
    """Verify the formula K = (T - L) // step + 1 used by TDADistanceComputer."""

    def test_subwindow_count(self):
        """K = (T - L) // step + 1 for T=63, L=21, step=1 gives K=43."""
        T1, L, step = 63, 21, 1
        K = (T1 - L) // step + 1
        assert K == 43, f"Expected K=43; got {K}."

    def test_subwindow_count_matches_computer_output(self, computer):
        """Number of diagrams from compute_all_diagrams matches the formula."""
        rng = np.random.default_rng(99)
        T1, L, step = 63, computer.L, computer.step
        Z = rng.standard_normal((T1, 3))
        diagrams = computer.compute_all_diagrams(Z)
        expected_K = (T1 - L) // step + 1
        assert len(diagrams) == expected_K, (
            f"Expected {expected_K} sub-windows, got {len(diagrams)}."
        )

    def test_subwindow_count_general(self, computer):
        """For T=84, L=21, step=1: K=64."""
        T1, L, step = 84, computer.L, computer.step
        expected_K = (T1 - L) // step + 1
        rng = np.random.default_rng(88)
        Z = rng.standard_normal((T1, 3))
        diagrams = computer.compute_all_diagrams(Z)
        assert len(diagrams) == expected_K
