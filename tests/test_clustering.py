"""pytest tests for src/clustering.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    compute_nn_distances,
    compute_threshold,
    agglomerative_cluster,
    identify_singletons,
    select_assets_from_clusters,
    compute_ari,
    compute_jaccard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_distance_matrix():
    """6×6 symmetric distance matrix with two clear clusters of 3."""
    rng = np.random.default_rng(0)
    n = 6
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Within-cluster distances are tight; between-cluster are large
            d = rng.uniform(0.01, 0.10) if (i < 3) == (j < 3) else rng.uniform(0.80, 1.0)
            D[i, j] = D[j, i] = d
    return D


@pytest.fixture
def medium_distance_matrix():
    """12×12 random symmetric distance matrix with zero diagonal."""
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, (12, 12))
    D = (raw + raw.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


@pytest.fixture
def tight_labels():
    """Labels for 6 assets: two clusters of size 3 (1-indexed, as scipy returns)."""
    return np.array([1, 1, 1, 2, 2, 2])


@pytest.fixture
def singleton_labels():
    """Labels where cluster 3 has only one member (asset index 4)."""
    return np.array([1, 1, 2, 2, 3, 1])


@pytest.fixture
def tickers_6():
    return ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']


@pytest.fixture
def sharpe_6(tickers_6):
    """Fixed Sharpe ratios for 6 tickers."""
    return pd.Series([0.5, 1.2, 0.3, 0.9, 0.7, 1.5], index=tickers_6)


# ---------------------------------------------------------------------------
# compute_nn_distances
# ---------------------------------------------------------------------------
class TestComputeNNDistances:
    def test_nn_distances_shape(self, small_distance_matrix):
        n = small_distance_matrix.shape[0]
        nn = compute_nn_distances(small_distance_matrix)
        assert nn.shape == (n,), f"Expected shape ({n},), got {nn.shape}."

    def test_nn_distances_nonneg(self, small_distance_matrix):
        nn = compute_nn_distances(small_distance_matrix)
        assert (nn >= 0.0).all(), "NN distances must be non-negative."

    def test_nn_distances_less_than_max(self, small_distance_matrix):
        """Each NN distance is the minimum off-diagonal entry in that row."""
        D = small_distance_matrix
        n = D.shape[0]
        nn = compute_nn_distances(D)
        for i in range(n):
            row = D[i].copy()
            row[i] = np.inf
            assert np.isclose(nn[i], row.min(), atol=1e-10)


# ---------------------------------------------------------------------------
# compute_threshold
# ---------------------------------------------------------------------------
class TestComputeThreshold:
    def test_threshold_alpha_percentile(self, small_distance_matrix):
        """Threshold must equal the alpha-th percentile of NN distances."""
        alpha = 0.90
        nn_dists = compute_nn_distances(small_distance_matrix)
        expected = np.percentile(nn_dists, alpha * 100)
        actual = compute_threshold(nn_dists, alpha)
        assert np.isclose(actual, expected, rtol=1e-6), (
            f"Expected threshold {expected:.6f}, got {actual:.6f}."
        )

    def test_threshold_is_positive(self, medium_distance_matrix):
        nn_dists = compute_nn_distances(medium_distance_matrix)
        t = compute_threshold(nn_dists, alpha=0.95)
        assert t > 0.0

    def test_threshold_increases_with_alpha(self, medium_distance_matrix):
        nn_dists = compute_nn_distances(medium_distance_matrix)
        t90 = compute_threshold(nn_dists, alpha=0.90)
        t99 = compute_threshold(nn_dists, alpha=0.99)
        assert t99 >= t90, "Threshold must be non-decreasing with alpha."

    def test_threshold_returns_float(self, medium_distance_matrix):
        nn_dists = compute_nn_distances(medium_distance_matrix)
        t = compute_threshold(nn_dists, alpha=0.50)
        assert isinstance(t, float)


# ---------------------------------------------------------------------------
# agglomerative_cluster
# ---------------------------------------------------------------------------
class TestAgglomerativeCluster:
    def test_agglomerative_labels_length(self, small_distance_matrix):
        """Labels array must have the same length as the distance matrix dimension."""
        labels = agglomerative_cluster(small_distance_matrix, alpha=0.90)
        assert len(labels) == small_distance_matrix.shape[0], (
            f"Expected {small_distance_matrix.shape[0]} labels, got {len(labels)}."
        )

    def test_agglomerative_labels_are_integers(self, small_distance_matrix):
        labels = agglomerative_cluster(small_distance_matrix, alpha=0.90)
        assert np.issubdtype(labels.dtype, np.integer), (
            "Cluster labels must be integer-valued."
        )

    def test_agglomerative_two_clusters(self, small_distance_matrix):
        """Block-diagonal distance matrix should yield >= 2 distinct clusters."""
        labels = agglomerative_cluster(small_distance_matrix, alpha=0.50)
        assert len(np.unique(labels)) >= 2, "Expected at least 2 clusters."

    def test_agglomerative_returns_ndarray(self, medium_distance_matrix):
        labels = agglomerative_cluster(medium_distance_matrix, alpha=0.95)
        assert isinstance(labels, np.ndarray)

    def test_agglomerative_all_labeled(self, medium_distance_matrix):
        """Every asset must receive a cluster label."""
        n = medium_distance_matrix.shape[0]
        labels = agglomerative_cluster(medium_distance_matrix, alpha=0.50)
        assert len(labels) == n
        assert not np.any(np.isnan(labels.astype(float)))


# ---------------------------------------------------------------------------
# identify_singletons
# ---------------------------------------------------------------------------
class TestIdentifySingletons:
    def test_singleton_identification(self, singleton_labels):
        """Cluster 3 has only asset at index 4 → that index must be True."""
        mask = identify_singletons(singleton_labels)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask[4], "Asset at index 4 (cluster 3, size 1) should be a singleton."

    def test_no_singletons_when_all_clusters_large(self, tight_labels):
        """No singletons when all clusters have >= 2 members."""
        mask = identify_singletons(tight_labels)
        assert not mask.any(), "No singletons when all clusters have >= 2 members."

    def test_singleton_mask_length(self, singleton_labels):
        """Mask length must match label array length."""
        mask = identify_singletons(singleton_labels)
        assert len(mask) == len(singleton_labels)

    def test_non_singleton_assets_are_false(self, singleton_labels):
        """Non-singleton assets must have False in the mask."""
        mask = identify_singletons(singleton_labels)
        # Clusters 1 and 2 are non-singletons; their members should be False.
        for idx in [0, 1, 2, 3, 5]:
            assert not mask[idx], f"Asset {idx} is not a singleton but mask[{idx}]=True."


# ---------------------------------------------------------------------------
# select_assets_from_clusters
# ---------------------------------------------------------------------------
class TestSelectAssets:
    def test_select_assets_rho(
        self, tight_labels, tickers_6, sharpe_6, small_distance_matrix
    ):
        """Selected count per non-singleton cluster must be ceil(rho * cluster_size)."""
        rho = 0.10
        selected = select_assets_from_clusters(tickers_6, tight_labels, sharpe_6, rho=rho)
        # Two clusters of 3; ceil(0.10 * 3) = 1 per cluster → 2 total
        expected_total = 2 * math.ceil(rho * 3)
        assert len(selected) == expected_total, (
            f"Expected {expected_total} selected tickers, got {len(selected)}: {selected}."
        )

    def test_select_assets_excludes_singletons(
        self, singleton_labels, tickers_6, sharpe_6
    ):
        """Assets in singleton clusters must not appear in selected set."""
        singleton_mask = identify_singletons(singleton_labels)
        singleton_tickers = {t for t, m in zip(tickers_6, singleton_mask) if m}
        selected = select_assets_from_clusters(tickers_6, singleton_labels, sharpe_6, rho=0.50)
        overlap = singleton_tickers & set(selected)
        assert len(overlap) == 0, (
            f"Singleton tickers {overlap} found in selected set."
        )

    def test_select_assets_returns_list_of_str(
        self, tight_labels, tickers_6, sharpe_6
    ):
        selected = select_assets_from_clusters(tickers_6, tight_labels, sharpe_6, rho=0.50)
        assert isinstance(selected, list)
        assert all(isinstance(t, str) for t in selected)

    def test_select_assets_within_universe(
        self, tight_labels, tickers_6, sharpe_6
    ):
        """All selected tickers must be members of the universe."""
        selected = select_assets_from_clusters(tickers_6, tight_labels, sharpe_6, rho=0.50)
        assert all(t in tickers_6 for t in selected), (
            "Selected tickers must be in the original universe."
        )

    def test_select_assets_no_duplicates(
        self, tight_labels, tickers_6, sharpe_6
    ):
        selected = select_assets_from_clusters(tickers_6, tight_labels, sharpe_6, rho=0.50)
        assert len(selected) == len(set(selected)), "No duplicate tickers allowed."

    def test_select_assets_top_sharpe(
        self, tight_labels, tickers_6, sharpe_6
    ):
        """With rho=0.50, top-50% by Sharpe from each cluster must be selected."""
        selected = select_assets_from_clusters(tickers_6, tight_labels, sharpe_6, rho=0.50)
        # Cluster 1: T0(0.5), T1(1.2), T2(0.3) → top 2 = T1, T0
        # Cluster 2: T3(0.9), T4(0.7), T5(1.5) → top 2 = T5, T3
        assert 'T1' in selected, "T1 has highest Sharpe in cluster 1 and must be selected."
        assert 'T5' in selected, "T5 has highest Sharpe in cluster 2 and must be selected."


# ---------------------------------------------------------------------------
# compute_ari
# ---------------------------------------------------------------------------
class TestARI:
    def test_ari_identical(self, tight_labels):
        """ARI(labels, labels) must equal 1.0."""
        ari = compute_ari(tight_labels, tight_labels)
        assert np.isclose(ari, 1.0, atol=1e-8), f"Expected ARI=1.0, got {ari}."

    def test_ari_random(self):
        """ARI between two random label arrays must lie in [-1, 1]."""
        rng = np.random.default_rng(9)
        a = rng.integers(0, 4, size=30)
        b = rng.integers(0, 4, size=30)
        ari = compute_ari(a, b)
        assert -1.0 <= ari <= 1.0, f"ARI must lie in [-1, 1]; got {ari}."

    def test_ari_returns_float(self, tight_labels, singleton_labels):
        ari = compute_ari(tight_labels, singleton_labels)
        assert isinstance(ari, float)


# ---------------------------------------------------------------------------
# compute_jaccard
# ---------------------------------------------------------------------------
class TestJaccard:
    def test_jaccard_identical_sets(self):
        """Jaccard(A, A) must equal 1.0."""
        A = {'T1', 'T2', 'T3', 'T4', 'T5'}
        assert np.isclose(compute_jaccard(A, A), 1.0), "Jaccard(A, A) must be 1."

    def test_jaccard_disjoint_sets(self):
        """Jaccard(A, B) must equal 0.0 when A ∩ B is empty."""
        A = {'T1', 'T2', 'T3'}
        B = {'T4', 'T5', 'T6'}
        assert np.isclose(compute_jaccard(A, B), 0.0), (
            "Jaccard must be 0 for disjoint sets."
        )

    def test_jaccard_partial_overlap(self):
        """Jaccard of two partially overlapping sets."""
        A = {'T1', 'T2', 'T3'}
        B = {'T2', 'T3', 'T4'}
        # |A ∩ B| = 2, |A ∪ B| = 4 → Jaccard = 0.5
        j = compute_jaccard(A, B)
        assert np.isclose(j, 0.5, atol=1e-8), f"Expected 0.5, got {j}."

    def test_jaccard_range(self):
        """Jaccard similarity must lie in [0, 1]."""
        rng = np.random.default_rng(13)
        A = {f'T{x}' for x in rng.integers(0, 20, size=10).tolist()}
        B = {f'T{x}' for x in rng.integers(0, 20, size=10).tolist()}
        j = compute_jaccard(A, B)
        assert 0.0 <= j <= 1.0, f"Jaccard must be in [0, 1]; got {j}."

    def test_jaccard_symmetry(self):
        A = {'T1', 'T2', 'T3', 'T4'}
        B = {'T3', 'T4', 'T5', 'T6'}
        assert np.isclose(compute_jaccard(A, B), compute_jaccard(B, A)), (
            "Jaccard must be symmetric."
        )

    def test_jaccard_empty_sets(self):
        """Both sets empty → Jaccard = 1.0 (per implementation)."""
        assert np.isclose(compute_jaccard(set(), set()), 1.0)
