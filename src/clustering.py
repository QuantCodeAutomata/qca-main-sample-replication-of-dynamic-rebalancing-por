"""
Agglomerative clustering and asset selection utilities for the TDA-based
portfolio construction pipeline.

The paper clusters assets using average-linkage hierarchical clustering with a
data-driven threshold (alpha-th percentile of nearest-neighbour distances).
Selected assets are the top-rho fraction by Sharpe ratio within each
non-singleton cluster.  A retention rule carries forward previously selected
high-Sharpe assets that were displaced by the new clustering.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import sklearn.metrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Threshold helpers                                                             #
# --------------------------------------------------------------------------- #

def compute_nn_distances(D: np.ndarray) -> np.ndarray:
    """Return the nearest-neighbour distance for each asset.

    For each asset i, the nearest-neighbour distance is the smallest
    positive entry in row i of the distance matrix.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Symmetric pairwise distance matrix with zero diagonal.

    Returns
    -------
    np.ndarray, shape (n,)
        Nearest-neighbour distances.  Assets with no valid neighbour (e.g.
        a 1 × 1 matrix) receive distance 0.
    """
    n = D.shape[0]
    nn_dists = np.zeros(n, dtype=float)
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf                          # exclude self
        min_dist = row.min()
        nn_dists[i] = 0.0 if np.isinf(min_dist) else min_dist
    return nn_dists


def compute_threshold(nn_distances: np.ndarray, alpha: float) -> float:
    """Compute the clustering distance threshold.

    alpha-th percentile of nearest-neighbor distances as clustering threshold.

    Parameters
    ----------
    nn_distances : np.ndarray
        Nearest-neighbour distances per asset (output of
        :func:`compute_nn_distances`).
    alpha : float
        Percentile fraction in (0, 1]; e.g. 0.5 → median.

    Returns
    -------
    float
        The alpha-th percentile of *nn_distances*.
    """
    return float(np.percentile(nn_distances, alpha * 100.0))


# --------------------------------------------------------------------------- #
# Clustering                                                                   #
# --------------------------------------------------------------------------- #

def agglomerative_cluster(D: np.ndarray, alpha: float) -> np.ndarray:
    """Cluster assets via average-linkage agglomerative clustering.

    Paper procedure: initialise singletons; merge the closest pair while the
    minimum inter-cluster average-linkage distance is below threshold c.

    Implementation using scipy:
    1. Convert D to condensed form.
    2. Build linkage matrix with ``method='average'``.
    3. Derive threshold c = alpha-th percentile of nearest-neighbour distances.
    4. Cut the dendrogram at height c using ``criterion='distance'``.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Symmetric pairwise distance matrix (zero diagonal).
    alpha : float
        Percentile fraction used to derive the cut height.

    Returns
    -------
    np.ndarray of int, shape (n,)
        Cluster label for each asset (1-indexed, as returned by
        ``scipy.cluster.hierarchy.fcluster``).
    """
    condensed = ssd.squareform(D, checks=False)
    Z = sch.linkage(condensed, method="average")

    nn_dists = compute_nn_distances(D)
    c = compute_threshold(nn_dists, alpha)

    logger.info(
        "Clustering threshold c=%.6f (alpha=%.2f, NN-dist range [%.4f, %.4f]).",
        c, alpha, nn_dists.min(), nn_dists.max(),
    )

    labels = sch.fcluster(Z, t=c, criterion="distance")
    n_clusters = len(np.unique(labels))
    logger.info("agglomerative_cluster: %d assets → %d clusters.", len(labels), n_clusters)
    return labels


# --------------------------------------------------------------------------- #
# Singleton identification                                                      #
# --------------------------------------------------------------------------- #

def identify_singletons(labels: np.ndarray) -> np.ndarray:
    """Build a boolean mask marking singleton-cluster assets.

    Parameters
    ----------
    labels : np.ndarray of int, shape (n,)
        Cluster label for each asset.

    Returns
    -------
    np.ndarray of bool, shape (n,)
        True where the asset belongs to a cluster of size 1 (outlier/singleton).
    """
    unique, counts = np.unique(labels, return_counts=True)
    singleton_labels = set(unique[counts == 1])
    return np.array([lbl in singleton_labels for lbl in labels], dtype=bool)


# --------------------------------------------------------------------------- #
# Asset selection                                                               #
# --------------------------------------------------------------------------- #

def select_assets_from_clusters(
    tickers: list[str],
    labels: np.ndarray,
    sharpe_ratios: pd.Series,
    rho: float = 0.10,
) -> list[str]:
    """Select the top-rho fraction of assets by Sharpe ratio within each cluster.

    rho=0.10, k_c=ceil(0.10*n_c) per paper; singletons excluded.

    Parameters
    ----------
    tickers : list[str]
        Ordered list of asset tickers (same order as *labels*).
    labels : np.ndarray of int, shape (n,)
        Cluster labels.
    sharpe_ratios : pd.Series
        Sharpe ratio for each ticker (index = ticker name).
    rho : float
        Selection fraction within each cluster.

    Returns
    -------
    list[str]
        Selected tickers from non-singleton clusters.
    """
    singleton_mask = identify_singletons(labels)
    selected: list[str] = []

    for cluster_id in np.unique(labels):
        member_mask = (labels == cluster_id)
        if member_mask.sum() == 1:
            # Singleton – excluded from selection per paper.
            continue

        member_tickers = [t for t, m in zip(tickers, member_mask) if m]
        n_c = len(member_tickers)
        k_c = max(1, math.ceil(rho * n_c))

        # Sort by Sharpe ratio descending; pick top k_c.
        cluster_sharpe = sharpe_ratios.reindex(member_tickers).fillna(-np.inf)
        top_k = cluster_sharpe.nlargest(k_c).index.tolist()
        selected.extend(top_k)

        logger.debug(
            "Cluster %d: n_c=%d, k_c=%d, selected=%s", cluster_id, n_c, k_c, top_k
        )

    return selected


# --------------------------------------------------------------------------- #
# Retention rule                                                                #
# --------------------------------------------------------------------------- #

def apply_retention_rule(
    prev_selected: list[str],
    curr_selected: list[str],
    all_tickers: list[str],
    sharpe_ratios: pd.Series,
    rho: float = 0.10,
) -> list[str]:
    """Carry forward previously selected assets that remain in the top rho-fraction.

    Assumed operational interpretation of retention rule:
    An asset from the previous portfolio that is no longer selected by the
    current clustering is retained if its Sharpe ratio is at or above the
    top-(rho * 100)-th percentile among ALL universe assets.  This prevents
    unnecessary turnover for consistently high-quality assets that were
    re-assigned to a different cluster.

    Parameters
    ----------
    prev_selected : list[str]
        Assets selected in the previous rebalancing period.
    curr_selected : list[str]
        Assets selected by the current clustering step.
    all_tickers : list[str]
        Full investable universe.
    sharpe_ratios : pd.Series
        Current-period Sharpe ratios indexed by ticker name.
    rho : float
        Top-fraction threshold (same rho used in :func:`select_assets_from_clusters`).

    Returns
    -------
    list[str]
        Deduplicated union of *curr_selected* and retained assets from
        *prev_selected*.
    """
    curr_set = set(curr_selected)
    prev_set = set(prev_selected)
    displaced = prev_set - curr_set

    if not displaced:
        return list(curr_set)

    # Top-rho Sharpe cutoff across the entire universe.
    universe_sharpe = sharpe_ratios.reindex(all_tickers).dropna()
    cutoff = float(np.percentile(universe_sharpe.values, (1.0 - rho) * 100.0))

    retained: list[str] = []
    for ticker in displaced:
        asset_sharpe = sharpe_ratios.get(ticker, -np.inf)
        if asset_sharpe >= cutoff:
            retained.append(ticker)
            logger.debug(
                "Retention rule: kept %s (Sharpe=%.4f >= cutoff=%.4f).",
                ticker, asset_sharpe, cutoff,
            )
        else:
            logger.debug(
                "Retention rule: dropped %s (Sharpe=%.4f < cutoff=%.4f).",
                ticker, asset_sharpe, cutoff,
            )

    merged = list(curr_set | set(retained))
    return merged


# --------------------------------------------------------------------------- #
# Diagnostics                                                                   #
# --------------------------------------------------------------------------- #

def compute_clustering_diagnostics(
    tickers: list[str],
    labels: np.ndarray,
    D: np.ndarray,
) -> dict:
    """Compute summary statistics for a clustering result.

    Parameters
    ----------
    tickers : list[str]
        Asset tickers (same order as *labels*).
    labels : np.ndarray of int, shape (n,)
        Cluster label per asset.
    D : np.ndarray, shape (n, n)
        Pairwise distance matrix (used as precomputed metric for silhouette).

    Returns
    -------
    dict with keys:
        * ``n_clusters``           – number of distinct clusters
        * ``n_singletons``         – number of singleton clusters
        * ``n_selected_potential`` – assets in non-singleton clusters
        * ``avg_silhouette_score`` – mean silhouette (−1 if not computable)
        * ``cluster_sizes``        – dict {cluster_id: size}
    """
    unique, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(unique.tolist(), counts.tolist()))

    n_clusters = len(unique)
    n_singletons = int((counts == 1).sum())
    n_selected_potential = int((counts > 1).sum() * 1)   # clusters, not assets
    n_selected_potential = int(sum(c for c in counts if c > 1))  # assets in non-singletons

    silhouette = -1.0
    if n_clusters > 1 and not np.all(counts == 1):
        try:
            silhouette = float(
                sklearn.metrics.silhouette_score(D, labels, metric="precomputed")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Silhouette computation failed: %s", exc)

    return {
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "n_selected_potential": n_selected_potential,
        "avg_silhouette_score": silhouette,
        "cluster_sizes": size_map,
    }


# --------------------------------------------------------------------------- #
# Comparison metrics                                                            #
# --------------------------------------------------------------------------- #

def compute_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index between two label vectors.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray of int
        Cluster assignments to compare.

    Returns
    -------
    float
        ARI in [−1, 1]; 1 = perfect agreement.
    """
    return float(sklearn.metrics.adjusted_rand_score(labels_a, labels_b))


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets of selected assets.

    Parameters
    ----------
    set_a, set_b : set
        Sets of ticker strings to compare.

    Returns
    -------
    float
        |A ∩ B| / |A ∪ B|.  Returns 1.0 if both sets are empty.
    """
    union = set_a | set_b
    if not union:
        return 1.0
    return float(len(set_a & set_b) / len(union))


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(99)
    N = 20
    tickers = [f"S{i:02d}" for i in range(N)]

    # Build a synthetic distance matrix (symmetric, zero diagonal)
    X = rng.standard_normal((N, 5))
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(X[i] - X[j]))
            D[i, j] = D[j, i] = d

    # ---- NN distances and threshold -----------------------------------------
    nn_dists = compute_nn_distances(D)
    assert nn_dists.shape == (N,)
    assert np.all(nn_dists >= 0)
    c = compute_threshold(nn_dists, alpha=0.5)
    logger.info("Median NN distance (threshold): %.4f", c)

    # ---- Clustering ----------------------------------------------------------
    labels = agglomerative_cluster(D, alpha=0.5)
    assert labels.shape == (N,)
    logger.info("Labels: %s", labels)

    # ---- Singletons ----------------------------------------------------------
    singleton_mask = identify_singletons(labels)
    assert singleton_mask.shape == (N,)
    logger.info("Singletons: %d / %d", singleton_mask.sum(), N)

    # ---- Asset selection -----------------------------------------------------
    sharpe = pd.Series(rng.standard_normal(N), index=tickers)
    selected = select_assets_from_clusters(tickers, labels, sharpe, rho=0.10)
    logger.info("Selected %d assets: %s", len(selected), selected)

    # ---- Retention rule ------------------------------------------------------
    prev = selected[:3] + [tickers[0]]    # include one that may not be in curr
    retained = apply_retention_rule(prev, selected, tickers, sharpe, rho=0.10)
    logger.info("After retention rule: %d assets", len(retained))

    # ---- Diagnostics ---------------------------------------------------------
    diag = compute_clustering_diagnostics(tickers, labels, D)
    logger.info("Diagnostics: %s", diag)
    assert "n_clusters" in diag
    assert "avg_silhouette_score" in diag

    # ---- ARI and Jaccard -----------------------------------------------------
    labels2 = agglomerative_cluster(D, alpha=0.6)
    ari = compute_ari(labels, labels2)
    logger.info("ARI between alpha=0.5 and alpha=0.6 clusterings: %.4f", ari)
    assert -1.0 <= ari <= 1.0

    j = compute_jaccard(set(selected), set(retained))
    logger.info("Jaccard(selected, retained): %.4f", j)
    assert 0.0 <= j <= 1.0

    # Edge case: empty sets
    assert compute_jaccard(set(), set()) == 1.0
    assert compute_jaccard({"A"}, set()) == 0.0

    logger.info("clustering.py self-test passed.")
