"""
Conventional (non-TDA) pairwise distance measures for asset feature arrays.

Two measures are implemented:

* **Correlation distance** (d_AC) – based on the average cross-feature Pearson
  correlation between two multi-dimensional time-series.
* **Euclidean distance** (d_AE) – Frobenius norm of the element-wise difference
  between two time-series matrices.

Both are referenced in the paper as baseline alternatives to AWD/ALD.
"""

import logging

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pairwise distance functions                                                  #
# --------------------------------------------------------------------------- #

def correlation_distance(Z_i: np.ndarray, Z_j: np.ndarray) -> float:
    """Compute the average-correlation-based distance between two asset arrays.

    Paper formula: d_AC(Z_i, Z_j) = sqrt(2 * (1 − rho(Z_i, Z_j)))

    where rho is the average Pearson correlation computed independently for
    each feature dimension d, then averaged across all valid dimensions.

    Parameters
    ----------
    Z_i, Z_j : np.ndarray, shape (T, d)
        Feature time-series for assets i and j.  Both must share the same
        shape.

    Returns
    -------
    float
        Non-negative distance value in [0, 2].

    Notes
    -----
    Dimensions whose feature vector has zero standard deviation (constant
    column) are excluded from the average.  If all dimensions are constant,
    rho is set to 0 and the returned distance is sqrt(2).
    """
    Z_i = np.asarray(Z_i, dtype=float)
    Z_j = np.asarray(Z_j, dtype=float)

    if Z_i.ndim == 1:
        Z_i = Z_i[:, None]
    if Z_j.ndim == 1:
        Z_j = Z_j[:, None]

    _, d = Z_i.shape
    correlations: list[float] = []

    for dim in range(d):
        xi = Z_i[:, dim]
        xj = Z_j[:, dim]

        std_i = xi.std()
        std_j = xj.std()

        if std_i == 0.0 or std_j == 0.0:
            # Constant column — correlation is undefined; skip this dimension.
            logger.debug(
                "Dimension %d has zero std for one or both assets; skipping.", dim
            )
            continue

        # Pearson correlation via numpy (numerically stable)
        rho_dim = float(np.corrcoef(xi, xj)[0, 1])

        # Guard against floating-point drift outside [-1, 1]
        rho_dim = max(-1.0, min(1.0, rho_dim))
        correlations.append(rho_dim)

    if len(correlations) == 0:
        logger.warning(
            "All dimensions had zero variance; defaulting rho=0 → dist=sqrt(2)."
        )
        rho = 0.0
    else:
        rho = float(np.mean(correlations))

    distance = float(np.sqrt(2.0 * (1.0 - rho)))
    return distance


def euclidean_distance(Z_i: np.ndarray, Z_j: np.ndarray) -> float:
    """Compute the cumulative Euclidean distance between two asset arrays.

    Paper formula: d_AE(Z_i, Z_j) = sqrt(sum_t ||Z_i_t − Z_j_t||_2^2)

    This equals the Frobenius norm of the difference matrix (Z_i − Z_j).

    Parameters
    ----------
    Z_i, Z_j : np.ndarray, shape (T, d)
        Feature time-series for assets i and j.

    Returns
    -------
    float
        Non-negative distance value.
    """
    Z_i = np.asarray(Z_i, dtype=float)
    Z_j = np.asarray(Z_j, dtype=float)

    diff = Z_i - Z_j
    distance = float(np.linalg.norm(diff, ord="fro"))
    return distance


# --------------------------------------------------------------------------- #
# Distance-matrix builder                                                      #
# --------------------------------------------------------------------------- #

def compute_distance_matrix(
    feature_dict: dict,
    tickers: list[str],
    method: str,
) -> np.ndarray:
    """Compute a symmetric n × n pairwise distance matrix.

    Parameters
    ----------
    feature_dict : dict
        ``{ticker: np.ndarray}`` of shape ``(T, d)``, one array per asset.
    tickers : list[str]
        Ordered list of tickers; defines the row/column order of the matrix.
    method : str
        Distance function to use.  Must be one of:

        * ``'correlation'`` – :func:`correlation_distance` (d_AC)
        * ``'euclidean'``   – :func:`euclidean_distance`   (d_AE)

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric matrix with zero diagonal.

    Raises
    ------
    ValueError
        If *method* is not one of the supported options.
    """
    supported = {"correlation", "euclidean"}
    if method not in supported:
        raise ValueError(
            f"method must be one of {supported!r}; got {method!r}."
        )

    dist_fn = correlation_distance if method == "correlation" else euclidean_distance

    n = len(tickers)
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(feature_dict[tickers[i]], feature_dict[tickers[j]])
            D[i, j] = D[j, i] = d

    assert np.allclose(D, D.T, atol=1e-10), "Distance matrix is not symmetric."
    assert np.allclose(np.diag(D), 0.0, atol=1e-10), "Diagonal is not zero."

    logger.info(
        "Distance matrix (%s, %d × %d) computed. "
        "Min off-diag: %.4f, Max: %.4f",
        method, n, n,
        D[D > 0].min() if (D > 0).any() else 0.0,
        D.max(),
    )
    return D


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(7)
    N_ASSETS = 6
    T, d = 80, 3
    tickers = [f"T{i}" for i in range(N_ASSETS)]
    feature_dict = {t: rng.standard_normal((T, d)) for t in tickers}

    # ---- correlation distance ------------------------------------------------
    D_corr = compute_distance_matrix(feature_dict, tickers, method="correlation")
    assert D_corr.shape == (N_ASSETS, N_ASSETS)
    assert np.allclose(D_corr, D_corr.T)
    assert np.allclose(np.diag(D_corr), 0.0)
    # Self-distance via correlation should be ~0 (rho=1 → sqrt(2*(1-1))=0)
    for t in tickers:
        self_dist = correlation_distance(feature_dict[t], feature_dict[t])
        assert abs(self_dist) < 1e-8, f"Self correlation-distance non-zero: {self_dist}"
    logger.info("Correlation distance matrix OK:\n%s", np.round(D_corr, 4))

    # ---- euclidean distance --------------------------------------------------
    D_euc = compute_distance_matrix(feature_dict, tickers, method="euclidean")
    assert D_euc.shape == (N_ASSETS, N_ASSETS)
    assert np.allclose(D_euc, D_euc.T)
    assert np.allclose(np.diag(D_euc), 0.0)
    for t in tickers:
        self_dist = euclidean_distance(feature_dict[t], feature_dict[t])
        assert abs(self_dist) < 1e-10, f"Self euclidean-distance non-zero: {self_dist}"
    logger.info("Euclidean distance matrix OK:\n%s", np.round(D_euc, 4))

    # ---- edge cases ----------------------------------------------------------
    # Constant column – should not crash; should return sqrt(2) per dimension skipped
    Z_const = np.ones((T, d))
    Z_rand = rng.standard_normal((T, d))
    dist_const = correlation_distance(Z_const, Z_rand)
    logger.info("Constant-column correlation distance: %.4f (expect sqrt(2)≈1.4142)", dist_const)

    # Invalid method
    try:
        compute_distance_matrix(feature_dict, tickers, method="invalid")
        raise AssertionError("Expected ValueError for invalid method.")
    except ValueError:
        pass

    logger.info("conventional_distances.py self-test passed.")
