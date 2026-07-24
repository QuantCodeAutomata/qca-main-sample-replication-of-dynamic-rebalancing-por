"""
TDA-based pairwise asset distance measures: Average Wasserstein Distance (AWD)
and Average Landscape Distance (ALD).

Each asset's feature time-series is treated as a point cloud.  The cloud is
split into overlapping subwindows; a Vietoris-Rips persistence diagram is
computed for each subwindow; distances between diagrams are averaged across
subwindows and homology dimensions.

External dependencies: ripser, persim.
Optional (preferred when available): gtda, gudhi.
"""

import logging

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Persistence diagram engine — prefer giotto-tda / gudhi, fall back to ripser #
# --------------------------------------------------------------------------- #
try:
    from gtda.homology import VietorisRipsPersistence as _GTDAVRips  # noqa: F401
    _BACKEND = "gtda"
except ImportError:
    _BACKEND = "ripser"

try:
    import gudhi  # noqa: F401
    _GUDHI_AVAILABLE = True
except ImportError:
    _GUDHI_AVAILABLE = False

import ripser
import persim

logger = logging.getLogger(__name__)
logger.debug("TDA backend: %s | gudhi available: %s", _BACKEND, _GUDHI_AVAILABLE)

_EMPTY_DIAGRAM = np.zeros((0, 2), dtype=float)


# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #

def _filter_infinite(dgm: np.ndarray) -> np.ndarray:
    """Remove bars whose death value is infinite (essential classes)."""
    if dgm is None or len(dgm) == 0:
        return _EMPTY_DIAGRAM
    finite_mask = np.isfinite(dgm[:, 1])
    return dgm[finite_mask]


def _safe_diagram(dgm: np.ndarray | None) -> np.ndarray:
    """Return a well-formed finite persistence diagram array."""
    if dgm is None:
        return _EMPTY_DIAGRAM
    dgm = np.asarray(dgm, dtype=float)
    if dgm.ndim != 2 or dgm.shape[1] < 2:
        return _EMPTY_DIAGRAM
    return _filter_infinite(dgm)


# --------------------------------------------------------------------------- #
# Main class                                                                   #
# --------------------------------------------------------------------------- #

class TDADistanceComputer:
    """Compute AWD and ALD pairwise distances between asset feature clouds.

    Parameters
    ----------
    L : int
        Sliding subwindow length.
        L=21 sliding subwindow length per paper; p_norm=2.0 chosen as L2;
        aggregation over r=0,1 by averaging.
    step : int
        Step size between consecutive subwindows.
    max_dim : int
        Maximum homology dimension (0 and 1 are used).
    p_norm : float
        Norm order used in the landscape distance computation.
    """

    def __init__(
        self,
        L: int = 21,
        step: int = 1,
        max_dim: int = 1,
        p_norm: float = 2.0,
    ) -> None:
        self.L = L
        self.step = step
        self.max_dim = max_dim
        self.p_norm = p_norm

    # ----------------------------------------------------------------------- #
    # Subwindow extraction                                                     #
    # ----------------------------------------------------------------------- #

    def _get_subwindows(self, Z: np.ndarray) -> list[np.ndarray]:
        """Split a T × d point cloud into overlapping subwindows.

        Split T x d point cloud into K sliding subwindows of length L with
        step=1.

        Parameters
        ----------
        Z : np.ndarray, shape (T, d)
            Asset feature time-series.

        Returns
        -------
        list of np.ndarray, each of shape (L, d)
        """
        T = len(Z)
        if T < self.L:
            logger.warning(
                "Time-series length %d < subwindow length %d; "
                "returning single truncated window.", T, self.L
            )
            return [Z]
        K = (T - self.L) // self.step + 1
        return [Z[k * self.step: k * self.step + self.L] for k in range(K)]

    # ----------------------------------------------------------------------- #
    # Persistence diagram computation                                          #
    # ----------------------------------------------------------------------- #

    def _compute_persistence_diagram(self, pts: np.ndarray) -> dict[str, np.ndarray]:
        """Compute Vietoris-Rips persistence for one subwindow using ripser.

        Compute Vietoris-Rips persistence for one subwindow using ripser.

        Parameters
        ----------
        pts : np.ndarray, shape (L, d)
            Point cloud for one subwindow.

        Returns
        -------
        dict with keys ``'H0'`` and ``'H1'``, each an ndarray of shape (n, 2).
        """
        try:
            result = ripser.ripser(pts, maxdim=self.max_dim)
            dgms = result["dgms"]
            h0 = _safe_diagram(dgms[0] if len(dgms) > 0 else None)
            h1 = _safe_diagram(dgms[1] if len(dgms) > 1 else None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ripser failed on subwindow: %s – using empty diagrams.", exc)
            h0, h1 = _EMPTY_DIAGRAM.copy(), _EMPTY_DIAGRAM.copy()
        return {"H0": h0, "H1": h1}

    def compute_all_diagrams(self, Z: np.ndarray) -> list[dict[str, np.ndarray]]:
        """Compute persistence diagrams for all K subwindows of asset point cloud Z.

        Compute persistence diagrams for all K subwindows of asset point cloud Z.

        Parameters
        ----------
        Z : np.ndarray, shape (T, d)
            Asset feature array (rows = time steps, cols = features).

        Returns
        -------
        list of dicts, one per subwindow, each with keys ``'H0'`` and ``'H1'``.
        """
        subwindows = self._get_subwindows(Z)
        return [self._compute_persistence_diagram(sw) for sw in subwindows]

    # ----------------------------------------------------------------------- #
    # Wasserstein distance between two diagrams                                #
    # ----------------------------------------------------------------------- #

    def wasserstein_distance_pair(
        self, dgm1: np.ndarray, dgm2: np.ndarray
    ) -> float:
        """Compute the Wasserstein distance between two persistence diagrams.

        Wasserstein p-distance between two persistence diagrams using persim.
        Infinite bars are removed before comparison.

        Parameters
        ----------
        dgm1, dgm2 : np.ndarray, shape (n, 2)
            Persistence diagrams (birth, death pairs).

        Returns
        -------
        float
        """
        d1 = _safe_diagram(dgm1)
        d2 = _safe_diagram(dgm2)

        if len(d1) == 0 and len(d2) == 0:
            return 0.0

        try:
            dist = persim.wasserstein(d1, d2)
            return float(dist)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "persim.wasserstein failed (%s); falling back to sliced_wasserstein.", exc
            )

        try:
            dist = persim.sliced_wasserstein(d1, d2)
            return float(dist)
        except Exception as exc2:  # noqa: BLE001
            logger.warning(
                "persim.sliced_wasserstein also failed (%s); returning 0.", exc2
            )
            return 0.0

    # ----------------------------------------------------------------------- #
    # AWD                                                                      #
    # ----------------------------------------------------------------------- #

    def compute_awd(
        self,
        diags_i: list[dict[str, np.ndarray]],
        diags_j: list[dict[str, np.ndarray]],
    ) -> float:
        """Compute Average Wasserstein Distance between two assets.

        Average Wasserstein Distance between two assets.
        Subwindow weights are uniform (1/K).  Distances from H0 and H1 are
        averaged across homology dimensions r=0,1.

        Parameters
        ----------
        diags_i, diags_j : list of diagram dicts
            Output of :meth:`compute_all_diagrams` for assets i and j.

        Returns
        -------
        float
        """
        K = len(diags_i)
        if K == 0:
            return 0.0

        w_k = 1.0 / K
        total = 0.0
        for k in range(K):
            d_h0 = self.wasserstein_distance_pair(diags_i[k]["H0"], diags_j[k]["H0"])
            d_h1 = self.wasserstein_distance_pair(diags_i[k]["H1"], diags_j[k]["H1"])
            total += w_k * (d_h0 + d_h1) / 2.0  # average over r=0,1
        return total

    # ----------------------------------------------------------------------- #
    # Persistence landscape and ALD                                            #
    # ----------------------------------------------------------------------- #

    def _persistence_landscape(
        self,
        dgm: np.ndarray,
        num_fns: int = 5,
        resolution: int = 100,
    ) -> np.ndarray:
        """Vectorise a persistence diagram as a persistence landscape.

        Custom implementation — Context7 found no simple standalone landscape
        function in persim.

        For a grid of t values in [min_birth, max_death], the k-th landscape
        function is:
            lambda_k(t) = k-th largest value of max(0, min(t−b, d−t))
                          over all finite bars (b, d) in the diagram.

        Parameters
        ----------
        dgm : np.ndarray, shape (n, 2)
            Persistence diagram.
        num_fns : int
            Number of landscape functions to retain (k = 1 … num_fns).
        resolution : int
            Number of t-grid points.

        Returns
        -------
        np.ndarray, shape (num_fns * resolution,)
            Flattened landscape vector.
        """
        d = _safe_diagram(dgm)
        zero_vec = np.zeros(num_fns * resolution, dtype=float)

        if len(d) == 0:
            return zero_vec

        births = d[:, 0]
        deaths = d[:, 1]

        t_min = births.min()
        t_max = deaths.max()
        if t_min >= t_max:
            return zero_vec

        t_grid = np.linspace(t_min, t_max, resolution)  # (resolution,)

        # For each t, compute tent-function values for all bars: (resolution, n_bars)
        tent = np.maximum(
            0.0,
            np.minimum(
                t_grid[:, None] - births[None, :],   # t - b
                deaths[None, :] - t_grid[:, None],   # d - t
            ),
        )

        # landscape_k(t) = k-th largest tent value at t (zero-padded if < k bars)
        n_bars = tent.shape[1]
        k_eff = min(num_fns, n_bars)

        # Partition-select the top-k in descending order for each t
        # np.partition gives the (n-k)-th smallest, i.e., k-th largest
        top_k = np.partition(tent, n_bars - k_eff, axis=1)[:, n_bars - k_eff:]
        top_k = np.sort(top_k, axis=1)[:, ::-1]  # sort desc → lambda_1, lambda_2, …

        landscape = np.zeros((resolution, num_fns), dtype=float)
        landscape[:, :k_eff] = top_k

        return landscape.flatten()

    def landscape_distance(self, dgm1: np.ndarray, dgm2: np.ndarray) -> float:
        """L^p distance between the persistence landscapes of two diagrams.

        Parameters
        ----------
        dgm1, dgm2 : np.ndarray
            Persistence diagrams.

        Returns
        -------
        float
        """
        v1 = self._persistence_landscape(dgm1)
        v2 = self._persistence_landscape(dgm2)
        return float(np.linalg.norm(v1 - v2, ord=self.p_norm))

    def compute_ald(
        self,
        diags_i: list[dict[str, np.ndarray]],
        diags_j: list[dict[str, np.ndarray]],
    ) -> float:
        """Compute Average Landscape Distance between two assets.

        Average Landscape Distance between two assets.
        Mirrors :meth:`compute_awd` but uses landscape distance instead of
        Wasserstein distance.

        Parameters
        ----------
        diags_i, diags_j : list of diagram dicts
            Output of :meth:`compute_all_diagrams` for assets i and j.

        Returns
        -------
        float
        """
        K = len(diags_i)
        if K == 0:
            return 0.0

        w_k = 1.0 / K
        total = 0.0
        for k in range(K):
            d_h0 = self.landscape_distance(diags_i[k]["H0"], diags_j[k]["H0"])
            d_h1 = self.landscape_distance(diags_i[k]["H1"], diags_j[k]["H1"])
            total += w_k * (d_h0 + d_h1) / 2.0
        return total

    # ----------------------------------------------------------------------- #
    # Distance-matrix builders                                                 #
    # ----------------------------------------------------------------------- #

    def compute_distance_matrix_awd(
        self, feature_dict: dict, tickers: list[str]
    ) -> np.ndarray:
        """Build the n × n AWD distance matrix over all tickers.

        Compute n x n AWD distance matrix over all tickers.

        Parameters
        ----------
        feature_dict : dict
            ``{ticker: np.ndarray}`` of shape ``(T, d)``.
        tickers : list[str]
            Ordered list of tickers.

        Returns
        -------
        np.ndarray, shape (n, n)
            Symmetric, zero-diagonal distance matrix.
        """
        n = len(tickers)
        logger.info("Computing AWD diagrams for %d tickers …", n)
        all_diags: dict[str, list[dict]] = {
            t: self.compute_all_diagrams(feature_dict[t]) for t in tickers
        }

        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.compute_awd(all_diags[tickers[i]], all_diags[tickers[j]])
                D[i, j] = D[j, i] = d

        assert np.allclose(D, D.T, atol=1e-10), "Distance matrix not symmetric"
        assert np.allclose(np.diag(D), 0.0, atol=1e-10), "Diagonal not zero"
        logger.info("AWD matrix (%d × %d) computed.", n, n)
        return D

    def compute_distance_matrix_ald(
        self, feature_dict: dict, tickers: list[str]
    ) -> np.ndarray:
        """Build the n × n ALD distance matrix over all tickers.

        Compute n x n ALD distance matrix.

        Parameters
        ----------
        feature_dict : dict
            ``{ticker: np.ndarray}`` of shape ``(T, d)``.
        tickers : list[str]
            Ordered list of tickers.

        Returns
        -------
        np.ndarray, shape (n, n)
            Symmetric, zero-diagonal distance matrix.
        """
        n = len(tickers)
        logger.info("Computing ALD diagrams for %d tickers …", n)
        all_diags: dict[str, list[dict]] = {
            t: self.compute_all_diagrams(feature_dict[t]) for t in tickers
        }

        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.compute_ald(all_diags[tickers[i]], all_diags[tickers[j]])
                D[i, j] = D[j, i] = d

        assert np.allclose(D, D.T, atol=1e-10), "ALD matrix not symmetric"
        logger.info("ALD matrix (%d × %d) computed.", n, n)
        return D


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(0)
    N_ASSETS = 5
    T = 63
    D_FEAT = 3
    tickers = [f"A{i}" for i in range(N_ASSETS)]
    feature_dict = {t: rng.standard_normal((T, D_FEAT)) for t in tickers}

    computer = TDADistanceComputer(L=21, step=1, max_dim=1, p_norm=2.0)

    logger.info("--- AWD ---")
    D_awd = computer.compute_distance_matrix_awd(feature_dict, tickers)
    logger.info("AWD matrix:\n%s", np.round(D_awd, 4))
    assert D_awd.shape == (N_ASSETS, N_ASSETS)
    assert np.allclose(D_awd, D_awd.T)
    assert np.allclose(np.diag(D_awd), 0.0)

    logger.info("--- ALD ---")
    D_ald = computer.compute_distance_matrix_ald(feature_dict, tickers)
    logger.info("ALD matrix:\n%s", np.round(D_ald, 4))
    assert D_ald.shape == (N_ASSETS, N_ASSETS)
    assert np.allclose(D_ald, D_ald.T)

    # Verify landscape vectorisation
    dgm_test = np.array([[0.0, 1.0], [0.2, 0.8], [0.5, 0.9]])
    vec = computer._persistence_landscape(dgm_test, num_fns=3, resolution=50)
    assert vec.shape == (3 * 50,), f"Unexpected landscape shape: {vec.shape}"
    assert np.all(vec >= 0.0), "Landscape values must be non-negative"

    logger.info("tda_distances.py self-test passed.")
