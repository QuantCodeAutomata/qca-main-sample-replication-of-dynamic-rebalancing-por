"""Rolling backtest engine for the DRMV + clustering pipeline."""

import logging
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dependency imports – wrapped so the module remains importable while sibling
# modules are still being developed.
# ---------------------------------------------------------------------------
try:
    from src.indicators import compute_all_indicators        # noqa: F401,F403
except ImportError:
    compute_all_indicators = None                            # type: ignore[assignment]

try:
    from src.clustering import cluster_assets                # noqa: F401,F403
except ImportError:
    cluster_assets = None                                    # type: ignore[assignment]

try:
    from src.conventional_distances import (                 # noqa: F401,F403
        compute_awd_matrix,
        compute_ald_matrix,
    )
except ImportError:
    compute_awd_matrix = None                                # type: ignore[assignment]
    compute_ald_matrix = None                                # type: ignore[assignment]

try:
    from src.tda_distances import TDADistanceComputer        # noqa: F401
except ImportError:
    TDADistanceComputer = None                               # type: ignore[assignment]

from src.optimizer import (
    compute_covariance,
    compute_mean_return,
    compute_portfolio_returns,
    solve_drmv,
    tune_lambda,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clustering specification registry
# ---------------------------------------------------------------------------
# Each entry maps a spec name to the distance metric and feature source used
# when building the pairwise asset distance matrix.
CLUSTERING_SPECS: dict[str, dict[str, str]] = {
    # Sentiment-feature specs
    "AWDS": {"dist": "awd",         "features": "sentiment"},
    "APLS": {"dist": "ald",         "features": "sentiment"},
    "ACS":  {"dist": "correlation", "features": "sentiment"},
    "AES":  {"dist": "euclidean",   "features": "sentiment"},
    # Technical-indicator specs
    "AWDI": {"dist": "awd",         "features": "indicators"},
    "APLI": {"dist": "ald",         "features": "indicators"},
    "ACI":  {"dist": "correlation", "features": "indicators"},
    "AEI":  {"dist": "euclidean",   "features": "indicators"},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_feature_panel(
    indicators_dict: dict,
    sentiment_panel: pd.DataFrame | dict,
    tickers: list[str],
    dates: pd.DatetimeIndex,
    kind: str,
) -> np.ndarray:
    """Build a (T, N, F) feature array for the given date range.

    Parameters
    ----------
    indicators_dict : {ticker: pd.DataFrame} – date-indexed technical indicators
                      with columns ['RSI', 'SO', 'MACD']; produced by
                      compute_all_indicators.
    sentiment_panel : sentiment data for 'sentiment' specs.
                      Accepted formats:
                        - pd.DataFrame with dates as index and tickers as columns
                          (flat, one sentiment score per ticker-day);
                        - pd.DataFrame with MultiIndex columns (ticker, feature);
                        - dict {ticker: pd.Series or pd.DataFrame}.
    tickers         : universe of asset identifiers (length N).
    dates           : DatetimeIndex of length T; must be a subset of the full index.
    kind            : 'indicators' → F=3 panel [RSI, SO, MACD];
                      'sentiment'  → F=4 panel [RSI, SO, MACD, sentiment_score].

    Returns
    -------
    np.ndarray of shape (T, N, F).

    Implementation notes
    --------------------
    - 'indicators' specs (AWDI, APLI, ACI, AEI) use only the 3 technical
      indicators, exactly as described in the paper (3D feature point cloud).
    - 'sentiment' specs (AWDS, APLS, ACS, AES) concatenate the 3 technical
      indicators with one sentiment score to form the 4D point cloud.
    - Standardisation is applied *after* this function, so raw values are
      returned here.
    """
    n = len(tickers)
    T = len(dates)
    indicator_cols = ["RSI", "SO", "MACD"]
    F_ind = len(indicator_cols)

    # ------------------------------------------------------------------ #
    # Build (T, N, 3) indicator panel from indicators_dict                #
    # ------------------------------------------------------------------ #
    ind_panel = np.zeros((T, n, F_ind))
    for i, ticker in enumerate(tickers):
        if ticker in indicators_dict:
            df_ind = (
                indicators_dict[ticker]
                .reindex(dates)[indicator_cols]
                .fillna(0.0)
            )
            ind_panel[:, i, :] = df_ind.values

    if kind == "indicators":
        return ind_panel  # (T, N, 3)

    # ------------------------------------------------------------------ #
    # Build (T, N, 1) sentiment feature, then concatenate → (T, N, 4)    #
    # ------------------------------------------------------------------ #
    sent_col = np.zeros((T, n, 1))

    if isinstance(sentiment_panel, dict):
        # {ticker: pd.Series/pd.DataFrame with date index}
        for i, ticker in enumerate(tickers):
            if ticker in sentiment_panel:
                raw = sentiment_panel[ticker]
                if isinstance(raw, pd.DataFrame):
                    raw = raw.mean(axis=1)  # aggregate multiple features → scalar
                col = raw.reindex(dates).fillna(0.0)
                sent_col[:, i, 0] = col.values

    elif isinstance(sentiment_panel, pd.DataFrame):
        if isinstance(sentiment_panel.columns, pd.MultiIndex):
            # MultiIndex: level 0 = ticker, level 1 = feature
            top_level = sentiment_panel.columns.get_level_values(0).unique()
            for i, ticker in enumerate(tickers):
                if ticker in top_level:
                    sub = sentiment_panel[ticker].reindex(dates).fillna(0.0)
                    if isinstance(sub, pd.DataFrame):
                        sub = sub.mean(axis=1)
                    sent_col[:, i, 0] = sub.values
        else:
            # Flat DataFrame: index = dates, columns = tickers
            # (output of create_mock_sentiment_panel)
            for i, ticker in enumerate(tickers):
                if ticker in sentiment_panel.columns:
                    col = sentiment_panel[ticker].reindex(dates).fillna(0.0)
                    sent_col[:, i, 0] = col.values

    # Concatenate [RSI, SO, MACD, sentiment] → (T, N, 4)
    return np.concatenate([ind_panel, sent_col], axis=-1)


def _standardize_features(panel: np.ndarray) -> np.ndarray:
    """Standardize (T, N, F) panel using its own mean/std along the time axis."""
    mean = panel.mean(axis=0, keepdims=True)
    std  = panel.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-12, 1.0, std)
    return (panel - mean) / std


def _compute_distance_matrix(
    features_is: np.ndarray,
    tickers: list[str],
    dist_type: str,
) -> np.ndarray:
    """Compute (N, N) symmetric pairwise distance matrix.

    Parameters
    ----------
    features_is : (T, N, F) standardised in-sample feature array.
    tickers     : asset identifiers of length N.
    dist_type   : one of 'awd', 'ald', 'correlation', 'euclidean'.

    For 'awd' and 'ald' the raw (T, N, F) panel is forwarded to the
    imported distance functions from src.conventional_distances.
    For 'correlation' and 'euclidean' distances are computed here
    from a (N, T*F) flattened representation.
    """
    n = len(tickers)
    T, _, F = features_is.shape
    # (N, T*F) – each asset represented as a single long feature vector
    flat = features_is.transpose(1, 0, 2).reshape(n, T * F)

    if dist_type == "correlation":
        corr = np.corrcoef(flat)
        corr = np.clip(corr, -1.0, 1.0)
        dist_matrix = 1.0 - corr

    elif dist_type == "euclidean":
        diff = flat[:, None, :] - flat[None, :, :]   # (N, N, T*F)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    elif dist_type == "awd":
        if compute_awd_matrix is None:
            raise ImportError("src.conventional_distances.compute_awd_matrix is not available.")
        dist_matrix = compute_awd_matrix(features_is, tickers)

    elif dist_type == "ald":
        if compute_ald_matrix is None:
            raise ImportError("src.conventional_distances.compute_ald_matrix is not available.")
        dist_matrix = compute_ald_matrix(features_is, tickers)

    else:
        raise ValueError(f"Unknown distance type: {dist_type!r}")

    # Guarantee symmetry and zero diagonal
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix.astype(float)


def _compute_sharpe(returns_df: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    """Compute in-sample Sharpe ratio (mean/std, no annualisation) per ticker."""
    sharpe: dict[str, float] = {}
    for ticker in tickers:
        r = returns_df[ticker].dropna()
        std = r.std()
        sharpe[ticker] = float(r.mean() / std) if std > 1e-12 else -np.inf
    return sharpe


def _select_assets_from_clusters(
    labels: np.ndarray,
    tickers: list[str],
    sharpe_dict: dict[str, float],
    prev_selected: list[str],
) -> list[str]:
    """Select one representative per non-singleton cluster.

    Retention rule: within each cluster, previously selected assets are
    preferred over newcomers to reduce turnover; the candidate with the
    highest in-sample Sharpe ratio is then chosen from the preferred pool.
    Singleton clusters are excluded from the selection.
    """
    unique_labels = np.unique(labels)
    selected: list[str] = []
    for label in unique_labels:
        members = [tickers[i] for i, lbl in enumerate(labels) if lbl == label]
        if len(members) <= 1:
            continue  # skip singletons
        prev_in_cluster = [t for t in members if t in prev_selected]
        candidates = prev_in_cluster if prev_in_cluster else members
        best = max(candidates, key=lambda t: sharpe_dict.get(t, -np.inf))
        selected.append(best)
    return selected


def _prepare_drifted_weights(
    selected: list[str],
    prev_selected: list[str],
    prev_w_plus_map: dict[str, float],
    prev_w_minus_map: dict[str, float],
    returns_last: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Build initialisation weight vectors for the DRMV solve.

    Mark-to-market drifted prior weights: prior holdings updated by realized
    returns of the final in-sample day, then renormalized.  New assets (not
    in prev_selected) start at zero so the solver infers their entry cost.

    Parameters
    ----------
    selected        : tickers selected for this window (length n_sel).
    prev_selected   : tickers selected in the previous window.
    prev_w_plus_map : {ticker → prior long weight}.
    prev_w_minus_map: {ticker → prior short weight}.
    returns_last    : last in-sample day's returns (Series indexed by ticker).

    Returns
    -------
    w_plus_init, w_minus_init : ndarrays of length n_sel.
    """
    n_sel = len(selected)
    w_plus_init  = np.zeros(n_sel)
    w_minus_init = np.zeros(n_sel)

    for i, ticker in enumerate(selected):
        if ticker in prev_selected:
            r = float(returns_last.get(ticker, 0.0))
            w_plus_init[i]  = prev_w_plus_map.get(ticker,  0.0) * (1.0 + r)
            w_minus_init[i] = prev_w_minus_map.get(ticker, 0.0) * (1.0 + r)

    # Renormalize each weight type independently (keeps sum stable)
    total_plus  = w_plus_init.sum()
    total_minus = w_minus_init.sum()
    if total_plus > 1e-12:
        w_plus_init  = w_plus_init  / total_plus  * total_plus   # identity – keeps magnitude
    if total_minus > 1e-12:
        w_minus_init = w_minus_init / total_minus * total_minus  # identity – keeps magnitude

    return w_plus_init, w_minus_init


# ---------------------------------------------------------------------------
# Mark-to-market weight drift
# ---------------------------------------------------------------------------

def compute_mark_to_market_weights(
    w_prev: np.ndarray,
    returns_last: np.ndarray,
) -> np.ndarray:
    """Drift portfolio weights by realised asset returns, then renormalise.

    Parameters
    ----------
    w_prev       : prior weight vector (any length).
    returns_last : vector of realised returns for the same period (same length).

    Returns
    -------
    Renormalised weight vector summing to 1, or zero vector if sum is zero.
    """
    w_new = w_prev * (1.0 + returns_last)
    total = w_new.sum()
    if abs(total) < 1e-12:
        return np.zeros_like(w_prev)
    return w_new / total


# ---------------------------------------------------------------------------
# Main rolling backtest
# ---------------------------------------------------------------------------

def run_rolling_backtest(
    close_df: pd.DataFrame,
    ohlcv_dict: dict,
    sentiment_panel: pd.DataFrame | dict,
    tickers: list[str],
    T1: int,
    T2: int,
    alpha: float,
    spec_name: str,
    results_dir: str = "results",
) -> dict:
    """Execute the DRMV rolling backtest with clustering-based asset selection.

    Parameters
    ----------
    close_df        : date-indexed close price DataFrame (all tickers).
    ohlcv_dict      : {ticker: pd.DataFrame} with OHLCV columns, date-indexed.
    sentiment_panel : sentiment feature data; pd.DataFrame with MultiIndex
                      columns (ticker, feature) or dict {ticker: pd.DataFrame}.
    tickers         : full asset universe (length n).
    T1              : in-sample window length (trading days).
    T2              : out-of-sample (rebalancing) interval (trading days).
    alpha           : hierarchical clustering distance threshold.
    spec_name       : key in CLUSTERING_SPECS, e.g. 'AWDS'.
    results_dir     : directory for persisting per-window artefacts.

    Returns
    -------
    dict with keys:
        'oos_returns'    : pd.Series of concatenated OOS daily portfolio returns.
        'window_records' : list of per-window diagnostic dicts.
        'M'              : total number of rolling windows executed.
    """
    if spec_name not in CLUSTERING_SPECS:
        raise ValueError(f"Unknown spec '{spec_name}'. Valid: {list(CLUSTERING_SPECS)}")

    spec        = CLUSTERING_SPECS[spec_name]
    dist_type   = spec["dist"]
    feature_src = spec["features"]

    # ------------------------------------------------------------------
    # Step 1 – compute technical indicators over the full sample period
    # ------------------------------------------------------------------
    if compute_all_indicators is not None:
        # compute_all_indicators expects dict {ticker: OHLCV DataFrame}
        indicators_dict = compute_all_indicators(ohlcv_dict)
    else:
        logger.warning("compute_all_indicators not available; using empty indicators.")
        indicators_dict = {}

    # ------------------------------------------------------------------
    # Step 2 – build feature panels for the full period
    # ------------------------------------------------------------------
    all_dates = close_df.index
    full_panel = _build_feature_panel(
        indicators_dict, sentiment_panel, tickers, all_dates, kind=feature_src
    )

    # ------------------------------------------------------------------
    # Step 3 – trading calendar and window schedule
    # ------------------------------------------------------------------
    n = len(tickers)
    dates = all_dates
    M = math.floor((len(dates) - T1) / T2)
    logger.info(
        "Rolling backtest: spec=%s  T1=%d  T2=%d  alpha=%.3f  M=%d",
        spec_name, T1, T2, alpha, M
    )

    # ------------------------------------------------------------------
    # Step 5 – initialise state
    # ------------------------------------------------------------------
    prev_w_plus_map:  dict[str, float] = {}
    prev_w_minus_map: dict[str, float] = {}
    prev_selected:    list[str]        = []

    all_oos_returns:  list[pd.Series]  = []
    window_records:   list[dict]       = []

    # Precompute daily returns (pct_change) from close prices
    returns_full = close_df[tickers].pct_change().fillna(0.0)

    # ------------------------------------------------------------------
    # Step 6 – main rolling loop
    # ------------------------------------------------------------------
    for m in range(M):
        is_start  = m * T2
        is_end    = m * T2 + T1
        oos_start = is_end
        oos_end   = min(is_end + T2, len(dates))

        if oos_end <= oos_start:
            logger.debug("Window %d: empty OOS slice, skipping.", m)
            continue

        is_dates  = dates[is_start:is_end]
        oos_dates = dates[oos_start:oos_end]

        returns_is  = returns_full.loc[is_dates]
        returns_oos = returns_full.loc[oos_dates]

        logger.debug(
            "Window %d: IS=%s→%s  OOS=%s→%s",
            m, is_dates[0].date(), is_dates[-1].date(),
            oos_dates[0].date(), oos_dates[-1].date(),
        )

        # -- 6c  feature panel slice (in-sample rows) --------------------
        panel_is = full_panel[is_start:is_end]           # (T1, N, F)

        # -- 6d  standardise using in-sample statistics only -------------
        panel_is_std = _standardize_features(panel_is)

        # -- 6e  distance matrix -----------------------------------------
        try:
            dist_matrix = _compute_distance_matrix(panel_is_std, tickers, dist_type)
        except Exception as exc:
            logger.error("Distance matrix failed for window %d: %s", m, exc)
            dist_matrix = np.zeros((n, n))

        # -- 6f  clustering with alpha threshold -------------------------
        if cluster_assets is not None:
            try:
                labels = np.asarray(cluster_assets(dist_matrix, tickers, alpha))
            except Exception as exc:
                logger.error("Clustering failed for window %d: %s", m, exc)
                labels = np.arange(n)   # fallback: all singletons
        else:
            logger.warning("cluster_assets not available; treating all as singletons.")
            labels = np.arange(n)

        # -- 6g  in-sample Sharpe ratios ---------------------------------
        sharpe_dict = _compute_sharpe(returns_is, tickers)

        # -- 6h  asset selection + retention rule ------------------------
        selected = _select_assets_from_clusters(
            labels, tickers, sharpe_dict, prev_selected
        )
        if not selected:
            logger.warning(
                "Window %d: no non-singleton clusters; carrying over prev_selected.", m
            )
            selected = list(prev_selected) if prev_selected else tickers[:1]

        n_sel = len(selected)
        logger.debug("Window %d: selected %d assets: %s", m, n_sel, selected)

        # -- 6i  tune lambda on in-sample (D1/D2 sub-split) --------------
        w_prev_plus_init_tune  = np.zeros(n_sel)
        w_prev_minus_init_tune = np.zeros(n_sel)
        best_lam = tune_lambda(
            returns_is[selected],
            selected,
            w_prev_plus_init_tune,
            w_prev_minus_init_tune,
        )

        # -- 6j  prepare mark-to-market drifted prior weights ------------
        returns_last = returns_is.iloc[-1]   # last IS day (Series by ticker)
        w_plus_init, w_minus_init = _prepare_drifted_weights(
            selected, prev_selected,
            prev_w_plus_map, prev_w_minus_map,
            returns_last,
        )

        # Compute mean/cov on full in-sample for the selected subset
        r_bar_sel = compute_mean_return(returns_is, selected)
        C_sel     = compute_covariance(returns_is, selected)

        # Solve DRMV on the selected subset
        opt_result = solve_drmv(
            r_bar_sel, C_sel,
            w_plus_init, w_minus_init,
            lam=best_lam,
        )

        w_plus_sol  = opt_result["w_plus"]
        w_minus_sol = opt_result["w_minus"]
        x_sol       = opt_result["x"]

        # -- 6k  OOS portfolio returns ------------------------------------
        oos_port_returns = compute_portfolio_returns(
            returns_oos, selected, x_sol
        )
        all_oos_returns.append(oos_port_returns)

        # -- 6l  record diagnostics ---------------------------------------
        cluster_sizes = {
            int(lbl): int((labels == lbl).sum())
            for lbl in np.unique(labels)
        }
        window_records.append({
            "window":           m,
            "is_start":         str(is_dates[0].date()),
            "is_end":           str(is_dates[-1].date()),
            "oos_start":        str(oos_dates[0].date()),
            "oos_end":          str(oos_dates[-1].date()),
            "spec":             spec_name,
            "alpha":            alpha,
            "T1":               T1,
            "T2":               T2,
            "selected":         list(selected),
            "n_selected":       n_sel,
            "lambda":           best_lam,
            "w_plus":           w_plus_sol.tolist(),
            "w_minus":          w_minus_sol.tolist(),
            "x":                x_sol.tolist(),
            "oos_mean_return":  float(oos_port_returns.mean()),
            "oos_std_return":   float(oos_port_returns.std()),
            "transaction_cost": opt_result["transaction_cost"],
            "opt_status":       opt_result["status"],
            "cluster_sizes":    cluster_sizes,
        })

        # -- update state for next window ---------------------------------
        prev_selected    = list(selected)
        prev_w_plus_map  = {t: float(w_plus_sol[i])  for i, t in enumerate(selected)}
        prev_w_minus_map = {t: float(w_minus_sol[i]) for i, t in enumerate(selected)}

    # ------------------------------------------------------------------
    # Aggregate OOS returns across all windows
    # ------------------------------------------------------------------
    if all_oos_returns:
        oos_series = pd.concat(all_oos_returns).sort_index()
    else:
        oos_series = pd.Series(dtype=float, name="portfolio_return")

    return {
        "oos_returns":    oos_series,
        "window_records": window_records,
        "M":              M,
    }


# ---------------------------------------------------------------------------
# Benchmark strategies
# ---------------------------------------------------------------------------

def run_benchmark_ew(
    returns_df: pd.DataFrame,
    tickers: list[str],
) -> pd.Series:
    """Equal-weight portfolio over the provided tickers, daily rebalanced.

    Naive EW over full selected universe (all non-singleton cluster members
    across all methods).

    Parameters
    ----------
    returns_df : date-indexed returns DataFrame (all dates × ≥ len(tickers) cols).
    tickers    : assets to include; equal weight 1/n each day.

    Returns
    -------
    pd.Series of daily portfolio returns.
    """
    weights = np.ones(len(tickers)) / len(tickers)
    port_returns = returns_df[tickers].dot(weights)
    port_returns.name = "EW"
    return port_returns


def run_benchmark_full_drmv(
    returns_df: pd.DataFrame,
    tickers: list[str],
    T1: int,
    T2: int,
) -> pd.Series:
    """DRMV without clustering: rolling windows using the full ticker universe.

    Uses the same T1/T2 rolling scheme as the main backtest but skips the
    clustering step – all tickers are passed to solve_drmv each window.

    Parameters
    ----------
    returns_df : date-indexed daily returns (full sample).
    tickers    : complete asset universe.
    T1         : in-sample window length.
    T2         : rebalancing interval.

    Returns
    -------
    pd.Series of daily OOS portfolio returns.
    """
    n = len(tickers)
    dates = returns_df.index
    M = math.floor((len(dates) - T1) / T2)

    prev_w_plus  = np.zeros(n)
    prev_w_minus = np.zeros(n)
    all_oos: list[pd.Series] = []

    for m in range(M):
        is_start  = m * T2
        is_end    = m * T2 + T1
        oos_start = is_end
        oos_end   = min(is_end + T2, len(dates))
        if oos_end <= oos_start:
            continue

        is_dates  = dates[is_start:is_end]
        oos_dates = dates[oos_start:oos_end]
        returns_is  = returns_df.loc[is_dates, tickers]
        returns_oos = returns_df.loc[oos_dates, tickers]

        r_bar = compute_mean_return(returns_is, tickers)
        C     = compute_covariance(returns_is, tickers)

        # Drift prior weights by last IS day's returns
        r_last      = returns_is.iloc[-1].values
        prev_w_plus  = compute_mark_to_market_weights(prev_w_plus,  r_last)
        prev_w_minus = compute_mark_to_market_weights(prev_w_minus, r_last)

        best_lam = tune_lambda(returns_is, tickers, prev_w_plus, prev_w_minus)
        result   = solve_drmv(r_bar, C, prev_w_plus, prev_w_minus, lam=best_lam)

        prev_w_plus  = result["w_plus"]
        prev_w_minus = result["w_minus"]

        oos_port = compute_portfolio_returns(returns_oos, tickers, result["x"])
        all_oos.append(oos_port)

    if all_oos:
        series = pd.concat(all_oos).sort_index()
        series.name = "FullDRMV"
        return series
    return pd.Series(dtype=float, name="FullDRMV")


def run_benchmark_bnh(
    close_df: pd.DataFrame,
    selected_tickers: list[str],
    T1: int,
    start_date: str,
    oos_days: int,
) -> pd.Series:
    """Buy-and-hold benchmark: solve DRMV once, then hold unchanged.

    DRMV is solved using the T1 trading days ending at start_date.  The
    resulting portfolio weights are applied to the subsequent oos_days.

    Parameters
    ----------
    close_df          : full date-indexed close price DataFrame.
    selected_tickers  : asset subset to include.
    T1                : in-sample length for the initial DRMV solve.
    start_date        : last date of the in-sample period (str 'YYYY-MM-DD').
    oos_days          : number of trading days to hold the portfolio.

    Returns
    -------
    pd.Series of daily portfolio returns over the holding period.
    """
    returns_full = close_df[selected_tickers].pct_change().fillna(0.0)
    all_dates    = returns_full.index

    # Locate start_date in the index
    if start_date in all_dates:
        end_idx = all_dates.get_loc(start_date)
    else:
        end_idx = all_dates.searchsorted(pd.Timestamp(start_date), side="right") - 1
    end_idx = int(end_idx)

    is_end   = end_idx + 1           # exclusive upper bound
    is_start = max(0, is_end - T1)

    is_dates = all_dates[is_start:is_end]
    returns_is = returns_full.loc[is_dates]

    n  = len(selected_tickers)
    r_bar = compute_mean_return(returns_is, selected_tickers)
    C     = compute_covariance(returns_is, selected_tickers)

    result = solve_drmv(
        r_bar, C,
        np.zeros(n), np.zeros(n),
        lam=0.5,
    )

    oos_start_idx = is_end
    oos_end_idx   = min(oos_start_idx + oos_days, len(all_dates))
    oos_dates     = all_dates[oos_start_idx:oos_end_idx]
    returns_oos   = returns_full.loc[oos_dates]

    port_returns  = compute_portfolio_returns(
        returns_oos, selected_tickers, result["x"]
    )
    port_returns.name = "BnH"
    return port_returns


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s %(name)s: %(message)s")

    rng = np.random.default_rng(0)
    N, T_FULL = 8, 300
    TICKERS = [f"S{i}" for i in range(N)]

    # Synthetic close prices
    prices = pd.DataFrame(
        np.cumprod(1.0 + rng.normal(0.0004, 0.012, (T_FULL, N)), axis=0),
        index=pd.bdate_range("2018-01-01", periods=T_FULL),
        columns=TICKERS,
    )
    returns_df = prices.pct_change().fillna(0.0)

    # --- EW benchmark ---
    ew_rets = run_benchmark_ew(returns_df, TICKERS)
    print("EW benchmark:")
    print(f"  annualised return : {ew_rets.mean()*252:.4f}")
    print(f"  annualised vol    : {ew_rets.std()*252**0.5:.4f}")

    # --- compute_mark_to_market_weights ---
    w0  = np.array([0.5, 0.3, 0.2])
    r0  = np.array([0.01, -0.02, 0.05])
    w1  = compute_mark_to_market_weights(w0, r0)
    print(f"\nMark-to-market weights: {np.round(w1, 4)}  (sum={w1.sum():.4f})")

    # --- Full-DRMV benchmark (small problem for speed) ---
    T1, T2 = 60, 20
    sub_tickers = TICKERS[:4]
    full_rets = run_benchmark_full_drmv(returns_df[sub_tickers], sub_tickers, T1, T2)
    print(f"\nFullDRMV benchmark ({len(full_rets)} OOS days):")
    if len(full_rets) > 0 and full_rets.std() > 0:
        sharpe = full_rets.mean() / full_rets.std() * (252 ** 0.5)
        print(f"  Sharpe (annualised): {sharpe:.4f}")

    # --- BnH benchmark ---
    start = prices.index[T1 - 1].strftime("%Y-%m-%d")
    bnh_rets = run_benchmark_bnh(prices[sub_tickers], sub_tickers, T1, start, T2)
    print(f"\nBnH benchmark ({len(bnh_rets)} OOS days):")
    if len(bnh_rets) > 0 and bnh_rets.std() > 0:
        sharpe_bnh = bnh_rets.mean() / bnh_rets.std() * (252 ** 0.5)
        print(f"  Sharpe (annualised): {sharpe_bnh:.4f}")

    # --- run_rolling_backtest stub (clustering modules not yet available) ---
    print("\nrun_rolling_backtest: skipped (clustering modules not yet available).")
