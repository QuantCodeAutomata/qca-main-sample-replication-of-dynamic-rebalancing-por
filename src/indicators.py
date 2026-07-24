"""
Technical indicators used as input features for the TDA-based portfolio selection model.

All indicators are computed per-asset on daily OHLCV data and then assembled into
feature panels consumed by the distance and clustering modules.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, L: int = 10) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using a simple rolling mean.

    Paper formula: RSI_t = 100 - 100 / (1 + RS_t)
    where RS_t = avg_gain_L / avg_loss_L and averages are plain rolling means.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    L : int, optional
        Look-back window length.
        L=10 per paper appendix (narrative suggests 14, appendix references L=10;
        using appendix).

    Returns
    -------
    pd.Series
        RSI values in [0, 100], aligned to the index of *close*.
    """
    delta = close.diff()

    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = gains.rolling(L).mean()
    avg_loss = losses.rolling(L).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # When avg_loss == 0 the asset has only gone up: RSI is defined as 100.
    rsi = rsi.where(avg_loss != 0.0, other=100.0)

    rsi.name = "RSI"
    return rsi


def compute_stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Compute the Stochastic Oscillator using the same-day high/low/close.

    Paper formula (same-day, no rolling lookback):
        SO_t = (C_t - L_t) / (H_t - L_t) * 100

    Using same-day formula per paper appendix; no rolling lookback window.

    Parameters
    ----------
    high, low, close : pd.Series
        Daily high, low, and closing prices sharing the same index.

    Returns
    -------
    pd.Series
        SO values in [0, 100]. When H_t == L_t the midpoint 50.0 is returned.
    """
    denominator = high - low
    numerator = close - low

    so = numerator / denominator.replace(0.0, np.nan) * 100.0
    so = so.where(denominator != 0.0, other=50.0)

    so.name = "SO"
    return so


def compute_macd(
    close: pd.Series,
    short: int = 12,
    long_: int = 26,
    signal: int = 9,
) -> pd.Series:
    """Compute MACD histogram (MACD line minus signal/DEA line).

    MACD parameters: short=12, long=26, signal=9 per paper.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    short, long_, signal : int
        EMA span parameters for the fast EMA, slow EMA, and DEA signal line.

    Returns
    -------
    pd.Series
        MACD histogram values (MACD_line - DEA).
    """
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long_, adjust=False).mean()

    macd_line = ema_short - ema_long
    dea = macd_line.ewm(span=signal, adjust=False).mean()

    macd = macd_line - dea
    macd.name = "MACD"
    return macd


# ---------------------------------------------------------------------------
# Panel-level aggregation
# ---------------------------------------------------------------------------

def compute_all_indicators(
    ohlcv_panel: dict | pd.DataFrame, L_rsi: int = 10
) -> dict:
    """Compute RSI, SO, and MACD for every ticker in the panel.

    Parameters
    ----------
    ohlcv_panel : dict or pd.DataFrame
        Either a dict ``{ticker: pd.DataFrame}`` where each DataFrame has columns
        ``['Open', 'High', 'Low', 'Close', 'Volume']`` (case-insensitive), or a
        wide MultiIndex DataFrame with tickers on columns level 0.
    L_rsi : int
        RSI look-back window forwarded to :func:`compute_rsi`.

    Returns
    -------
    dict
        ``{ticker: pd.DataFrame}`` where each DataFrame has columns
        ``['RSI', 'SO', 'MACD']`` with the original date index.
    """
    if isinstance(ohlcv_panel, pd.DataFrame):
        # MultiIndex columns: level 0 = field, level 1 = ticker  OR
        #                     level 0 = ticker, level 1 = field
        # We normalise to a dict of per-ticker DataFrames.
        if isinstance(ohlcv_panel.columns, pd.MultiIndex):
            level0 = [c.lower() if isinstance(c, str) else c
                      for c in ohlcv_panel.columns.get_level_values(0).unique()]
            ohlcv_like = {"high", "low", "close", "open", "volume"}
            if set(level0) & ohlcv_like:
                # (field, ticker) orientation
                ohlcv_panel = {
                    ticker: ohlcv_panel.xs(ticker, axis=1, level=1)
                    for ticker in ohlcv_panel.columns.get_level_values(1).unique()
                }
            else:
                # (ticker, field) orientation
                ohlcv_panel = {
                    ticker: ohlcv_panel[ticker]
                    for ticker in ohlcv_panel.columns.get_level_values(0).unique()
                }
        else:
            raise ValueError(
                "DataFrame ohlcv_panel must have MultiIndex columns. "
                "Pass a dict {ticker: DataFrame} instead."
            )

    result: dict[str, pd.DataFrame] = {}
    for ticker, df in ohlcv_panel.items():
        cols = {c.lower(): c for c in df.columns}
        try:
            high = df[cols["high"]]
            low = df[cols["low"]]
            close = df[cols["close"]]
        except KeyError as exc:
            logger.warning("Ticker %s missing OHLC columns: %s – skipped.", ticker, exc)
            continue

        rsi = compute_rsi(close, L=L_rsi)
        so = compute_stochastic_oscillator(high, low, close)
        macd = compute_macd(close)

        result[ticker] = pd.DataFrame({"RSI": rsi, "SO": so, "MACD": macd},
                                      index=df.index)

    return result


# ---------------------------------------------------------------------------
# 3-D and 4-D feature panel builders
# ---------------------------------------------------------------------------

def build_feature_panel_3d(
    indicators: dict,
    tickers: list[str],
    dates: pd.DatetimeIndex,
) -> dict:
    """Align per-ticker indicator DataFrames to a common date grid.

    Parameters
    ----------
    indicators : dict
        Output of :func:`compute_all_indicators`.
    tickers : list[str]
        Ordered list of tickers to include.
    dates : pd.DatetimeIndex
        Target date grid (rows).

    Returns
    -------
    dict
        ``{ticker: np.ndarray}`` of shape ``(T, 3)`` with column order
        ``[RSI, SO, MACD]``.
    """
    panel: dict[str, np.ndarray] = {}
    for ticker in tickers:
        df = indicators[ticker].reindex(dates)
        panel[ticker] = df[["RSI", "SO", "MACD"]].to_numpy(dtype=float)
    return panel


def build_feature_panel_4d(
    indicators: dict,
    sentiment_panel: pd.DataFrame,
    tickers: list[str],
    dates: pd.DatetimeIndex,
) -> dict:
    """Extend the 3-D panel with a 4th sentiment feature.

    Parameters
    ----------
    indicators : dict
        Output of :func:`compute_all_indicators`.
    sentiment_panel : pd.DataFrame
        Shape ``(T_full, n_tickers)`` with tickers as columns and dates as index.
    tickers : list[str]
        Ordered list of tickers to include.
    dates : pd.DatetimeIndex
        Target date grid.

    Returns
    -------
    dict
        ``{ticker: np.ndarray}`` of shape ``(T, 4)`` with column order
        ``[RSI, SO, MACD, sentiment]``.
    """
    panel_3d = build_feature_panel_3d(indicators, tickers, dates)
    panel: dict[str, np.ndarray] = {}
    for ticker in tickers:
        tech = panel_3d[ticker]                              # (T, 3)
        if ticker in sentiment_panel.columns:
            sent = sentiment_panel[ticker].reindex(dates).to_numpy(dtype=float)
        else:
            logger.warning("No sentiment data for %s; filling with NaN.", ticker)
            sent = np.full(len(dates), np.nan)
        panel[ticker] = np.column_stack([tech, sent])       # (T, 4)
    return panel


# ---------------------------------------------------------------------------
# Z-score standardisation (in-sample fit, full-sample transform)
# ---------------------------------------------------------------------------

def standardize_features(
    feature_dict: dict,
    fit_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
) -> dict:
    """Z-score normalise feature arrays using in-sample statistics only.

    z-scoring with in-sample statistics only to avoid look-ahead leakage.

    Parameters
    ----------
    feature_dict : dict
        ``{ticker: np.ndarray}`` of shape ``(T_all, d)`` aligned to *all_dates*.
    fit_dates : pd.DatetimeIndex
        In-sample date range used to compute mean and std.
    all_dates : pd.DatetimeIndex
        Full date range of the arrays stored in *feature_dict*.

    Returns
    -------
    dict
        ``{ticker: np.ndarray}`` normalised arrays of the same shape.
    """
    # Build integer index mask for fit_dates within all_dates
    all_dates_idx = pd.Index(all_dates)
    fit_mask = all_dates_idx.isin(fit_dates)

    normalised: dict[str, np.ndarray] = {}
    for ticker, arr in feature_dict.items():
        in_sample = arr[fit_mask]                           # (T_fit, d)
        mu = np.nanmean(in_sample, axis=0)                  # (d,)
        sigma = np.nanstd(in_sample, axis=0)                # (d,)

        # Avoid division by zero when a feature has zero variance in-sample
        sigma = np.where(sigma == 0.0, 1.0, sigma)

        normalised[ticker] = (arr - mu) / sigma
    return normalised


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    T = 120
    dates = pd.date_range("2020-01-02", periods=T, freq="B")

    tickers = ["AAPL", "MSFT", "GOOG"]
    ohlcv: dict[str, pd.DataFrame] = {}
    for t in tickers:
        price = 100 + np.cumsum(rng.normal(0, 1, T))
        high = price + rng.uniform(0, 2, T)
        low = price - rng.uniform(0, 2, T)
        ohlcv[t] = pd.DataFrame(
            {"Open": price, "High": high, "Low": low, "Close": price,
             "Volume": rng.integers(1_000, 10_000, T)},
            index=dates,
        )

    indicators = compute_all_indicators(ohlcv)
    for ticker, df in indicators.items():
        logger.info("%s indicators shape: %s, first valid row: %s",
                    ticker, df.shape, df.dropna().index[0])

    panel_3d = build_feature_panel_3d(indicators, tickers, dates)
    for ticker, arr in panel_3d.items():
        assert arr.shape == (T, 3), f"Expected ({T}, 3), got {arr.shape}"
    logger.info("3D panel shapes OK.")

    sentiment = pd.DataFrame(
        rng.standard_normal((T, len(tickers))), index=dates, columns=tickers
    )
    panel_4d = build_feature_panel_4d(indicators, sentiment, tickers, dates)
    for ticker, arr in panel_4d.items():
        assert arr.shape == (T, 4), f"Expected ({T}, 4), got {arr.shape}"
    logger.info("4D panel shapes OK.")

    fit_dates = dates[:80]
    normed = standardize_features(panel_3d, fit_dates, dates)
    for ticker, arr in normed.items():
        assert arr.shape == (T, 3)
    logger.info("Standardisation OK.")
    logger.info("indicators.py self-test passed.")
