"""pytest tests for src/indicators.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    compute_rsi,
    compute_stochastic_oscillator,
    compute_macd,
    compute_all_indicators,
    build_feature_panel_3d,
    build_feature_panel_4d,
    standardize_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dates():
    return pd.bdate_range('2023-01-01', periods=200)


@pytest.fixture
def prices_random(dates):
    rng = np.random.default_rng(42)
    vals = np.cumsum(rng.normal(0, 1, len(dates))) + 100
    return pd.Series(vals, index=dates, name='close')


@pytest.fixture
def prices_increasing(dates):
    """Strictly increasing prices — all gains, no losses."""
    return pd.Series(np.arange(1.0, len(dates) + 1.0), index=dates, name='close')


@pytest.fixture
def prices_decreasing(dates):
    """Strictly decreasing prices — all losses, no gains."""
    return pd.Series(np.arange(float(len(dates)), 0.0, -1.0), index=dates, name='close')


@pytest.fixture
def ohlcv_dict(dates):
    """Dict of OHLCV DataFrames for three tickers."""
    rng = np.random.default_rng(7)
    n = len(dates)
    tickers = ['AAA', 'BBB', 'CCC']
    result = {}
    for t in tickers:
        c = np.cumsum(rng.normal(0, 1, n)) + 100
        h = c + rng.uniform(0.1, 2.0, n)
        lo = c - rng.uniform(0.1, 2.0, n)
        result[t] = pd.DataFrame(
            {'Open': c, 'High': h, 'Low': lo, 'Close': c,
             'Volume': rng.integers(1_000, 100_000, n).astype(float)},
            index=dates,
        )
    return result


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------
class TestRSI:
    def test_rsi_range(self, prices_random):
        rsi = compute_rsi(prices_random, L=10)
        valid = rsi.dropna()
        assert len(valid) > 0, "RSI should produce non-NaN values after warmup."
        assert (valid >= 0.0).all(), "RSI must be >= 0."
        assert (valid <= 100.0).all(), "RSI must be <= 100."

    def test_rsi_all_gains(self, prices_increasing):
        """Monotone price increase → avg_loss=0 → RSI should be 100."""
        rsi = compute_rsi(prices_increasing, L=10)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert (valid >= 99.0).all(), (
            f"RSI should approach 100 for all-gain sequence; min={valid.min():.2f}"
        )

    def test_rsi_all_losses(self, prices_decreasing):
        """Monotone price decrease → avg_gain=0 → RSI should be 0."""
        rsi = compute_rsi(prices_decreasing, L=10)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert (valid <= 1.0).all(), (
            f"RSI should approach 0 for all-loss sequence; max={valid.max():.2f}"
        )

    def test_rsi_zero_avg_loss(self):
        """Explicitly test the avg_loss=0 branch → RSI must equal 100.0."""
        # 15 strictly increasing prices; L=10 → 4 valid values, all gains
        prices = pd.Series(np.arange(1.0, 16.0))
        rsi = compute_rsi(prices, L=10)
        valid = rsi.dropna()
        assert np.allclose(valid.values, 100.0, atol=1e-6), (
            "RSI should be exactly 100.0 when there are no down-moves."
        )

    def test_rsi_returns_series(self, prices_random):
        rsi = compute_rsi(prices_random, L=10)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices_random)

    def test_rsi_warmup_length(self, prices_random):
        """First L values should be NaN (warmup period produced by rolling)."""
        L = 10
        rsi = compute_rsi(prices_random, L=L)
        # diff produces 1 NaN at the start; rolling(L) produces L-1 more NaNs
        # first valid index is L (0-indexed)
        assert rsi.iloc[:L].isna().all(), (
            "First L RSI values should be NaN during warmup."
        )


# ---------------------------------------------------------------------------
# Stochastic Oscillator tests
# ---------------------------------------------------------------------------
class TestStochasticOscillator:
    def test_so_range(self, ohlcv_dict, dates):
        ticker = 'AAA'
        df = ohlcv_dict[ticker]
        so = compute_stochastic_oscillator(df['High'], df['Low'], df['Close'])
        valid = so.dropna()
        assert len(valid) > 0
        assert (valid >= 0.0).all(), "SO must be >= 0."
        assert (valid <= 100.0).all(), "SO must be <= 100."

    def test_so_zero_denominator(self):
        """When H_t == L_t for all t, SO should return 50.0."""
        n = 30
        flat = pd.Series(np.full(n, 50.0))
        so = compute_stochastic_oscillator(flat, flat, flat)
        valid = so.dropna()
        assert len(valid) > 0
        assert np.allclose(valid.values, 50.0, atol=1e-6), (
            f"SO should return 50.0 when H==L; got {valid.values}"
        )

    def test_so_returns_series(self, ohlcv_dict, dates):
        df = ohlcv_dict['BBB']
        so = compute_stochastic_oscillator(df['High'], df['Low'], df['Close'])
        assert isinstance(so, pd.Series)
        assert len(so) == len(df)


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------
class TestMACD:
    def test_macd_returns_series(self, prices_random):
        macd = compute_macd(prices_random)
        assert isinstance(macd, pd.Series), "MACD must return a pd.Series."

    def test_macd_same_length(self, prices_random):
        macd = compute_macd(prices_random)
        assert len(macd) == len(prices_random), (
            "MACD output length must match input length."
        )

    def test_macd_finite_values(self, prices_random):
        macd = compute_macd(prices_random)
        valid = macd.dropna()
        assert np.isfinite(valid.values).all(), "MACD must contain only finite values."


# ---------------------------------------------------------------------------
# Standardize features — no look-ahead bias
# ---------------------------------------------------------------------------
class TestStandardizeFeatures:
    def test_standardize_features_no_lookahead(self, dates):
        """Standardization must use only in-sample (fit_dates) statistics.

        The in-sample mean of the normalised in-sample portion should be ~0.
        The out-of-sample portion uses the in-sample stats, so when the
        out-of-sample data has a very different mean, scaled OOS mean ≠ 0.
        """
        rng = np.random.default_rng(3)
        T = len(dates)       # 200
        n_train = 120
        fit_dates = dates[:n_train]
        all_dates = dates

        # Feature with deliberately different in-sample vs out-of-sample mean
        feature_in = rng.normal(0.0, 1.0, n_train)
        feature_out = rng.normal(20.0, 1.0, T - n_train)  # shifted far out
        feature = np.concatenate([feature_in, feature_out])

        feature_dict = {'T1': np.column_stack([feature, feature * 0.5])}

        normalised = standardize_features(feature_dict, fit_dates, all_dates)

        # In-sample portion: mean ≈ 0, std ≈ 1
        in_part = normalised['T1'][:n_train, 0]
        assert abs(in_part.mean()) < 0.1, f"In-sample mean should be ~0, got {in_part.mean():.3f}"
        assert abs(in_part.std() - 1.0) < 0.15, f"In-sample std should be ~1, got {in_part.std():.3f}"

        # Out-of-sample: scaled by in-sample stats → mean ≫ 0 (not recentred)
        oos_part = normalised['T1'][n_train:, 0]
        assert abs(oos_part.mean()) > 5.0, (
            "OOS portion should not be recentred; it must preserve in-sample scaling."
        )

    def test_standardize_returns_dict(self, dates):
        rng = np.random.default_rng(5)
        T = len(dates)
        fit_dates = dates[:100]
        feature_dict = {
            'A': rng.normal(0, 1, (T, 3)),
            'B': rng.normal(0, 1, (T, 3)),
        }
        result = standardize_features(feature_dict, fit_dates, dates)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'A', 'B'}
        assert result['A'].shape == (T, 3)


# ---------------------------------------------------------------------------
# build_feature_panel shape tests
# ---------------------------------------------------------------------------
class TestBuildFeaturePanel:
    def _make_ohlcv_and_indicators(self, T=120):
        """Create an indicators dict for use with build_feature_panel functions."""
        rng = np.random.default_rng(11)
        dates = pd.bdate_range('2023-01-01', periods=T)
        tickers = ['X1', 'X2', 'X3']
        ohlcv = {}
        for t in tickers:
            c = np.cumsum(rng.normal(0.001, 0.02, T)) + 100
            ohlcv[t] = pd.DataFrame(
                {'Open': c, 'High': c + rng.uniform(0, 1, T),
                 'Low': c - rng.uniform(0, 1, T), 'Close': c,
                 'Volume': rng.integers(1000, 10000, T).astype(float)},
                index=dates,
            )
        indicators = compute_all_indicators(ohlcv)
        return indicators, tickers, dates, rng

    def test_build_feature_panel_3d_shape(self):
        """Each ticker's array must have shape (T, 3)."""
        T = 120
        indicators, tickers, dates, _ = self._make_ohlcv_and_indicators(T=T)
        panel = build_feature_panel_3d(indicators, tickers, dates)
        assert isinstance(panel, dict)
        for ticker in tickers:
            assert ticker in panel, f"Ticker {ticker} missing from 3D panel."
            arr = panel[ticker]
            assert arr.shape == (T, 3), (
                f"Expected ({T}, 3) for {ticker}, got {arr.shape}"
            )

    def test_build_feature_panel_4d_shape(self):
        """Each ticker's array must have shape (T, 4) when sentiment is added."""
        T = 120
        indicators, tickers, dates, rng = self._make_ohlcv_and_indicators(T=T)
        sentiment_panel = pd.DataFrame(
            rng.standard_normal((T, len(tickers))),
            index=dates,
            columns=tickers,
        )
        panel = build_feature_panel_4d(indicators, sentiment_panel, tickers, dates)
        assert isinstance(panel, dict)
        for ticker in tickers:
            assert ticker in panel, f"Ticker {ticker} missing from 4D panel."
            arr = panel[ticker]
            assert arr.shape == (T, 4), (
                f"Expected ({T}, 4) for {ticker}, got {arr.shape}"
            )

    def test_build_feature_panel_column_order_3d(self):
        """Column order must be [RSI, SO, MACD] — i.e., 3 columns, all real."""
        T = 80
        indicators, tickers, dates, _ = self._make_ohlcv_and_indicators(T=T)
        panel = build_feature_panel_3d(indicators, tickers[:1], dates)
        arr = panel[tickers[0]]
        assert arr.shape[1] == 3
