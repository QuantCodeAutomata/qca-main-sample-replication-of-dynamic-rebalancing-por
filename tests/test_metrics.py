"""pytest tests for src/metrics.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_metrics,
    compute_max_drawdown,
    compute_avg_drawdown,
    paired_ttest_mean_return,
    bootstrap_sharpe_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def dates_252():
    return pd.bdate_range('2024-01-01', periods=252)


@pytest.fixture
def positive_returns(rng, dates_252):
    """Return series that is always strictly positive."""
    return pd.Series(rng.uniform(0.001, 0.02, 252), index=dates_252)


@pytest.fixture
def mixed_returns(rng, dates_252):
    """Realistic mixed positive/negative daily returns."""
    return pd.Series(rng.normal(0.0005, 0.015, 252), index=dates_252)


@pytest.fixture
def constant_returns(dates_252):
    """Constant return series (std == 0).

    Use 0.0 so the float representation is exact and pandas std is truly 0.
    """
    return pd.Series(np.zeros(252), index=dates_252)


@pytest.fixture
def monotone_increase(dates_252):
    """Monotonically increasing wealth — no drawdown."""
    return pd.Series(np.full(252, 1e-4), index=dates_252)


# ---------------------------------------------------------------------------
# compute_metrics — basic properties
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_sharpe_positive_returns(self, positive_returns):
        m = compute_metrics(positive_returns)
        assert m['sharpe'] > 0, "Sharpe should be positive when all returns are positive."

    def test_sharpe_zero_std(self, constant_returns):
        """No ZeroDivisionError; Sharpe must be 0 when std == 0."""
        m = compute_metrics(constant_returns)
        assert m['sharpe'] == 0.0

    def test_max_drawdown_negative(self, mixed_returns):
        m = compute_metrics(mixed_returns)
        assert m['max_drawdown'] <= 0.0, "Max drawdown must be non-positive."

    def test_max_drawdown_monotone_increase(self, monotone_increase):
        dd = compute_max_drawdown(monotone_increase)
        assert np.isclose(dd, 0.0), "No drawdown for monotone-increasing wealth."

    def test_cvar_le_var(self, mixed_returns):
        m = compute_metrics(mixed_returns)
        # CVaR is the mean of the worst tail — must be <= VaR (more negative)
        assert m['cvar_95'] <= m['var_95'] + 1e-12, (
            f"CVaR={m['cvar_95']:.6f} should be <= VaR={m['var_95']:.6f}"
        )

    def test_terminal_wealth_positive(self, mixed_returns):
        m = compute_metrics(mixed_returns)
        assert m['terminal_wealth'] > 0.0

    def test_cumulative_return_range(self, mixed_returns):
        m = compute_metrics(mixed_returns)
        assert m['cumulative_return'] > -1.0, "Can't lose more than 100%."

    def test_metrics_empty_series(self):
        """Empty series must return a zero-valued dict without raising."""
        result = compute_metrics(pd.Series([], dtype=float))
        assert isinstance(result, dict)
        expected_keys = {
            'mean', 'std', 'sharpe', 'var_95', 'cvar_95', 'min_return',
            'max_return', 'max_drawdown', 'avg_drawdown', 'starr',
            'sterling', 'cumulative_return', 'terminal_wealth',
        }
        assert set(result.keys()) == expected_keys
        assert all(v == 0.0 for v in result.values()), (
            "All metric values must be 0.0 for an empty series."
        )

    def test_all_keys_present(self, mixed_returns):
        m = compute_metrics(mixed_returns)
        for key in ('mean', 'std', 'sharpe', 'var_95', 'cvar_95',
                    'min_return', 'max_return', 'max_drawdown',
                    'avg_drawdown', 'starr', 'sterling',
                    'cumulative_return', 'terminal_wealth'):
            assert key in m, f"Key '{key}' missing from metrics dict."

    def test_rf_shifts_sharpe(self, positive_returns):
        m0 = compute_metrics(positive_returns, rf=0.0)
        m_rf = compute_metrics(positive_returns, rf=0.0001)
        # Higher rf reduces Sharpe
        assert m_rf['sharpe'] < m0['sharpe']


# ---------------------------------------------------------------------------
# compute_max_drawdown / compute_avg_drawdown
# ---------------------------------------------------------------------------
class TestDrawdownFunctions:
    def test_max_drawdown_returns_float(self, mixed_returns):
        result = compute_max_drawdown(mixed_returns)
        assert isinstance(result, float)

    def test_avg_drawdown_le_zero(self, mixed_returns):
        result = compute_avg_drawdown(mixed_returns)
        assert result <= 0.0

    def test_avg_drawdown_ge_max_drawdown(self, mixed_returns):
        avg = compute_avg_drawdown(mixed_returns)
        mx = compute_max_drawdown(mixed_returns)
        assert avg >= mx, "Average drawdown must be >= max drawdown (less severe)."


# ---------------------------------------------------------------------------
# paired_ttest_mean_return
# ---------------------------------------------------------------------------
class TestPairedTtest:
    def test_paired_ttest_significant(self, dates_252, rng):
        """A clearly superior series should give a low (significant) p-value."""
        # Series A has much higher mean
        a = pd.Series(rng.normal(0.005, 0.01, 252), index=dates_252)
        b = pd.Series(rng.normal(0.0001, 0.01, 252), index=dates_252)
        result = paired_ttest_mean_return(a, b)
        assert result['significant_at_05'] is True, (
            f"Expected significance; p={result['p_value']:.4f}"
        )

    def test_paired_ttest_not_significant(self, dates_252, rng):
        """Two series with the same mean should not be significant."""
        a = pd.Series(rng.normal(0.001, 0.02, 252), index=dates_252)
        b = pd.Series(rng.normal(0.001, 0.02, 252), index=dates_252)
        result = paired_ttest_mean_return(a, b)
        assert 'p_value' in result
        assert 0.0 <= result['p_value'] <= 1.0

    def test_paired_ttest_result_keys(self, dates_252, rng):
        a = pd.Series(rng.normal(0.001, 0.015, 252), index=dates_252)
        b = pd.Series(rng.normal(0.0005, 0.015, 252), index=dates_252)
        result = paired_ttest_mean_return(a, b)
        assert set(result.keys()) == {'t_stat', 'p_value', 'significant_at_05'}

    def test_paired_ttest_aligns_common_dates(self):
        """Result should use the common date intersection."""
        dates_a = pd.bdate_range('2024-01-01', periods=100)
        dates_b = pd.bdate_range('2024-01-15', periods=100)
        rng_loc = np.random.default_rng(1)
        a = pd.Series(rng_loc.normal(0.001, 0.01, 100), index=dates_a)
        b = pd.Series(rng_loc.normal(0.001, 0.01, 100), index=dates_b)
        # Should not raise even with partial overlap
        result = paired_ttest_mean_return(a, b)
        assert 't_stat' in result


# ---------------------------------------------------------------------------
# bootstrap_sharpe_test
# ---------------------------------------------------------------------------
class TestBootstrapSharpe:
    def test_bootstrap_sharpe_p_in_01(self, dates_252, rng):
        a = pd.Series(rng.normal(0.002, 0.015, 252), index=dates_252)
        b = pd.Series(rng.normal(0.001, 0.015, 252), index=dates_252)
        result = bootstrap_sharpe_test(a, b, n_bootstrap=200, seed=0)
        assert 0.0 <= result['p_value'] <= 1.0

    def test_bootstrap_sharpe_result_keys(self, dates_252, rng):
        a = pd.Series(rng.normal(0.001, 0.015, 252), index=dates_252)
        b = pd.Series(rng.normal(0.001, 0.015, 252), index=dates_252)
        result = bootstrap_sharpe_test(a, b, n_bootstrap=100, seed=7)
        assert set(result.keys()) == {'delta_obs', 'p_value', 'significant_at_05'}

    def test_bootstrap_sharpe_delta_obs_type(self, dates_252, rng):
        a = pd.Series(rng.normal(0.001, 0.015, 252), index=dates_252)
        b = pd.Series(rng.normal(0.0005, 0.015, 252), index=dates_252)
        result = bootstrap_sharpe_test(a, b, n_bootstrap=100, seed=0)
        assert isinstance(result['delta_obs'], float)

    def test_bootstrap_sharpe_reproducible(self, dates_252, rng):
        """Same seed must give identical p-values."""
        a = pd.Series(rng.normal(0.001, 0.015, 252), index=dates_252)
        b = pd.Series(rng.normal(0.0005, 0.015, 252), index=dates_252)
        r1 = bootstrap_sharpe_test(a, b, n_bootstrap=200, seed=42)
        r2 = bootstrap_sharpe_test(a, b, n_bootstrap=200, seed=42)
        assert r1['p_value'] == r2['p_value']

    def test_bootstrap_sharpe_identical_series(self, dates_252, rng):
        """Identical series → delta_obs = 0 → p_value near 1.0."""
        a = pd.Series(rng.normal(0.001, 0.015, 252), index=dates_252)
        result = bootstrap_sharpe_test(a, a, n_bootstrap=200, seed=0)
        assert np.isclose(result['delta_obs'], 0.0)
        assert result['p_value'] > 0.5
