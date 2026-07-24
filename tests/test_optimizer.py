"""pytest tests for src/optimizer.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    solve_drmv,
    tune_lambda,
    compute_mean_return,
    compute_covariance,
    LAMBDA_CANDIDATES,
    M_PLUS,
    m_PLUS,
    K_MARGIN,
    P1, P2, P3, P4,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_setup():
    """5-asset, 120-day synthetic problem (r_bar, C, w0_plus, w0_minus, tickers, rets).

    Use 120 days and a pronounced positive drift so that the D1/D2 split
    produces enough D2 data (24 rows) for the optimizer to yield non-trivial
    portfolios and differentiate lambda candidates.
    """
    rng = np.random.default_rng(42)
    n, T = 5, 120
    tickers = [f'A{i}' for i in range(n)]
    dates = pd.bdate_range('2023-01-01', periods=T)
    prices = pd.DataFrame(
        np.cumprod(1 + rng.normal(0.0008, 0.01, (T, n)), axis=0),
        columns=tickers, index=dates,
    )
    rets = prices.pct_change().dropna()
    r_bar = compute_mean_return(rets, tickers)
    C = compute_covariance(rets, tickers)
    w0 = np.zeros(n)
    return r_bar, C, w0, w0, tickers, rets


@pytest.fixture
def medium_setup():
    """10-asset, 200-day synthetic problem."""
    rng = np.random.default_rng(7)
    n, T = 10, 200
    tickers = [f'B{i}' for i in range(n)]
    dates = pd.bdate_range('2023-01-01', periods=T)
    prices = pd.DataFrame(
        np.cumprod(1 + rng.normal(0.0005, 0.012, (T, n)), axis=0),
        columns=tickers, index=dates,
    )
    rets = prices.pct_change().dropna()
    r_bar = compute_mean_return(rets, tickers)
    C = compute_covariance(rets, tickers)
    w0 = np.zeros(n)
    return r_bar, C, w0, w0, tickers, rets


@pytest.fixture
def lam():
    return 0.6   # mid-range from LAMBDA_CANDIDATES


# ---------------------------------------------------------------------------
# solve_drmv — return type & shape
# ---------------------------------------------------------------------------
class TestSolverReturnType:
    def test_solver_returns_weights(self, small_setup, lam):
        """solve_drmv must return a dict with 'w_plus' and 'w_minus' keys."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        assert isinstance(result, dict), "solve_drmv must return a dict."
        assert 'w_plus' in result, "Result dict must contain 'w_plus'."
        assert 'w_minus' in result, "Result dict must contain 'w_minus'."

    def test_solver_weight_shapes(self, small_setup, lam):
        """w_plus and w_minus must have length n (number of assets)."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        n = len(r_bar)
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        assert result['w_plus'].shape == (n,), (
            f"Expected w_plus shape ({n},), got {result['w_plus'].shape}."
        )
        assert result['w_minus'].shape == (n,), (
            f"Expected w_minus shape ({n},), got {result['w_minus'].shape}."
        )


# ---------------------------------------------------------------------------
# solve_drmv — constraint verification
# ---------------------------------------------------------------------------
class TestWeightConstraints:
    def test_weights_nonneg(self, small_setup, lam):
        """w_plus >= 0, w_minus >= 0 element-wise."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        assert (result['w_plus'] >= -1e-6).all(), "w_plus must be non-negative."
        assert (result['w_minus'] >= -1e-6).all(), "w_minus must be non-negative."

    def test_budget_constraint(self, small_setup, lam):
        """The full budget constraint (including margin + transaction costs) must be ≈ 1."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        budget = (
            result['w_plus'].sum()
            + K_MARGIN * result['w_minus'].sum()
            + P1 * result['l_plus'].sum()
            + P2 * result['l_minus'].sum()
            + P3 * result['s_plus'].sum()
            + P4 * result['s_minus'].sum()
        )
        assert np.isclose(budget, 1.0, atol=1e-3), (
            f"Budget constraint must be ≈ 1.0; got {budget:.6f}."
        )

    def test_no_simultaneous_long_short(self, small_setup, lam):
        """For each asset i, not both w_plus[i] > 0 and w_minus[i] > 0."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        simultaneous = (result['w_plus'] > 1e-4) & (result['w_minus'] > 1e-4)
        assert not simultaneous.any(), (
            "No asset may simultaneously hold a long and short position."
        )

    def test_min_long_weight(self, small_setup, lam):
        """Active long positions (w_plus[i] > threshold) must be >= m_PLUS."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        w_p = result['w_plus']
        active = w_p > 1e-4
        if active.any():
            assert (w_p[active] >= m_PLUS - 1e-3).all(), (
                f"Active long weights must be >= m_PLUS={m_PLUS}; "
                f"min active = {w_p[active].min():.5f}."
            )

    def test_max_long_weight(self, small_setup, lam):
        """All long weights must be <= M_PLUS."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        assert (result['w_plus'] <= M_PLUS + 1e-3).all(), (
            f"All long weights must be <= M_PLUS={M_PLUS}; "
            f"max = {result['w_plus'].max():.5f}."
        )

    def test_constraints_medium_universe(self, medium_setup, lam):
        """All weight constraints must hold for a larger 10-asset universe."""
        r_bar, C, w0p, w0m, _, _ = medium_setup
        n = len(r_bar)
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        assert result['w_plus'].shape == (n,)
        assert (result['w_plus'] >= -1e-6).all()
        assert (result['w_minus'] >= -1e-6).all()
        simultaneous = (result['w_plus'] > 1e-4) & (result['w_minus'] > 1e-4)
        assert not simultaneous.any()


# ---------------------------------------------------------------------------
# tune_lambda
# ---------------------------------------------------------------------------
class TestTuneLambda:
    # The implementation returns 0.5 as a documented fallback when all
    # LAMBDA_CANDIDATES produce identical (e.g., infeasible) D2 Sharpe.
    _VALID_RETURNS = set(LAMBDA_CANDIDATES) | {0.5}

    def test_tune_lambda_returns_valid(self, small_setup):
        """tune_lambda must return a value from LAMBDA_CANDIDATES or the fallback 0.5."""
        _, _, w0p, w0m, tickers, rets = small_setup
        lam = tune_lambda(rets, tickers, w0p, w0m)
        assert lam in self._VALID_RETURNS, (
            f"tune_lambda must return from {self._VALID_RETURNS}; got {lam}."
        )

    def test_tune_lambda_returns_float(self, small_setup):
        _, _, w0p, w0m, tickers, rets = small_setup
        lam = tune_lambda(rets, tickers, w0p, w0m)
        assert isinstance(lam, float), "tune_lambda must return a float."

    def test_tune_lambda_medium_universe(self, medium_setup):
        _, _, w0p, w0m, tickers, rets = medium_setup
        lam = tune_lambda(rets, tickers, w0p, w0m)
        assert lam in self._VALID_RETURNS


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_fallback_on_empty_universe(self):
        """solve_drmv with 0 assets must not raise an unhandled exception."""
        r_bar = np.array([])
        C = np.empty((0, 0))
        w0 = np.array([])
        try:
            result = solve_drmv(r_bar, C, w0, w0, lam=0.5)
            assert len(result['w_plus']) == 0
            assert len(result['w_minus']) == 0
        except Exception as exc:
            pytest.fail(
                f"solve_drmv raised an unhandled exception for empty universe: {exc}"
            )

    def test_result_keys_complete(self, small_setup, lam):
        """Result dict must contain all documented keys."""
        r_bar, C, w0p, w0m, _, _ = small_setup
        result = solve_drmv(r_bar, C, w0p, w0m, lam=lam)
        required_keys = {
            'status', 'w_plus', 'w_minus', 'x',
            'l_plus', 'l_minus', 's_plus', 's_minus',
            'transaction_cost', 'objective',
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )
