"""Portfolio performance metrics for the 2025 TDA-DRMV study."""

import numpy as np
import pandas as pd
import scipy.stats
import logging

logger = logging.getLogger(__name__)

_METRIC_KEYS = (
    'mean', 'std', 'sharpe', 'var_95', 'cvar_95', 'min_return',
    'max_return', 'max_drawdown', 'avg_drawdown', 'starr',
    'sterling', 'cumulative_return', 'terminal_wealth',
)


def compute_max_drawdown(returns: pd.Series) -> float:
    """Return maximum drawdown (a non-positive float) of a return series."""
    wealth = (1 + returns).cumprod()
    rolling_max = wealth.cummax()
    drawdown = (wealth - rolling_max) / rolling_max
    return float(drawdown.min())


def compute_avg_drawdown(returns: pd.Series) -> float:
    """Return the time-average of drawdown values (a non-positive float)."""
    wealth = (1 + returns).cumprod()
    rolling_max = wealth.cummax()
    drawdown = (wealth - rolling_max) / rolling_max
    return float(drawdown.mean())


def compute_metrics(returns: pd.Series, rf: float = 0.0) -> dict:
    """Compute the full set of out-of-sample performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (net, not gross).
    rf : float
        Daily risk-free rate used in Sharpe computation.

    Returns
    -------
    dict
        Keys: mean, std, sharpe, var_95, cvar_95, min_return, max_return,
        max_drawdown, avg_drawdown, starr, sterling, cumulative_return,
        terminal_wealth.
    """
    if returns.empty:
        logger.warning("compute_metrics received empty returns series; returning zeros.")
        return {k: 0.0 for k in _METRIC_KEYS}

    mean = returns.mean()
    std = returns.std(ddof=1)
    sharpe = (mean - rf) / std if std > 0 else 0.0

    var_95 = np.percentile(returns, 5)
    tail = returns[returns <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95

    min_return = returns.min()
    max_return = returns.max()
    max_dd = compute_max_drawdown(returns)
    avg_dd = compute_avg_drawdown(returns)

    starr = mean / abs(cvar_95) if cvar_95 != 0 else 0.0
    sterling = mean / abs(max_dd) if max_dd != 0 else 0.0

    cum_wealth = (1 + returns).cumprod()
    terminal_wealth = float(cum_wealth.iloc[-1])
    cumulative_return = terminal_wealth - 1.0

    return {
        'mean': float(mean),
        'std': float(std),
        'sharpe': float(sharpe),
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'min_return': float(min_return),
        'max_return': float(max_return),
        'max_drawdown': float(max_dd),
        'avg_drawdown': float(avg_dd),
        'starr': float(starr),
        'sterling': float(sterling),
        'cumulative_return': float(cumulative_return),
        'terminal_wealth': float(terminal_wealth),
    }


def paired_ttest_mean_return(returns_a: pd.Series, returns_b: pd.Series) -> dict:
    """One-sided paired t-test: H0: mean(a) <= mean(b), H1: mean(a) > mean(b).

    Aligns both series on their common dates before computing the difference.

    Returns
    -------
    dict
        t_stat, p_value (one-sided), significant_at_05.
    """
    common = returns_a.index.intersection(returns_b.index)
    if len(common) == 0:
        raise ValueError("No common dates between the two return series.")

    diff = returns_a.loc[common].values - returns_b.loc[common].values
    t_stat, p_two_sided = scipy.stats.ttest_1samp(diff, 0.0)

    # One-sided p-value for H1: mean(a) > mean(b)
    p_one_sided = p_two_sided / 2.0 if t_stat > 0 else 1.0 - p_two_sided / 2.0

    return {
        't_stat': float(t_stat),
        'p_value': float(p_one_sided),
        'significant_at_05': bool(p_one_sided < 0.05),
    }


def bootstrap_sharpe_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Ledoit-Wolf (2008) bootstrap Sharpe ratio test.

    H0: SR(a) <= SR(b), H1: SR(a) > SR(b).

    Parameters
    ----------
    returns_a, returns_b : pd.Series
        Daily return series aligned on a common index.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        delta_obs (SR_a - SR_b), p_value, significant_at_05.
    """
    common = returns_a.index.intersection(returns_b.index)
    if len(common) == 0:
        raise ValueError("No common dates between the two return series.")

    a = returns_a.loc[common].values
    b = returns_b.loc[common].values

    def _sharpe(x: np.ndarray) -> float:
        s = x.std(ddof=1)
        return float(x.mean() / s) if s > 0 else 0.0

    delta_obs = _sharpe(a) - _sharpe(b)

    rng = np.random.default_rng(seed)
    n = len(a)
    count_ge = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta_boot = _sharpe(a[idx]) - _sharpe(b[idx])
        if delta_boot >= delta_obs:
            count_ge += 1

    p_value = count_ge / n_bootstrap

    return {
        'delta_obs': float(delta_obs),
        'p_value': float(p_value),
        'significant_at_05': bool(p_value < 0.05),
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2024-01-01', periods=252)

    ret_a = pd.Series(rng.normal(0.001, 0.015, 252), index=dates, name='A')
    ret_b = pd.Series(rng.normal(0.0005, 0.015, 252), index=dates, name='B')

    print("=== Metrics for series A ===")
    m = compute_metrics(ret_a)
    for k, v in m.items():
        print(f"  {k:<22}: {v:.6f}")

    print("\n=== Paired t-test (A > B) ===")
    ttest = paired_ttest_mean_return(ret_a, ret_b)
    print(ttest)

    print("\n=== Bootstrap Sharpe test (A > B) ===")
    btest = bootstrap_sharpe_test(ret_a, ret_b, n_bootstrap=500)
    print(btest)

    print("\n=== Empty series edge case ===")
    print(compute_metrics(pd.Series([], dtype=float)))
