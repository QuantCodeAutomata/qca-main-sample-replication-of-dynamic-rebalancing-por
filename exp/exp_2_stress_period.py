"""Stress-period experiment (2025-10-01 to 2026-03-31).

Tests portfolio resilience during the conflict window with a fixed formation
window T1=63 and a grid over rebalancing frequencies T2 and CVaR level alpha.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.fetch_market_data import (
    get_sp500_tickers,
    fetch_ohlcv,
    fetch_sp500_index,
    compute_daily_returns,
)
from src.sentiment import create_mock_sentiment_panel
from src.backtest import (
    run_rolling_backtest,
    run_benchmark_ew,
    run_benchmark_full_drmv,
    CLUSTERING_SPECS,
)
from src.metrics import compute_metrics, paired_ttest_mean_return, bootstrap_sharpe_test

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STRESS_START = '2025-10-01'
STRESS_END = '2026-03-31'
T1 = 63                       # Fixed formation window for stress analysis
RESULTS_DIR = 'results'

# FAST_RUN: restricts grid to a single (T2, alpha) to allow quick end-to-end
# validation in CI.  Set FAST_RUN=0 in the environment for the full run.
FAST_RUN: bool = bool(int(os.environ.get("FAST_RUN", "1")))

USE_MOCK_SENTIMENT = True     # Set False only when real news API is available
USE_MOCK_MARKET: bool = FAST_RUN or not bool(os.environ.get("MASSIVE_API_KEY", ""))

T2_VALUES    = [3]        if FAST_RUN else [3, 5, 10]
ALPHA_VALUES = [0.95]     if FAST_RUN else [0.95, 0.99]
_ALL_SPECS   = ["AWDS", "APLS", "ACS", "AES", "AWDI", "APLI", "ACI", "AEI"]
_FAST_SPECS  = ["ACS", "AES", "ACI", "AEI"]
ACTIVE_SPECS: list[str] = _FAST_SPECS if FAST_RUN else _ALL_SPECS


# ---------------------------------------------------------------------------
# Mock data helper (stress-period variant)
# ---------------------------------------------------------------------------
def _create_mock_stress_data(
    n_tickers: int = 30,
    seed: int = 99,
) -> tuple:
    """Simulate a stress period via a random walk with higher volatility.

    The stress window is 2025-10-01 to 2026-03-31 (~130 business days).
    A burn-in of T1 days before the window is prepended so the rolling
    backtest has enough history for the first formation window.

    Returns
    -------
    tuple
        (close_df, ohlcv_dict, returns_df, tickers, dates)
    """
    rng = np.random.default_rng(seed)
    n_burn = T1 + 10  # extra buffer
    stress_dates = pd.bdate_range(STRESS_START, STRESS_END)
    burn_dates = pd.bdate_range(
        end=pd.Timestamp(STRESS_START) - pd.Timedelta(days=1),
        periods=n_burn,
    )
    dates = burn_dates.append(stress_dates)
    n_days = len(dates)

    tickers = [f'MOCK{i:02d}' for i in range(n_tickers)]

    # Higher volatility during stress (sigma=0.03 vs normal 0.02)
    log_rets = rng.normal(-0.0002, 0.03, size=(n_days, n_tickers))
    log_prices = np.cumsum(log_rets, axis=0)
    prices = np.exp(log_prices) * 100.0

    close_df = pd.DataFrame(prices, index=dates, columns=tickers)
    returns_df = close_df.pct_change().dropna()

    ohlcv_dict: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        c = close_df[ticker].values
        o = np.empty_like(c)
        o[0] = c[0]
        o[1:] = c[:-1]
        spread_h = rng.uniform(1.000, 1.015, size=n_days)
        spread_l = rng.uniform(0.985, 1.000, size=n_days)
        h = np.maximum(o, c) * spread_h
        lo = np.minimum(o, c) * spread_l
        vol = rng.integers(200_000, 2_000_000, size=n_days)
        ohlcv_dict[ticker] = pd.DataFrame(
            {'Open': o, 'High': h, 'Low': lo, 'Close': c, 'Volume': vol},
            index=dates,
        )

    return close_df, ohlcv_dict, returns_df, tickers, dates


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_cumulative_returns(
    series_dict: dict[str, pd.Series],
    title: str,
    filepath: str,
) -> None:
    """Save a cumulative-return line chart to *filepath*."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, ret in series_dict.items():
        cum = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values, label=label, linewidth=1.2)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Wealth ($1 invested)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logging.getLogger(__name__).info("Saved plot: %s", filepath)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------
def _metrics_to_serialisable(m: dict) -> dict:
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
            for k, v in m.items()}


def _save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    logging.getLogger(__name__).info("Saved JSON: %s", path)


def _append_results_md(lines: list[str], results_dir: str) -> None:
    """Write *lines* to RESULTS.md, replacing any existing Experiment 2 section."""
    import re as _re
    path = os.path.join(results_dir, 'RESULTS.md')
    os.makedirs(results_dir, exist_ok=True)
    new_block = '\n'.join(lines) + '\n'
    if os.path.exists(path):
        with open(path) as fh:
            content = fh.read()
        # Remove the existing Experiment 2 block (up to next "## Experiment" or EOF)
        content = _re.sub(
            r'(?m)^## Experiment 2.*?(?=^## Experiment |\Z)',
            '', content, flags=_re.DOTALL,
        ).rstrip('\n')
        updated = (content + '\n\n' + new_block).lstrip('\n')
    else:
        updated = new_block
    with open(path, 'w') as fh:
        fh.write(updated)
    logging.getLogger(__name__).info("Updated %s", path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger(__name__)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Market data
    # ------------------------------------------------------------------
    if USE_MOCK_MARKET:
        _n_tick = 20 if FAST_RUN else 30
        log.info("Using MOCK stress-period market data (%d tickers).", _n_tick)
        close_df, ohlcv_dict, returns_df, tickers, dates = _create_mock_stress_data(
            n_tickers=_n_tick, seed=99,
        )
    else:
        log.info("Fetching real market data for %s – %s.", STRESS_START, STRESS_END)
        fetch_start = str(
            (pd.Timestamp(STRESS_START) - pd.DateOffset(days=T1 + 30)).date()
        )
        tickers = get_sp500_tickers()
        # fetch_ohlcv returns (close_wide: DataFrame, ohlcv_panel: long DataFrame)
        close_df, _ohlcv_panel = fetch_ohlcv(tickers, fetch_start, STRESS_END)
        tickers = list(close_df.columns)  # only tickers with complete history
        returns_df = compute_daily_returns(close_df)
        dates = close_df.index
        # Convert long-format ohlcv_panel → dict {ticker: OHLCV DataFrame}
        ohlcv_dict: dict[str, pd.DataFrame] = {}
        for _ticker, _grp in _ohlcv_panel.groupby('ticker'):
            if _ticker in tickers:
                _df = _grp.set_index('date').rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume',
                })
                ohlcv_dict[str(_ticker)] = _df

    log.info(
        "Stress-period data: %d tickers, %d trading days (%s to %s).",
        len(tickers), len(dates), dates[0].date(), dates[-1].date(),
    )

    # ------------------------------------------------------------------
    # 2. Sentiment panel
    # ------------------------------------------------------------------
    if USE_MOCK_SENTIMENT:
        log.info("Using MOCK sentiment panel.")
        sentiment_df = create_mock_sentiment_panel(dates, list(returns_df.columns))
    else:
        raise NotImplementedError(
            "Real sentiment loading not implemented; set USE_MOCK_SENTIMENT=True."
        )

    # ------------------------------------------------------------------
    # 3. Grid search (T1 fixed at 63)
    # ------------------------------------------------------------------
    spec_names = ACTIVE_SPECS  # FAST_RUN uses conventional-only subset
    total_runs = len(T2_VALUES) * len(ALPHA_VALUES) * len(spec_names)
    log.info(
        "Starting stress grid: T1=%d, %d T2 × %d alpha × %d specs = %d runs "
        "(FAST_RUN=%s).",
        T1, len(T2_VALUES), len(ALPHA_VALUES), len(spec_names), total_runs, FAST_RUN,
    )

    grid_returns: dict[str, pd.Series] = {}
    grid_metrics: dict[str, dict] = {}

    run_idx = 0
    for T2 in T2_VALUES:
        for alpha in ALPHA_VALUES:
            for spec_name in spec_names:
                run_idx += 1
                key = f"T1={T1}_T2={T2}_a={alpha}_{spec_name}"
                log.info("[%d/%d] %s", run_idx, total_runs, key)

                try:
                    bt_result = run_rolling_backtest(
                        close_df=close_df,
                        ohlcv_dict=ohlcv_dict,
                        sentiment_panel=sentiment_df,
                        tickers=tickers,
                        T1=T1,
                        T2=T2,
                        alpha=alpha,
                        spec_name=spec_name,
                        results_dir=RESULTS_DIR,
                    )
                    oos_returns = bt_result["oos_returns"]
                    metrics = compute_metrics(oos_returns)
                except Exception as exc:
                    log.error("Run %s failed: %s", key, exc)
                    continue

                grid_returns[key] = oos_returns
                grid_metrics[key] = _metrics_to_serialisable(metrics)
                log.info(
                    "  Sharpe=%.3f  MaxDD=%.2f%%",
                    metrics['sharpe'],
                    metrics['max_drawdown'] * 100,
                )

    # ------------------------------------------------------------------
    # 4. Benchmarks
    # ------------------------------------------------------------------
    log.info("Running benchmarks: EW, Full-DRMV.")
    benchmark_returns: dict[str, pd.Series] = {}
    benchmark_metrics: dict[str, dict] = {}

    try:
        bm_ew = run_benchmark_ew(returns_df, tickers=tickers)
        benchmark_returns['EW'] = bm_ew
        benchmark_metrics['EW'] = _metrics_to_serialisable(compute_metrics(bm_ew))
    except Exception as exc:
        log.error("EW benchmark failed: %s", exc)

    if not USE_MOCK_MARKET:
        try:
            # fetch_sp500_index returns a pd.Series of SPY close prices
            spy_series = fetch_sp500_index(STRESS_START, STRESS_END)
            if not spy_series.empty:
                bm_bnh = spy_series.pct_change().dropna()
                benchmark_returns['B&H (S&P500)'] = bm_bnh
                benchmark_metrics['B&H (S&P500)'] = _metrics_to_serialisable(
                    compute_metrics(bm_bnh)
                )
        except Exception as exc:
            log.error("S&P500 B&H benchmark failed: %s", exc)

    # Full-universe DRMV benchmark ((DRMV)_A): single run at T1=63, T2=3
    try:
        bm_drmv = run_benchmark_full_drmv(
            returns_df, tickers=tickers, T1=T1, T2=T2_VALUES[0]
        )
        benchmark_returns['Full-DRMV'] = bm_drmv
        benchmark_metrics['Full-DRMV'] = _metrics_to_serialisable(compute_metrics(bm_drmv))
    except Exception as exc:
        log.error("Full-DRMV benchmark failed: %s", exc)

    # ------------------------------------------------------------------
    # 5. Statistical tests — best spec vs. Full-DRMV
    # ------------------------------------------------------------------
    stat_tests: dict[str, dict] = {}
    if grid_metrics:
        # In stress periods, prefer max-drawdown minimisation over Sharpe
        best_key = max(
            grid_metrics,
            key=lambda k: (
                -grid_metrics[k].get('max_drawdown', -np.inf),
                grid_metrics[k].get('sharpe', -np.inf),
            ),
        )
        log.info("Best stress spec (by min MaxDD then Sharpe): %s", best_key)
        ref_key = 'Full-DRMV'

        if best_key in grid_returns and ref_key in benchmark_returns:
            try:
                ttest_res = paired_ttest_mean_return(
                    grid_returns[best_key], benchmark_returns[ref_key]
                )
                boot_res = bootstrap_sharpe_test(
                    grid_returns[best_key], benchmark_returns[ref_key], n_bootstrap=1000
                )
                stat_tests['paired_ttest'] = {
                    'comparison': f"{best_key} vs {ref_key}", **ttest_res
                }
                stat_tests['bootstrap_sharpe'] = {
                    'comparison': f"{best_key} vs {ref_key}", **boot_res
                }
            except Exception as exc:
                log.error("Statistical tests failed: %s", exc)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    all_metrics = {**grid_metrics, **benchmark_metrics}
    _save_json(all_metrics, os.path.join(RESULTS_DIR, 'exp2_metrics.json'))
    if stat_tests:
        _save_json(stat_tests, os.path.join(RESULTS_DIR, 'exp2_stat_tests.json'))

    # Cumulative return plot
    plot_series: dict[str, pd.Series] = {}
    plot_series.update(benchmark_returns)
    for k, v in grid_returns.items():
        plot_series[k] = v

    if plot_series:
        _plot_cumulative_returns(
            plot_series,
            title=f"Exp 2 Stress-Period Cumulative Returns ({STRESS_START} – {STRESS_END})",
            filepath=os.path.join(RESULTS_DIR, 'exp2_cumret_stress.png'),
        )

    # ------------------------------------------------------------------
    # 7. Update RESULTS.md
    # ------------------------------------------------------------------
    md_lines = [
        '',
        f'## Experiment 2 — Stress Period ({STRESS_START} to {STRESS_END})',
        '',
        f'T1={T1} (fixed), T2={T2_VALUES}, alpha={ALPHA_VALUES}, specs={spec_names}',
        '',
        '### Top-5 specs by minimum MaxDrawdown',
        '',
        '| Spec | MaxDD | Sharpe | CumRet | STARR |',
        '|------|-------|--------|--------|-------|',
    ]
    sorted_keys = sorted(
        grid_metrics,
        key=lambda k: (
            -grid_metrics[k].get('max_drawdown', -np.inf),
            grid_metrics[k].get('sharpe', -np.inf),
        ),
    )
    for k in sorted_keys[:5]:
        m = grid_metrics[k]
        md_lines.append(
            f"| {k} | {m.get('max_drawdown', 0)*100:.1f}% | "
            f"{m.get('sharpe', 0):.3f} | "
            f"{m.get('cumulative_return', 0)*100:.1f}% | "
            f"{m.get('starr', 0):.3f} |"
        )
    md_lines += ['', '### Benchmarks', '']
    for bk, bm in benchmark_metrics.items():
        md_lines.append(
            f"- **{bk}**: Sharpe={bm.get('sharpe', 0):.3f}, "
            f"MaxDD={bm.get('max_drawdown', 0)*100:.1f}%"
        )
    if stat_tests:
        md_lines += ['', '### Statistical Tests', '']
        for test_name, res in stat_tests.items():
            md_lines.append(
                f"- **{test_name}** ({res.get('comparison', '')}): "
                f"p={res.get('p_value', 'N/A')}, "
                f"sig@5%={res.get('significant_at_05', 'N/A')}"
            )

    _append_results_md(md_lines, RESULTS_DIR)

    # ------------------------------------------------------------------
    # 8. Console summary table
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print(f'EXPERIMENT 2 SUMMARY — Stress Period ({STRESS_START} – {STRESS_END})')
    print('=' * 80)
    header = f"{'Spec':<50} {'MaxDD':>8} {'Sharpe':>8} {'CumRet':>8} {'STARR':>8}"
    print(header)
    print('-' * 80)
    for k in sorted_keys[:10]:
        m = grid_metrics[k]
        print(
            f"{k:<50} "
            f"{m.get('max_drawdown', 0)*100:>7.1f}% "
            f"{m.get('sharpe', 0):>8.3f} "
            f"{m.get('cumulative_return', 0)*100:>7.1f}% "
            f"{m.get('starr', 0):>8.3f}"
        )
    print('-' * 80)
    for bk, bm in benchmark_metrics.items():
        print(
            f"{'[BM] ' + bk:<50} "
            f"{bm.get('max_drawdown', 0)*100:>7.1f}% "
            f"{bm.get('sharpe', 0):>8.3f} "
            f"{bm.get('cumulative_return', 0)*100:>7.1f}% "
            f"{bm.get('starr', 0):>8.3f}"
        )
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
