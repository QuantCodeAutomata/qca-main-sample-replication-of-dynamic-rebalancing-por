"""Main-sample experiment (2025-01-01 to 2025-12-31).

Runs a full grid search over (T1, T2, alpha, clustering spec), computes
out-of-sample performance metrics, statistical tests, and saves all results.
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
MAIN_START = '2025-01-01'
MAIN_END = '2025-12-31'
RESULTS_DIR = 'results'

# FAST_RUN: restricts grid to a single (T1, T2, alpha) to allow quick
# end-to-end validation in CI.  Set FAST_RUN=0 in the environment for the
# full production run.
FAST_RUN: bool = bool(int(os.environ.get("FAST_RUN", "1")))

USE_MOCK_SENTIMENT = True   # Set False only when real news API is available
# In FAST_RUN the Massive API key is typically absent; fall back to mock data.
USE_MOCK_MARKET: bool = FAST_RUN or not bool(os.environ.get("MASSIVE_API_KEY", ""))

T1_VALUES    = [63]       if FAST_RUN else [63, 84]
T2_VALUES    = [3]        if FAST_RUN else [3, 5, 10]
ALPHA_VALUES = [0.95]     if FAST_RUN else [0.95, 0.99]
# In FAST_RUN only conventional (non-TDA) specs are included so the run
# finishes in a few minutes without ripser.
_ALL_SPECS   = ["AWDS", "APLS", "ACS", "AES", "AWDI", "APLI", "ACI", "AEI"]
_FAST_SPECS  = ["ACS", "AES", "ACI", "AEI"]
ACTIVE_SPECS: list[str] = _FAST_SPECS if FAST_RUN else _ALL_SPECS


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------
def create_mock_market_data(
    n_tickers: int = 20,
    n_days: int = 250,
    seed: int = 42,
) -> tuple:
    """Generate synthetic OHLCV data via a geometric random walk.

    Each close price follows  C_t = C_{t-1} * exp(N(0.0005, 0.02)).
    Trading dates start at 2024-01-01 for a burn-in period that precedes
    the main-sample window.

    Returns
    -------
    tuple
        (close_df, ohlcv_dict, returns_df, tickers, dates)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_days)
    tickers = [f'MOCK{i:02d}' for i in range(n_tickers)]

    # Geometric random walk: log-return ~ N(mu, sigma)
    log_rets = rng.normal(0.0005, 0.02, size=(n_days, n_tickers))
    log_prices = np.cumsum(log_rets, axis=0)
    prices = np.exp(log_prices) * 100.0  # anchor at $100

    close_df = pd.DataFrame(prices, index=dates, columns=tickers)
    returns_df = close_df.pct_change().dropna()

    ohlcv_dict: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        c = close_df[ticker].values
        o = np.empty_like(c)
        o[0] = c[0]
        o[1:] = c[:-1]

        spread_h = rng.uniform(1.000, 1.010, size=n_days)
        spread_l = rng.uniform(0.990, 1.000, size=n_days)
        h = np.maximum(o, c) * spread_h
        lo = np.minimum(o, c) * spread_l
        vol = rng.integers(100_000, 1_000_000, size=n_days)

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
    """Convert metric dict values to JSON-serialisable Python scalars."""
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
            for k, v in m.items()}


def _save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    logging.getLogger(__name__).info("Saved JSON: %s", path)


def _append_results_md(lines: list[str], results_dir: str) -> None:
    """Write *lines* to RESULTS.md, replacing any existing Experiment 1 section."""
    path = os.path.join(results_dir, 'RESULTS.md')
    os.makedirs(results_dir, exist_ok=True)
    new_block = '\n'.join(lines) + '\n'
    if os.path.exists(path):
        with open(path) as fh:
            content = fh.read()
        # Remove the existing Experiment 1 block (up to next "## Experiment" or EOF)
        import re as _re
        content = _re.sub(
            r'(?m)^## Experiment 1.*?(?=^## Experiment |\Z)',
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
        _n_days = 250 if FAST_RUN else 300
        log.info("Using MOCK market data (%d tickers, %d days).", _n_tick, _n_days)
        close_df, ohlcv_dict, returns_df, tickers, dates = create_mock_market_data(
            n_tickers=_n_tick, n_days=_n_days, seed=42,
        )
    else:
        log.info("Fetching real market data for %s – %s.", MAIN_START, MAIN_END)
        # Use a one-year look-back before the sample for burn-in
        fetch_start = str(
            (pd.Timestamp(MAIN_START) - pd.DateOffset(years=1)).date()
        )
        tickers = get_sp500_tickers()
        # fetch_ohlcv returns (close_wide: DataFrame, ohlcv_panel: long DataFrame)
        close_df, _ohlcv_panel = fetch_ohlcv(tickers, fetch_start, MAIN_END)
        tickers = list(close_df.columns)  # only tickers with complete history
        returns_df = compute_daily_returns(close_df)
        dates = close_df.index
        # Convert long-format ohlcv_panel → dict {ticker: OHLCV DataFrame}
        # (required by compute_all_indicators and run_rolling_backtest)
        ohlcv_dict: dict[str, pd.DataFrame] = {}
        for _ticker, _grp in _ohlcv_panel.groupby('ticker'):
            if _ticker in tickers:
                _df = _grp.set_index('date').rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume',
                })
                ohlcv_dict[str(_ticker)] = _df

    log.info(
        "Market data: %d tickers, %d trading days (%s to %s).",
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
    # 3. Grid search
    # ------------------------------------------------------------------
    grid_returns: dict[str, pd.Series] = {}  # key -> OOS return series
    grid_metrics: dict[str, dict] = {}

    spec_names = ACTIVE_SPECS
    total_runs = len(T1_VALUES) * len(T2_VALUES) * len(ALPHA_VALUES) * len(spec_names)
    log.info(
        "Starting grid (FAST_RUN=%s): %d T1 × %d T2 × %d alpha × %d specs = %d runs.",
        FAST_RUN, len(T1_VALUES), len(T2_VALUES), len(ALPHA_VALUES), len(spec_names), total_runs,
    )

    run_idx = 0
    for T1 in T1_VALUES:
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
                        "  Sharpe=%.3f  CumRet=%.2f%%",
                        metrics['sharpe'],
                        metrics['cumulative_return'] * 100,
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
            spy_series = fetch_sp500_index(MAIN_START, MAIN_END)
            if not spy_series.empty:
                bm_bnh = spy_series.pct_change().dropna()
                benchmark_returns['B&H (S&P500)'] = bm_bnh
                benchmark_metrics['B&H (S&P500)'] = _metrics_to_serialisable(
                    compute_metrics(bm_bnh)
                )
        except Exception as exc:
            log.error("S&P500 B&H benchmark failed: %s", exc)

    # Full-universe DRMV benchmark ((DRMV)_A): single run at primary settings
    # T1=63, T2=3 matches the paper's primary comparison table settings.
    try:
        bm_drmv = run_benchmark_full_drmv(
            returns_df, tickers=tickers, T1=T1_VALUES[0], T2=T2_VALUES[0]
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
        best_key = max(grid_metrics, key=lambda k: grid_metrics[k].get('sharpe', -np.inf))
        log.info("Best grid spec by Sharpe: %s", best_key)
        ref_key = 'Full-DRMV'

        if best_key in grid_returns and ref_key in benchmark_returns:
            try:
                ttest_res = paired_ttest_mean_return(
                    grid_returns[best_key], benchmark_returns[ref_key]
                )
                boot_res = bootstrap_sharpe_test(
                    grid_returns[best_key], benchmark_returns[ref_key], n_bootstrap=1000
                )
                stat_tests['paired_ttest'] = {'comparison': f"{best_key} vs {ref_key}", **ttest_res}
                stat_tests['bootstrap_sharpe'] = {'comparison': f"{best_key} vs {ref_key}", **boot_res}
            except Exception as exc:
                log.error("Statistical tests failed: %s", exc)

    # ------------------------------------------------------------------
    # 6. Clustering diagnostics
    # ------------------------------------------------------------------
    clustering_diagnostics: dict[str, dict] = {}
    for key, metrics in grid_metrics.items():
        clustering_diagnostics[key] = {
            'sharpe': metrics.get('sharpe', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'starr': metrics.get('starr', 0.0),
        }

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    all_metrics = {**grid_metrics, **benchmark_metrics}
    _save_json(all_metrics, os.path.join(RESULTS_DIR, 'exp1_metrics.json'))
    _save_json(clustering_diagnostics, os.path.join(RESULTS_DIR, 'exp1_clustering_diagnostics.json'))
    if stat_tests:
        _save_json(stat_tests, os.path.join(RESULTS_DIR, 'exp1_stat_tests.json'))

    # Cumulative-return plots for T1=63, T2=3
    for alpha_val, alpha_str in [(0.95, '095'), (0.99, '099')]:
        plot_series: dict[str, pd.Series] = {}
        prefix = f"T1=63_T2=3_a={alpha_val}_"
        for k, v in grid_returns.items():
            if k.startswith(prefix):
                plot_series[k.replace(prefix, '')] = v
        for bm_k, bm_v in benchmark_returns.items():
            plot_series[bm_k] = bm_v

        if plot_series:
            png_path = os.path.join(
                RESULTS_DIR, f"exp1_cumret_T1_63_T2_3_alpha_{alpha_str}.png"
            )
            _plot_cumulative_returns(
                plot_series,
                title=f"Exp 1 Cumulative Returns — T1=63, T2=3, α={alpha_val}",
                filepath=png_path,
            )

    # ------------------------------------------------------------------
    # 8. Update RESULTS.md
    # ------------------------------------------------------------------
    md_lines = [
        '',
        '## Experiment 1 — Main Sample (2025-01-01 to 2025-12-31)',
        '',
        f'Grid: T1={T1_VALUES}, T2={T2_VALUES}, alpha={ALPHA_VALUES}, '
        f'specs={spec_names}',
        '',
        '### Top-5 specs by Sharpe ratio',
        '',
        '| Spec | Sharpe | CumRet | MaxDD | STARR |',
        '|------|--------|--------|-------|-------|',
    ]
    sorted_keys = sorted(grid_metrics, key=lambda k: grid_metrics[k].get('sharpe', 0), reverse=True)
    for k in sorted_keys[:5]:
        m = grid_metrics[k]
        md_lines.append(
            f"| {k} | {m.get('sharpe', 0):.3f} | "
            f"{m.get('cumulative_return', 0)*100:.1f}% | "
            f"{m.get('max_drawdown', 0)*100:.1f}% | "
            f"{m.get('starr', 0):.3f} |"
        )
    md_lines += ['', '### Benchmarks', '']
    for bk, bm in benchmark_metrics.items():
        md_lines.append(
            f"- **{bk}**: Sharpe={bm.get('sharpe', 0):.3f}, "
            f"CumRet={bm.get('cumulative_return', 0)*100:.1f}%"
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
    # 9. Console summary table
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('EXPERIMENT 1 SUMMARY — Main Sample')
    print('=' * 80)
    header = f"{'Spec':<50} {'Sharpe':>8} {'CumRet':>8} {'MaxDD':>8} {'STARR':>8}"
    print(header)
    print('-' * 80)
    for k in sorted_keys[:10]:
        m = grid_metrics[k]
        print(
            f"{k:<50} "
            f"{m.get('sharpe', 0):>8.3f} "
            f"{m.get('cumulative_return', 0)*100:>7.1f}% "
            f"{m.get('max_drawdown', 0)*100:>7.1f}% "
            f"{m.get('starr', 0):>8.3f}"
        )
    print('-' * 80)
    for bk, bm in benchmark_metrics.items():
        print(
            f"{'[BM] ' + bk:<50} "
            f"{bm.get('sharpe', 0):>8.3f} "
            f"{bm.get('cumulative_return', 0)*100:>7.1f}% "
            f"{bm.get('max_drawdown', 0)*100:>7.1f}% "
            f"{bm.get('starr', 0):>8.3f}"
        )
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
