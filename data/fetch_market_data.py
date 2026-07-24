"""
S&P 500 market data fetcher using the Massive REST API.

Notes
-----
Prices returned by list_aggs are assumed to be split-adjusted by Massive's
data provider.  Adjustment status depends on the Massive subscription plan;
this assumption is logged at runtime.  Users should verify with their Massive
account details.
"""

import json
import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from massive import RESTClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

_SP500_TICKERS: List[str] = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'UNH', 'LLY', 'JPM', 'V', 'XOM', 'AVGO', 'PG', 'MA', 'HD', 'COST',
    'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'WMT', 'BAC', 'CRM', 'ACN', 'MCD',
    'TMO', 'CSCO', 'ADBE', 'ABT', 'NFLX', 'DIS', 'AMD', 'TXN', 'INTC',
    'WFC', 'PM', 'BMY', 'UPS', 'GE', 'HON', 'QCOM', 'SPGI', 'RTX', 'CAT',
    'AMGN', 'LIN', 'NOW',
]


def get_sp500_tickers() -> List[str]:
    """Return a representative hardcoded list of 50 major S&P 500 tickers.

    Returns
    -------
    List[str]
        Ticker symbols.
    """
    logger.info('S&P 500 representative ticker list: %s', _SP500_TICKERS)
    logger.info('Total tickers in list: %d', len(_SP500_TICKERS))
    return _SP500_TICKERS.copy()


def fetch_ohlcv(
    tickers: List[str],
    start_date: str,
    end_date: str,
    warmup_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch daily OHLCV bars for *tickers* from the Massive REST API.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to fetch.
    start_date : str
        Inclusive start of the *main* sample period (``'YYYY-MM-DD'``).
    end_date : str
        Inclusive end of the sample period (``'YYYY-MM-DD'``).
    warmup_days : int
        Business days prepended before *start_date* so that rolling
        indicators have enough history at the first in-sample date.

    Returns
    -------
    close_wide : pd.DataFrame
        Wide-format daily close prices.
        Index = date (``pd.Timestamp``), columns = tickers.
        Only tickers with a *complete* close-price history over
        ``[start_date, end_date]`` are retained.
    ohlcv_panel : pd.DataFrame
        Long-format panel with columns
        ``[ticker, date, open, high, low, close, volume]`` spanning
        the full window including warm-up days.

    Notes
    -----
    Prices are assumed to be split-adjusted by Massive's data provider.
    Adjustment status depends on the Massive subscription plan; this
    assumption is logged at runtime.
    """
    logger.info(
        'Price-adjustment assumption: split-adjusted '
        '(verify against your Massive subscription plan).'
    )

    api_key = os.getenv('MASSIVE_TOKEN') or os.getenv('MASSIVE_API_KEY')
    client = RESTClient(api_key=api_key)

    warmup_start: str = (
        pd.Timestamp(start_date) - pd.offsets.BDay(warmup_days)
    ).strftime('%Y-%m-%d')

    logger.info(
        'Fetching OHLCV | tickers=%d | warmup_start=%s | end_date=%s',
        len(tickers), warmup_start, end_date,
    )

    records: List[dict] = []
    api_failed: List[str] = []

    for ticker in tickers:
        try:
            aggs = client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='day',
                from_=warmup_start,
                to=end_date,
                limit=50000,
            )
            ticker_records: List[dict] = []
            for agg in aggs:
                date = pd.Timestamp(agg.timestamp, unit='ms', tz='UTC').date()
                ticker_records.append({
                    'ticker': ticker,
                    'date': date,
                    'open': float(agg.open),
                    'high': float(agg.high),
                    'low': float(agg.low),
                    'close': float(agg.close),
                    'volume': float(agg.volume),
                })
            records.extend(ticker_records)
            logger.info('Fetched %d bars for %s.', len(ticker_records), ticker)
        except Exception as exc:
            logger.warning(
                'API fetch failed for %s (%s) — ticker skipped.', ticker, exc
            )
            api_failed.append(ticker)

    if api_failed:
        logger.warning(
            'Tickers skipped due to API errors (%d): %s',
            len(api_failed), api_failed,
        )

    _empty_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    if not records:
        logger.error('No OHLCV data returned for any ticker.')
        return pd.DataFrame(), pd.DataFrame(columns=_empty_cols)

    ohlcv_panel = pd.DataFrame(records)
    ohlcv_panel['date'] = pd.to_datetime(ohlcv_panel['date'])
    ohlcv_panel.sort_values(['ticker', 'date'], inplace=True)
    ohlcv_panel.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------ #
    # Build wide-format close prices restricted to the main sample period
    # ------------------------------------------------------------------ #
    main_mask = (
        (ohlcv_panel['date'] >= pd.Timestamp(start_date))
        & (ohlcv_panel['date'] <= pd.Timestamp(end_date))
    )
    main_df = ohlcv_panel.loc[main_mask]

    close_wide: pd.DataFrame = main_df.pivot(
        index='date', columns='ticker', values='close'
    )
    close_wide.index.name = 'date'
    close_wide.columns.name = 'ticker'

    # Remove tickers that have ANY missing close price in [start_date, end_date]
    incomplete: List[str] = close_wide.columns[close_wide.isna().any()].tolist()
    if incomplete:
        logger.warning(
            'Removing %d tickers with missing close prices in [%s, %s] '
            '(reason: gaps in Massive data): %s',
            len(incomplete), start_date, end_date, incomplete,
        )
        close_wide.drop(columns=incomplete, inplace=True)
        ohlcv_panel = ohlcv_panel[~ohlcv_panel['ticker'].isin(incomplete)].copy()
        ohlcv_panel.reset_index(drop=True, inplace=True)

    logger.info(
        'close_wide shape: %s | ohlcv_panel shape: %s',
        close_wide.shape, ohlcv_panel.shape,
    )
    return close_wide, ohlcv_panel


def fetch_sp500_index(start_date: str, end_date: str) -> pd.Series:
    """Fetch SPY daily close prices as an S&P 500 benchmark proxy.

    Parameters
    ----------
    start_date : str
        Inclusive start date (``'YYYY-MM-DD'``).
    end_date : str
        Inclusive end date (``'YYYY-MM-DD'``).

    Returns
    -------
    pd.Series
        Daily SPY close prices indexed by ``pd.Timestamp`` date,
        named ``'SPY'``.  Returns an empty series on API failure.
    """
    api_key = os.getenv('MASSIVE_TOKEN') or os.getenv('MASSIVE_API_KEY')
    client = RESTClient(api_key=api_key)

    logger.info(
        'Fetching SPY (S&P 500 proxy) from %s to %s.', start_date, end_date
    )
    try:
        aggs = client.list_aggs(
            ticker='SPY',
            multiplier=1,
            timespan='day',
            from_=start_date,
            to=end_date,
            limit=50000,
        )
        records: List[dict] = []
        for agg in aggs:
            date = pd.Timestamp(agg.timestamp, unit='ms', tz='UTC').date()
            records.append({'date': date, 'close': float(agg.close)})

        if not records:
            logger.warning('No SPY data returned for [%s, %s].', start_date, end_date)
            return pd.Series(dtype=float, name='SPY')

        series = (
            pd.DataFrame(records)
            .set_index('date')['close']
            .rename('SPY')
        )
        series.index = pd.to_datetime(series.index)
        series.sort_index(inplace=True)
        logger.info('SPY series length: %d bars.', len(series))
        return series

    except Exception as exc:
        logger.error('Failed to fetch SPY: %s', exc)
        return pd.Series(dtype=float, name='SPY')


def compute_daily_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns r_it = (C_t − C_{t-1}) / C_{t-1}.

    Parameters
    ----------
    close_df : pd.DataFrame
        Wide-format close prices (index = date, columns = tickers).

    Returns
    -------
    pd.DataFrame
        Same shape as *close_df*; the first row contains NaN values
        because no prior close is available.
    """
    returns = close_df.pct_change()
    logger.info(
        'Computed daily returns: shape=%s, first row is NaN by construction.',
        returns.shape,
    )
    return returns


if __name__ == '__main__':
    _sample_tickers = get_sp500_tickers()[:5]
    _close_wide, _ohlcv_panel = fetch_ohlcv(
        _sample_tickers, '2024-12-01', '2025-01-31'
    )
    print('close_wide shape :', _close_wide.shape)
    print('ohlcv_panel shape:', _ohlcv_panel.shape)
