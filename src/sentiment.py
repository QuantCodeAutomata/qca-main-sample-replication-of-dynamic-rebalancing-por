"""
FinBERT-based sentiment analysis for financial news headlines.

Sentiment score S = P_pos - P_neg lies in [-1, 1], where P_pos and P_neg
are the FinBERT-assigned probabilities for the 'positive' and 'negative'
classes respectively.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

FINBERT_MODEL = 'ProsusAI/finbert'


class SentimentAnalyzer:
    """FinBERT-based financial news sentiment scorer.

    Wraps the ``ProsusAI/finbert`` model (or any compatible
    sequence-classification checkpoint) in a Hugging Face pipeline and
    exposes convenience methods for scoring individual headlines, batches,
    and aggregated daily scores.
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        device: str = 'cpu',
    ) -> None:
        """Load FinBERT and create the inference pipeline.

        Parameters
        ----------
        model_name : str
            Hugging Face model identifier.
        device : str
            ``'cpu'`` or a CUDA device string such as ``'cuda:0'``.
            The pipeline ``device`` argument is set to ``-1`` for CPU
            and to the numeric GPU index otherwise.
        """
        logger.info(
            'Loading FinBERT model: %s  |  device=%s', model_name, device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device == 'cpu':
            pipeline_device = -1
        else:
            # Accepts 'cuda', 'cuda:0', 'cuda:1', …
            pipeline_device = int(device.split(':')[-1]) if ':' in device else 0

        self.pipe = pipeline(
            'text-classification',
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=pipeline_device,
        )
        logger.info('FinBERT pipeline ready on device=%s.', device)

    # ------------------------------------------------------------------ #
    # Single-headline scoring
    # ------------------------------------------------------------------ #

    def score_headline(self, text: str) -> float:
        """Score one headline and return S = P_pos − P_neg.

        Parameters
        ----------
        text : str
            A single financial news headline.  Truncated to 512 characters
            before tokenisation.

        Returns
        -------
        float
            Sentiment score in ``[-1, 1]``.  Returns ``0.0`` on any error.
        """
        try:
            label_scores: List[dict] = self.pipe(text[:512])[0]
            score_map = {d['label'].lower(): d['score'] for d in label_scores}
            return float(
                score_map.get('positive', 0.0) - score_map.get('negative', 0.0)
            )
        except Exception as exc:
            logger.warning(
                'score_headline failed for text=%r: %s', text[:80], exc
            )
            return 0.0

    # ------------------------------------------------------------------ #
    # Batch scoring
    # ------------------------------------------------------------------ #

    def score_headlines_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """Score a list of headlines in mini-batches.

        Parameters
        ----------
        texts : list[str]
            Financial news headlines.
        batch_size : int
            Number of texts per inference call.

        Returns
        -------
        list[float]
            Sentiment scores in the same order as *texts*.
        """
        results: List[float] = []
        for start in range(0, len(texts), batch_size):
            batch = [t[:512] for t in texts[start: start + batch_size]]
            try:
                batch_output: List[List[dict]] = self.pipe(batch)
                for label_scores in batch_output:
                    score_map = {
                        d['label'].lower(): d['score'] for d in label_scores
                    }
                    results.append(
                        float(
                            score_map.get('positive', 0.0)
                            - score_map.get('negative', 0.0)
                        )
                    )
            except Exception as exc:
                logger.warning(
                    'Batch scoring failed at offset %d: %s — filling with 0.0.',
                    start, exc,
                )
                results.extend([0.0] * len(batch))
        return results

    # ------------------------------------------------------------------ #
    # Daily aggregation
    # ------------------------------------------------------------------ #

    def aggregate_daily_sentiment(
        self, headlines_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute mean FinBERT sentiment score per (ticker, date).

        Parameters
        ----------
        headlines_df : pd.DataFrame
            Must contain columns ``[ticker, date, headline]``.

        Returns
        -------
        pd.DataFrame
            Columns: ``[ticker, date, sentiment_score]``.
            One row per unique (ticker, date) pair.
        """
        if headlines_df.empty:
            logger.warning('aggregate_daily_sentiment received empty DataFrame.')
            return pd.DataFrame(columns=['ticker', 'date', 'sentiment_score'])

        texts = headlines_df['headline'].tolist()
        scores = self.score_headlines_batch(texts)

        scored = headlines_df[['ticker', 'date']].copy()
        scored['sentiment_score'] = scores

        aggregated = (
            scored.groupby(['ticker', 'date'], as_index=False)['sentiment_score']
            .mean()
        )
        logger.info(
            'aggregate_daily_sentiment: %d (ticker, date) pairs from %d headlines.',
            len(aggregated), len(headlines_df),
        )
        return aggregated


# --------------------------------------------------------------------------- #
# Panel construction helpers
# --------------------------------------------------------------------------- #

def build_sentiment_panel(
    headlines_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    tickers: List[str],
    analyzer: SentimentAnalyzer,
    missing_rule: str = 'zero',
) -> pd.DataFrame:
    """Build a (trading_dates × tickers) sentiment panel from raw headlines.

    Non-trading-day sentiment (weekends, holidays) is reassigned to the
    *next* available trading day before aggregation.

    Parameters
    ----------
    headlines_df : pd.DataFrame
        Columns: ``[ticker, date, headline]``.
    trading_dates : pd.DatetimeIndex
        Ordered set of valid trading dates that defines the panel index.
    tickers : list[str]
        Universe of tickers that defines the panel columns.
    analyzer : SentimentAnalyzer
        Fitted FinBERT sentiment scorer.
    missing_rule : {'zero', 'ffill'}
        Strategy for (ticker, date) pairs with no news coverage.
        ``'zero'`` fills with ``0.0``; ``'ffill'`` forward-fills from
        the most recent available score.

    Returns
    -------
    pd.DataFrame
        Index = *trading_dates*, columns = *tickers*.
    """
    daily_scores = analyzer.aggregate_daily_sentiment(headlines_df)

    trading_dates_sorted: pd.DatetimeIndex = trading_dates.sort_values()

    def _to_next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
        """Map *d* to itself if it is a trading day, else to the next one."""
        idx = trading_dates_sorted.searchsorted(d)
        if idx < len(trading_dates_sorted):
            return trading_dates_sorted[idx]
        return pd.NaT

    daily_scores['date'] = pd.to_datetime(daily_scores['date'])
    daily_scores['date'] = daily_scores['date'].apply(_to_next_trading_day)
    daily_scores.dropna(subset=['date'], inplace=True)

    # Re-aggregate: multiple calendar days can now share the same trading date
    daily_scores = (
        daily_scores.groupby(['ticker', 'date'], as_index=False)['sentiment_score']
        .mean()
    )

    panel = daily_scores.pivot(index='date', columns='ticker', values='sentiment_score')
    panel = panel.reindex(index=trading_dates_sorted, columns=tickers)

    if missing_rule == 'ffill':
        panel = panel.ffill()
    else:
        panel = panel.fillna(0.0)

    logger.info(
        'Sentiment panel built: shape=%s | missing_rule=%s', panel.shape, missing_rule
    )
    return panel


def create_mock_sentiment_panel(
    trading_dates: pd.DatetimeIndex,
    tickers: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic sentiment data for testing when real news is unavailable.

    Values are drawn from N(0, 0.1) and clipped to [-1, 1].

    Parameters
    ----------
    trading_dates : pd.DatetimeIndex
        Panel row index.
    tickers : list[str]
        Panel column labels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Index = *trading_dates*, columns = *tickers*.
    """
    logger.warning(
        'Using MOCK sentiment data - real FinBERT inference unavailable'
    )
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=0.1, size=(len(trading_dates), len(tickers)))
    data = np.clip(data, -1.0, 1.0)
    return pd.DataFrame(data, index=trading_dates, columns=tickers)


if __name__ == '__main__':
    _tickers = ['AAPL', 'MSFT', 'GOOGL']
    _dates = pd.date_range('2025-01-01', periods=10, freq='B')
    _mock_panel = create_mock_sentiment_panel(_dates, _tickers)
    print(_mock_panel)
