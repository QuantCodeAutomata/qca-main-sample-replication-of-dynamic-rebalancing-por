# Dynamic Rebalancing Portfolio Optimisation with TDA / Conventional Clustering and News Sentiment

Replication repository for the 2025 study that implements a two-stage framework:

1. **Asset filtering** — agglomerative clustering using eight distance-feature
   specifications (TDA-based: AWD, APL; conventional: correlation, Euclidean) ×
   two feature sets (indicator-only, sentiment-enhanced).
2. **Dynamic Rebalancing Mean-Variance (DRMV) optimisation** — mixed-integer
   quadratic programme with long/short positions, binary activation variables,
   and proportional transaction costs, solved with CVXPY + SCIP/HiGHS.

Two experiments are implemented:

| ID | Period | Purpose |
|----|--------|---------|
| `exp_1` | 2025-01-01 – 2025-12-31 | Main-sample replication |
| `exp_2` | 2025-10-01 – 2026-03-31 | Stress-period (conflict window) robustness |

---

## Repository layout

```
.
├── data/
│   └── fetch_market_data.py       # Massive REST API data loader
├── src/
│   ├── indicators.py              # RSI (L=10), SO, MACD
│   ├── sentiment.py               # FinBERT headline scoring + daily aggregation
│   ├── tda_distances.py           # AWD (Wasserstein) and APL (landscape) distances
│   ├── conventional_distances.py  # Correlation-based and Euclidean distances
│   ├── clustering.py              # Agglomerative clustering + asset selection (top 10%)
│   ├── optimizer.py               # DRMV MIQP solver (CVXPY) + lambda tuning
│   ├── backtest.py                # Rolling-window engine + retention mechanism
│   └── metrics.py                 # Sharpe, STARR, Sterling, VaR/CVaR, drawdowns
├── exp/
│   ├── exp_1_main_sample.py       # Experiment 1 runner
│   └── exp_2_stress_period.py     # Experiment 2 runner
├── tests/
│   ├── test_indicators.py
│   ├── test_clustering.py
│   ├── test_tda.py
│   ├── test_conventional_distances.py
│   ├── test_optimizer.py
│   └── test_metrics.py
├── results/
│   ├── RESULTS.md                 # Experiment summary (auto-generated)
│   ├── exp1_metrics.json
│   ├── exp1_clustering_diagnostics.json
│   ├── exp1_stat_tests.json
│   ├── exp1_cumret_*.png
│   ├── exp2_metrics.json
│   ├── exp2_stat_tests.json
│   └── exp2_cumret_stress.png
└── AGENTS.md
```

---

## Setup

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
            cvxpy ta ripser persim gudhi transformers torch sentencepiece \
            quantstats massive
```

Set the Massive API key:

```bash
export MASSIVE_TOKEN="<your-key>"
```

---

## Running experiments

```bash
# Full run (real Massive data required)
python exp/exp_1_main_sample.py
python exp/exp_2_stress_period.py

# Fast smoke-test with synthetic mock data (no API key needed)
FAST_RUN=1 python exp/exp_1_main_sample.py
FAST_RUN=1 python exp/exp_2_stress_period.py
```

---

## Tests

```bash
pytest tests/ -v
```

126 unit tests covering every module; all pass in < 5 s on a standard laptop.

---

## Key implementation decisions

| Choice | Decision | Rationale |
|--------|----------|-----------|
| RSI lookback | L = 10 | Appendix formula overrides narrative 14-day convention |
| SO formula | Same-day H/L (no rolling window) | Exact printed formula from paper appendix |
| MACD | EMA₁₂, EMA₂₆, signal M = 9 | Paper-specified parameters |
| TDA norm order p | p = 2 (L² / Wasserstein-2) | Standard choice; documented assumption |
| Homology aggregation | Average over r = 0 and r = 1 | Symmetric, conservative; documented assumption |
| Missing-sentiment days | Zero-imputation | Reproducible; no forward leakage |
| D1 / D2 split | D1 = ⌊0.7 × T1⌋, D2 = T1 − D1 | 70 / 30 in-sample train-validate split |
| Mark-to-market drift | Applied before each rebalance | Realistic; standard industry convention |
| Risk-free rate | r_f = 0 | Paper does not specify an alternative |
| Transaction cost rates | p₁ = p₂ = p₃ = p₄ = 0.0025 | Exact paper parameters |
| Clustering selection ρ | 0.10 (top 10 % per cluster) | Exact paper parameter |
| Lambda candidates | {0.2, 0.4, 0.6, 0.8, 1.0} | Exact paper grid |
| MIQP solver | CVXPY with SCIP / HiGHS | Open-source; falls back gracefully |
