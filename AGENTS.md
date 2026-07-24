# AGENTS.md – Project Knowledge Base

## Project overview
Dynamic Rebalancing Mean-Variance (DRMV) portfolio optimizer with clustering-based
asset selection.  Paper reference: DRMV formulation with binary activation variables
u_t (long) and v_t (short) solved via SCIP MIQP.

## Repository layout
```
src/
  optimizer.py            – DRMV MIQP solver, statistical helpers, lambda tuning
  backtest.py             – Rolling backtest engine + benchmark strategies
  indicators.py           – (to be created) compute_all_indicators(close_df, ohlcv_dict, tickers)
  clustering.py           – (to be created) cluster_assets(dist_matrix, tickers, alpha)
  conventional_distances.py – (to be created) compute_awd_matrix, compute_ald_matrix
  tda_distances.py        – (to be created) TDADistanceComputer class
```

## Key constants (optimizer.py)
| Symbol        | Value  | Meaning                                    |
|---------------|--------|--------------------------------------------|
| P1=P2=P3=P4   | 0.0025 | Proportional transaction cost rates        |
| M_PLUS        | 0.20   | Max long weight per asset                  |
| M_MINUS       | 0.05   | Max short weight per asset                 |
| m_PLUS        | 0.05   | Min long weight when asset is activated    |
| m_MINUS       | 0.01   | Min short weight when asset is activated   |
| K_MARGIN      | 1.0    | Full collateral for short positions        |
| LAMBDA_CANDIDATES | [0.2,0.4,0.6,0.8,1.0] | Risk-aversion grid        |

## CLUSTERING_SPECS (backtest.py)
Eight specs: AWDS/APLS/ACS/AES (sentiment features) + AWDI/APLI/ACI/AEI (indicator
features).  Each combines a distance metric (awd/ald/correlation/euclidean) with a
feature source.

## Solver chain
1. Primary: SCIP MIQP (boolean u, v).
2. Fallback: CLARABEL with relaxed binaries (continuous u, v ∈ [0,1]).
3. If both fail: return zero-weight dict and log at ERROR.

## DRMV budget constraint
sum(w_plus + K_MARGIN*w_minus + P1*l_plus + P2*l_minus + P3*s_plus + P4*s_minus) == 1

## Rolling window notation
- T1 = in-sample length (e.g. 63 trading days)
- T2 = OOS / rebalancing interval
- M  = floor((len(dates) - T1) / T2)   windows executed

## Lambda tuning (D1/D2 split)
80/20 split of T1 (e.g. 50/13 for T1=63); chosen assumption, not in paper.
Falls back to lambda=0.5 when all candidates yield identical (or −inf) Sharpe.

## Feature panel shapes
- 'sentiment' → 3D panel: (T, N, F_sentiment)
- 'indicators' → 4D panel: (T, N, F_indicators)
Both stored as numpy ndarrays.

## Expected APIs for yet-to-be-created modules
- compute_all_indicators(close_df, ohlcv_dict, tickers) → dict[str, pd.DataFrame]
- cluster_assets(dist_matrix, tickers, alpha) → list/ndarray of integer cluster labels
- compute_awd_matrix(features_is, tickers) → (N, N) ndarray
- compute_ald_matrix(features_is, tickers) → (N, N) ndarray
- sentiment_panel: pd.DataFrame with MultiIndex columns (ticker, feature_name) OR
  dict {ticker: pd.DataFrame}

## Installed solvers (verified)
CLARABEL, HIGHS, OSQP, SCIP, SCIPY, SCS
numpy 1.26.4 | pandas 2.2.2 | cvxpy 1.7.5 | Python /usr/local/bin/python

## Confirmed module APIs (verified against actual implementations)

### src/optimizer.py
- `solve_drmv(r_bar, C, w_prev_plus, w_prev_minus, lam, solver="SCIP") -> dict`
  Returns dict with keys: status, w_plus, w_minus, x, l_plus, l_minus, s_plus, s_minus,
  transaction_cost, objective. NOT a tuple. Takes pre-computed r_bar (ndarray) and C (ndarray).
- `tune_lambda(returns_is, tickers, w_prev_plus, w_prev_minus, d1_frac=0.80) -> float`
  Takes a pd.DataFrame of returns (not raw returns matrix). Returns float from LAMBDA_CANDIDATES
  OR fallback 0.5 when all candidates tie (documented behaviour).
- `compute_mean_return(returns_df, tickers) -> ndarray`
- `compute_covariance(returns_df, tickers) -> ndarray` (regularised: C + 1e-6*I)
- `compute_portfolio_returns(returns_oos, tickers, x) -> pd.Series`

### src/indicators.py (class-based or functional – see actual file)
- Uses function-based API: `compute_rsi(...)`, `compute_stochastic_oscillator(...)`,
  `compute_macd(...)`, `standardize_features(...)`, `build_feature_panel(...)`
- RSI window default: 14; SO window default: 14; MACD fast/slow/signal: 12/26/9

### src/clustering.py (class-based API)
- Methods: `compute_nn_distances(...)`, `compute_threshold(alpha)`, `fit(...)`,
  `identify_singletons(...)`, `select_assets(rho=0.10)`, `compute_ari(...)`,
  `compute_jaccard(...)`
- alpha-th percentile of NN distances used as cluster threshold
- Singletons = clusters of size 1 (excluded from asset selection)
- select_assets: ceil(rho * n_c) per cluster

### src/tda_distances.py (class-based API: TDADistanceComputer)
- Constructor: `TDADistanceComputer(T1, L, maxdim=1, n_landscape_bins=50)`
- Methods: `compute_persistence_diagram(point_cloud)` → dict with 'H0', 'H1' keys
  `compute_all_diagrams(feature_matrix)` → list of dicts
  `compute_distance_matrix(feature_matrix)` → (N, N) symmetric ndarray
  `compute_awd(diag_a, diag_b)` → float (Wasserstein distance)
  `compute_landscape_vector(diag)` → flat ndarray (nonneg values)
  `n_subwindows` property → K = T1 - L + 1

### src/conventional_distances.py (function-based)
- `compute_distance_matrix(Z, method='correlation') -> ndarray` (dict-based return OR direct ndarray)
- `correlation_distance(Z_a, Z_b) -> float`; self-distance ~1.5e-8 (not exactly 0; use atol=1e-7)
- `euclidean_distance(Z_a, Z_b) -> float`; self-distance exactly 0.0

## Test suite status (as of last session)
Total: 126 tests, 126 passed (0 failed).

Files and test counts:
- tests/test_metrics.py          22 tests   (all pass)
- tests/test_indicators.py       17 tests   (all pass)
- tests/test_clustering.py       31 tests   (all pass)
- tests/test_conventional_distances.py  all pass
- tests/test_tda.py              20 tests   (all pass)
- tests/test_optimizer.py        13 tests   (all pass)

## Critical test-writing lessons
1. `correlation_distance(Z, Z)` returns ~1.5e-8 (not 0.0) — use `atol=1e-7` not exact equality.
2. `tune_lambda` returns 0.5 fallback when all candidates tie; tests must allow this value.
3. `solve_drmv` returns a **dict**, not a tuple — tests must unpack via `result['w_plus']`.
4. Budget constraint for optimizer is the FULL constraint including margin and TC, not just
   `sum(w_plus + w_minus) ≈ 1`. Test with the explicit formula.
5. test_optimizer fixtures need ≥120 days and positive drift (0.0008) to get non-trivial
   D2 splits in tune_lambda and avoid all-infeasible scenarios.

## Testing
Run smoke tests:
  python -m src.optimizer    (5-asset synthetic: ~5 s)
  python -m src.backtest     (8-asset synthetic: ~30 s, benchmarks only)

Run full test suite:
  python -m pytest tests/ -v   (126 tests, ~3 s)
