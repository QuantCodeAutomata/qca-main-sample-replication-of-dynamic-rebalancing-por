"""DRMV formulation per paper; binary variables u_t,v_t via SCIP MIQP solver"""

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Transaction cost rates (proportional, applied to trade size)
P1 = 0.0025  # long purchase cost rate
P2 = 0.0025  # long sale cost rate
P3 = 0.0025  # short increase cost rate
P4 = 0.0025  # short decrease cost rate

# Portfolio weight bounds for individual assets
M_PLUS = 0.20   # max long weight per asset
M_MINUS = 0.05  # max short weight per asset
m_PLUS = 0.05   # min long weight if asset is held long (u_i = 1)
m_MINUS = 0.01  # min short weight if asset is held short (v_i = 1)

# Margin requirement: fraction of short notional held as collateral
K_MARGIN = 1.0  # trading margin (1.0 = full collateral for short positions)

# Risk-aversion grid searched via D1/D2 cross-validation inside tune_lambda
LAMBDA_CANDIDATES = [0.2, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_mean_return(returns_df: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """Return mean daily return vector of length n for the given tickers."""
    return returns_df[tickers].mean().values.astype(float)


def compute_covariance(returns_df: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """Return n x n sample covariance matrix, regularised to be positive semi-definite.

    Regularisation: C = C + 1e-6 * I ensures strict PSD required by cp.quad_form.
    """
    C = returns_df[tickers].cov().values.astype(float)
    C = (C + C.T) / 2                          # symmetrise before regularising
    C = C + 1e-6 * np.eye(C.shape[0])
    return C


# ---------------------------------------------------------------------------
# Core MIQP solver
# ---------------------------------------------------------------------------

def _build_and_solve(
    r_bar: np.ndarray,
    C: np.ndarray,
    w_prev_plus: np.ndarray,
    w_prev_minus: np.ndarray,
    lam: float,
    relax_binaries: bool = False,
    solver: str = "SCIP",
) -> cp.Problem:
    """Build the DRMV MIQP and return the solved cp.Problem object.

    When *relax_binaries* is True the boolean activation variables u, v are
    replaced by continuous variables in [0, 1], converting the MIQP into a
    standard QP suitable for CLARABEL.
    """
    n = len(r_bar)

    w_plus = cp.Variable(n, nonneg=True, name="w_plus")
    w_minus = cp.Variable(n, nonneg=True, name="w_minus")

    if relax_binaries:
        u = cp.Variable(n, name="u")   # relaxed: bounded below/above by constraints
        v = cp.Variable(n, name="v")
    else:
        u = cp.Variable(n, boolean=True, name="u")
        v = cp.Variable(n, boolean=True, name="v")

    l_plus = cp.Variable(n, nonneg=True, name="l_plus")    # long purchases
    l_minus = cp.Variable(n, nonneg=True, name="l_minus")  # long sales
    s_plus = cp.Variable(n, nonneg=True, name="s_plus")    # short increases
    s_minus = cp.Variable(n, nonneg=True, name="s_minus")  # short decreases

    x = w_plus - w_minus  # net portfolio weight vector

    objective = cp.Maximize(
        r_bar @ x
        - lam * cp.quad_form(x, C)
        - P1 * cp.sum(l_plus)
        - P2 * cp.sum(l_minus)
        - P3 * cp.sum(s_plus)
        - P4 * cp.sum(s_minus)
    )

    constraints = [
        # Weight update identities
        w_plus == w_prev_plus + l_plus - l_minus,
        w_minus == w_prev_minus + s_plus - s_minus,
        # Big-M bounds: force w_plus_i in {0} ∪ [m_PLUS, M_PLUS]
        w_plus >= m_PLUS * u,
        w_plus <= M_PLUS * u,
        # Big-M bounds: force w_minus_i in {0} ∪ [m_MINUS, M_MINUS]
        w_minus >= m_MINUS * v,
        w_minus <= M_MINUS * v,
        # No simultaneous long and short in the same asset (component-wise)
        u + v <= np.ones(n),
        # Budget: long + short margin + all transaction costs == total capital
        cp.sum(
            w_plus
            + K_MARGIN * w_minus
            + P1 * l_plus
            + P2 * l_minus
            + P3 * s_plus
            + P4 * s_minus
        ) == 1.0,
    ]

    if relax_binaries:
        constraints += [u >= 0, u <= 1, v >= 0, v <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)
    return problem


def solve_drmv(
    r_bar: np.ndarray,
    C: np.ndarray,
    w_prev_plus: np.ndarray,
    w_prev_minus: np.ndarray,
    lam: float,
    solver: str = "SCIP",
) -> dict:
    """Solve the DRMV mixed-integer quadratic programme.

    Parameters
    ----------
    r_bar : (n,) mean return vector.
    C     : (n,n) PSD covariance matrix.
    w_prev_plus  : (n,) prior long weights (mark-to-market drifted).
    w_prev_minus : (n,) prior short weights (mark-to-market drifted).
    lam   : risk-aversion coefficient.
    solver: primary solver; falls back to CLARABEL with relaxed binaries on failure.

    Returns
    -------
    dict with keys: status, w_plus, w_minus, x, l_plus, l_minus, s_plus,
    s_minus, transaction_cost, objective.  On failure returns zero-weight
    fallback.
    """
    n = len(r_bar)
    C = (C + C.T) / 2  # guarantee symmetry before quad_form

    _zero_result = {
        "status": "infeasible",
        "w_plus": np.zeros(n),
        "w_minus": np.zeros(n),
        "x": np.zeros(n),
        "l_plus": np.zeros(n),
        "l_minus": np.zeros(n),
        "s_plus": np.zeros(n),
        "s_minus": np.zeros(n),
        "transaction_cost": 0.0,
        "objective": 0.0,
    }

    # ---- primary solve: SCIP MIQP ----------------------------------------
    try:
        problem = _build_and_solve(r_bar, C, w_prev_plus, w_prev_minus, lam,
                                   relax_binaries=False, solver=solver)
        status = problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise cp.SolverError(f"SCIP returned status '{status}'")

        prob_vars = {v.name(): v for v in problem.variables()}
        w_plus_val  = np.maximum(prob_vars["w_plus"].value,  0.0)
        w_minus_val = np.maximum(prob_vars["w_minus"].value, 0.0)
        l_plus_val  = np.maximum(prob_vars["l_plus"].value,  0.0)
        l_minus_val = np.maximum(prob_vars["l_minus"].value, 0.0)
        s_plus_val  = np.maximum(prob_vars["s_plus"].value,  0.0)
        s_minus_val = np.maximum(prob_vars["s_minus"].value, 0.0)

        tc = (P1 * l_plus_val.sum() + P2 * l_minus_val.sum()
              + P3 * s_plus_val.sum() + P4 * s_minus_val.sum())

        return {
            "status": status,
            "w_plus": w_plus_val,
            "w_minus": w_minus_val,
            "x": w_plus_val - w_minus_val,
            "l_plus": l_plus_val,
            "l_minus": l_minus_val,
            "s_plus": s_plus_val,
            "s_minus": s_minus_val,
            "transaction_cost": float(tc),
            "objective": float(problem.value),
        }

    except Exception as exc:
        logger.warning(
            "SCIP solver failed (%s); retrying with CLARABEL + relaxed binaries.", exc
        )

    # ---- fallback: CLARABEL with relaxed (continuous) binary variables -----
    try:
        problem = _build_and_solve(r_bar, C, w_prev_plus, w_prev_minus, lam,
                                   relax_binaries=True, solver="CLARABEL")
        status = problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise cp.SolverError(f"CLARABEL returned status '{status}'")

        prob_vars = {v.name(): v for v in problem.variables()}
        w_plus_val  = np.maximum(prob_vars["w_plus"].value,  0.0)
        w_minus_val = np.maximum(prob_vars["w_minus"].value, 0.0)
        l_plus_val  = np.maximum(prob_vars["l_plus"].value,  0.0)
        l_minus_val = np.maximum(prob_vars["l_minus"].value, 0.0)
        s_plus_val  = np.maximum(prob_vars["s_plus"].value,  0.0)
        s_minus_val = np.maximum(prob_vars["s_minus"].value, 0.0)

        tc = (P1 * l_plus_val.sum() + P2 * l_minus_val.sum()
              + P3 * s_plus_val.sum() + P4 * s_minus_val.sum())

        return {
            "status": f"relaxed/{status}",
            "w_plus": w_plus_val,
            "w_minus": w_minus_val,
            "x": w_plus_val - w_minus_val,
            "l_plus": l_plus_val,
            "l_minus": l_minus_val,
            "s_plus": s_plus_val,
            "s_minus": s_minus_val,
            "transaction_cost": float(tc),
            "objective": float(problem.value),
        }

    except Exception as exc:
        logger.error("CLARABEL fallback also failed (%s); returning zero weights.", exc)
        return _zero_result


# ---------------------------------------------------------------------------
# Lambda tuning via D1/D2 cross-validation
# ---------------------------------------------------------------------------

def tune_lambda(
    returns_is: pd.DataFrame,
    tickers: list[str],
    w_prev_plus: np.ndarray,
    w_prev_minus: np.ndarray,
    d1_frac: float = 0.80,
) -> float:
    """Select risk-aversion lambda via Sharpe on a held-out sub-period.

    D1/D2 split: 80/20 of T1 (e.g., 50/13 for T1=63); chosen assumption,
    not in paper.

    Parameters
    ----------
    returns_is   : in-sample returns DataFrame (T1 rows × n columns).
    tickers      : asset identifiers matching columns of returns_is.
    w_prev_plus  : prior long weights for the start of returns_is.
    w_prev_minus : prior short weights for the start of returns_is.
    d1_frac      : fraction of in-sample rows assigned to D1 (default 0.80).

    Returns
    -------
    float: lambda from LAMBDA_CANDIDATES with highest D2 Sharpe, or 0.5 if
    all candidates produce the same (e.g., all-infeasible) performance.
    """
    T = len(returns_is)
    d1_end = max(2, int(T * d1_frac))   # at least 2 rows for covariance

    returns_d1 = returns_is.iloc[:d1_end]
    returns_d2 = returns_is.iloc[d1_end:]

    if len(returns_d2) == 0:
        logger.warning("D2 is empty; returning default lambda 0.5.")
        return 0.5

    r_bar_d1 = compute_mean_return(returns_d1, tickers)
    C_d1     = compute_covariance(returns_d1, tickers)

    sharpes: dict[float, float] = {}
    for lam in LAMBDA_CANDIDATES:
        result = solve_drmv(r_bar_d1, C_d1, w_prev_plus, w_prev_minus, lam)
        if result["status"] in ("infeasible",) or not np.any(result["x"] != 0):
            sharpes[lam] = -np.inf
            continue
        port_returns = compute_portfolio_returns(returns_d2, tickers, result["x"])
        std = port_returns.std()
        sharpes[lam] = float(port_returns.mean() / std) if std > 1e-12 else -np.inf

    # Guard: all identical (e.g., all -inf) → return midpoint 0.5
    unique_vals = set(sharpes.values())
    if len(unique_vals) == 1:
        logger.warning(
            "All lambda candidates yielded identical D2 Sharpe; returning default 0.5."
        )
        return 0.5

    best_lam = max(sharpes, key=sharpes.__getitem__)
    logger.debug("tune_lambda: best lambda=%.2f (Sharpe=%.4f)", best_lam, sharpes[best_lam])
    return best_lam


# ---------------------------------------------------------------------------
# Portfolio return evaluation
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    returns_oos: pd.DataFrame,
    tickers: list[str],
    x: np.ndarray,
) -> pd.Series:
    """Compute daily portfolio returns for the OOS period.

    Parameters
    ----------
    returns_oos : DataFrame of daily asset returns (T_oos × n).
    tickers     : asset identifiers (must match columns of returns_oos).
    x           : net weight vector (long - short) of length n.

    Returns
    -------
    pd.Series of portfolio daily returns indexed by the OOS dates.
    """
    weights = pd.Series(x, index=tickers)
    port_returns = returns_oos[tickers].dot(weights)
    return port_returns


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s %(name)s: %(message)s")

    rng = np.random.default_rng(42)
    N, T = 5, 120

    # Synthetic price series → daily returns
    prices = pd.DataFrame(
        np.cumprod(1 + rng.normal(0.0005, 0.01, size=(T, N)), axis=0),
        columns=[f"A{i}" for i in range(N)],
    )
    rets = prices.pct_change().dropna()
    tickers = list(prices.columns)

    r_bar = compute_mean_return(rets, tickers)
    C     = compute_covariance(rets, tickers)

    w0_plus  = np.zeros(N)
    w0_minus = np.zeros(N)

    print("Mean returns:", np.round(r_bar, 6))
    print("Covariance diagonal:", np.round(np.diag(C), 8))

    result = solve_drmv(r_bar, C, w0_plus, w0_minus, lam=0.5)
    print("\nsolve_drmv result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    best_lam = tune_lambda(rets, tickers, w0_plus, w0_minus)
    print(f"\nbest lambda from tune_lambda: {best_lam}")

    result2 = solve_drmv(r_bar, C, w0_plus, w0_minus, lam=best_lam)
    port_rets = compute_portfolio_returns(rets.tail(20), tickers, result2["x"])
    print(f"\nOOS portfolio return stats (20 days):")
    print(f"  mean={port_rets.mean():.6f}  std={port_rets.std():.6f}")
    print(f"  Sharpe (daily)={port_rets.mean()/port_rets.std():.4f}")
