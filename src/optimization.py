"""
optimization.py
---------------
Grid-search parameter optimisation for the EMA crossover strategy.

IMPORTANT — In-sample vs out-of-sample
---------------------------------------
``grid_search`` evaluates all parameter combinations on the FULL dataset.
Results are therefore in-sample and subject to look-ahead / data-snooping
bias.  Use ``walk_forward_optimize`` or manually split your data into a
training window and a held-out test window before drawing any conclusions
about out-of-sample performance.
"""

import itertools
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from src.indicators import add_emas
from src.signals import generate_signals
from src.backtester import run_backtest
from src.metrics import (
    sharpe_ratio,
    total_return,
    cagr,
    DEFAULT_RISK_FREE_RATE,
)


# ---------------------------------------------------------------------------
# Grid search (in-sample)
# ---------------------------------------------------------------------------

def grid_search(
    df: pd.DataFrame,
    short_range: range = range(5, 51, 5),
    long_range: range = range(20, 201, 10),
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    transaction_cost: float = 0.001,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Exhaustive in-sample grid search over (short_ema, long_ema) pairs.

    .. warning::
        Results are **in-sample only**.  Always validate best parameters on
        a held-out test period or with :func:`walk_forward_optimize` before
        drawing conclusions.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV DataFrame (will be used as-is; split beforehand if needed).
    short_range : range
        Short EMA spans to search.
    long_range : range
        Long EMA spans to search.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.
    transaction_cost : float
        One-way transaction cost fraction.
    verbose : bool
        Print progress every 50 combinations.

    Returns
    -------
    pd.DataFrame
        All valid (short, long) combinations ranked by Sharpe, descending.
        Columns: Short_EMA, Long_EMA, Sharpe, Total_Return, CAGR.
    """
    combos = [(s, l) for s, l in itertools.product(short_range, long_range) if s < l]
    total = len(combos)
    results = []

    for done, (short, long) in enumerate(combos, start=1):
        try:
            data = add_emas(df, short, long)
            data = generate_signals(data)
            data = run_backtest(data, transaction_cost=transaction_cost)

            sr = sharpe_ratio(data["Strategy_Return"], risk_free_rate)
            tr = total_return(data["Equity_Curve"])
            cg = cagr(data["Equity_Curve"])

        except Exception as exc:
            if verbose:
                print(f"[Optimizer] Error for ({short},{long}): {exc}")
            sr, tr, cg = np.nan, np.nan, np.nan

        results.append(
            {
                "Short_EMA":    short,
                "Long_EMA":     long,
                "Sharpe":       sr,
                "Total_Return": tr,
                "CAGR":         cg,
            }
        )

        if verbose and done % 50 == 0:
            print(f"[Optimizer] {done}/{total} combinations tested ...")

    if verbose:
        print(f"[Optimizer] Done. {done}/{total} combinations evaluated.")

    result_df = pd.DataFrame(results).dropna(subset=["Sharpe"])
    return result_df.sort_values("Sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward optimisation (out-of-sample)
# ---------------------------------------------------------------------------

def walk_forward_optimize(
    df: pd.DataFrame,
    short_range: range = range(5, 51, 5),
    long_range: range = range(20, 201, 10),
    n_splits: int = 5,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Walk-forward optimisation: train on a window, test on the next window.

    Each fold:
    1. Runs a full grid search on the TRAINING portion.
    2. Identifies the best (short, long) pair by in-sample Sharpe.
    3. Evaluates those parameters on the out-of-sample TEST portion.

    The returned DataFrame shows the out-of-sample performance of the
    best in-sample parameters for each fold — a realistic estimate of
    live-trading performance.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV DataFrame.
    short_range, long_range : range
        Parameter search spaces.
    n_splits : int
        Number of walk-forward folds (default 5).
    risk_free_rate : float
        Annualised risk-free rate.
    transaction_cost : float
        One-way transaction cost fraction.

    Returns
    -------
    pd.DataFrame
        One row per fold with columns:
        Fold, Train_Start, Train_End, Test_Start, Test_End,
        Best_Short, Best_Long,
        InSample_Sharpe, OOS_Sharpe, OOS_Total_Return, OOS_CAGR.
    """
    n = len(df)
    fold_size = n // (n_splits + 1)
    fold_results = []

    print(f"[WalkForward] Running {n_splits} folds ...")

    for fold in range(1, n_splits + 1):
        train_end_idx = fold * fold_size
        test_end_idx = min((fold + 1) * fold_size, n)

        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[train_end_idx:test_end_idx].copy()

        if len(test_df) < 30:
            print(f"[WalkForward] Fold {fold}: test set too small, skipping.")
            continue

        # In-sample optimisation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = grid_search(
                train_df,
                short_range=short_range,
                long_range=long_range,
                risk_free_rate=risk_free_rate,
                transaction_cost=transaction_cost,
                verbose=False,
            )

        if opt.empty:
            print(f"[WalkForward] Fold {fold}: optimisation returned no results.")
            continue

        best_short = int(opt.iloc[0]["Short_EMA"])
        best_long = int(opt.iloc[0]["Long_EMA"])
        is_sharpe = float(opt.iloc[0]["Sharpe"])

        # Out-of-sample evaluation
        try:
            oos_data = add_emas(test_df, best_short, best_long)
            oos_data = generate_signals(oos_data)
            oos_data = run_backtest(oos_data, transaction_cost=transaction_cost)

            oos_sr = sharpe_ratio(oos_data["Strategy_Return"], risk_free_rate)
            oos_tr = total_return(oos_data["Equity_Curve"])
            oos_cg = cagr(oos_data["Equity_Curve"])
        except Exception as exc:
            print(f"[WalkForward] Fold {fold} OOS error: {exc}")
            oos_sr, oos_tr, oos_cg = np.nan, np.nan, np.nan

        fold_results.append(
            {
                "Fold":              fold,
                "Train_Start":       train_df.index[0].date(),
                "Train_End":         train_df.index[-1].date(),
                "Test_Start":        test_df.index[0].date(),
                "Test_End":          test_df.index[-1].date(),
                "Best_Short":        best_short,
                "Best_Long":         best_long,
                "InSample_Sharpe":   round(is_sharpe, 3),
                "OOS_Sharpe":        round(oos_sr, 3) if not np.isnan(oos_sr) else np.nan,
                "OOS_Total_Return":  round(oos_tr, 4) if not np.isnan(oos_tr) else np.nan,
                "OOS_CAGR":          round(oos_cg, 4) if not np.isnan(oos_cg) else np.nan,
            }
        )

        print(
            f"[WalkForward] Fold {fold}: best=({best_short},{best_long}) "
            f"IS Sharpe={is_sharpe:.3f}  OOS Sharpe={oos_sr:.3f}"
        )

    print("[WalkForward] Done.")
    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_params(opt_df: pd.DataFrame) -> Tuple[int, int]:
    """
    Return the (short, long) EMA pair with the highest Sharpe Ratio.

    Parameters
    ----------
    opt_df : pd.DataFrame
        Output of :func:`grid_search`.

    Returns
    -------
    Tuple[int, int]
        (short_ema, long_ema).
    """
    if opt_df.empty:
        raise ValueError("[Optimizer] Optimisation result is empty.")
    row = opt_df.iloc[0]
    return int(row["Short_EMA"]), int(row["Long_EMA"])


def build_heatmap_pivot(opt_df: pd.DataFrame, metric: str = "Sharpe") -> pd.DataFrame:
    """
    Reshape optimisation results into a 2-D pivot table for heatmap plotting.

    Parameters
    ----------
    opt_df : pd.DataFrame
        Output of :func:`grid_search`.
    metric : str
        Column name to use as values (e.g. 'Sharpe', 'Total_Return').

    Returns
    -------
    pd.DataFrame
        Pivot table with Short_EMA as rows and Long_EMA as columns.
    """
    return opt_df.pivot_table(
        index="Short_EMA",
        columns="Long_EMA",
        values=metric,
        aggfunc="mean",
    )