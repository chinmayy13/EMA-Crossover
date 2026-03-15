"""
metrics.py
----------
Performance metric calculations for strategy evaluation.

All annualisation uses 252 trading days per year (equity markets).
For crypto tickers (24/7 markets) pass trading_days=365 where relevant.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


TRADING_DAYS: int = 252

# Approximate annualised risk-free rate (update periodically).
# Using a conservative 4.5% to reflect recent US Fed Funds rate environment.
DEFAULT_RISK_FREE_RATE: float = 0.045


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns (net of costs).
    risk_free_rate : float
        Annualised risk-free rate (default 4.5%).
    trading_days : int
        Number of trading days per year.

    Returns
    -------
    float
        Annualised Sharpe Ratio, or NaN if standard deviation is zero.
    """
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    std = returns.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float((excess.mean() / std) * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sortino Ratio.

    Like Sharpe but penalises only downside volatility, making it more
    appropriate for long-only strategies with positive skew.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    risk_free_rate : float
        Annualised risk-free rate.
    trading_days : int
        Number of trading days per year.

    Returns
    -------
    float
        Annualised Sortino Ratio, or NaN if downside deviation is zero.
    """
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    downside = returns[returns < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std) or len(downside) == 0:
        return float("nan")
    return float((excess.mean() / downside_std) * np.sqrt(trading_days))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (fraction, negative value).

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative portfolio value over time.

    Returns
    -------
    float
        Most negative drawdown fraction (e.g. -0.25 = -25%).
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def total_return(equity_curve: pd.Series) -> float:
    """
    Total return over the full period (fraction).

    Returns
    -------
    float
        e.g. 0.50 = +50%, -0.20 = -20%.
    """
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def cagr(equity_curve: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """
    Compound Annual Growth Rate.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative portfolio value.
    trading_days : int
        Trading days per year used to convert bar count to years.

    Returns
    -------
    float
        CAGR as a fraction, e.g. 0.12 = 12% p.a.
    """
    n_bars = len(equity_curve)
    if n_bars < 2:
        return float("nan")
    years = n_bars / trading_days
    end_val = equity_curve.iloc[-1]
    start_val = equity_curve.iloc[0]
    if start_val <= 0:
        return float("nan")
    return float((end_val / start_val) ** (1.0 / years) - 1)


def calmar_ratio(
    equity_curve: pd.Series,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|.

    Measures return per unit of maximum drawdown risk.

    Returns
    -------
    float
        Calmar Ratio, or NaN if max drawdown is zero.
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(equity_curve, trading_days) / abs(mdd))


def annualised_volatility(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Annualised standard deviation of daily returns.

    Returns
    -------
    float
        Volatility as a fraction, e.g. 0.18 = 18% p.a.
    """
    return float(returns.std() * np.sqrt(trading_days))


def rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> pd.Series:
    """
    Rolling annualised Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    window : int
        Look-back window in bars (default 126 ~ 6 months).
    risk_free_rate : float
        Annualised risk-free rate.
    trading_days : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Rolling Sharpe values.
    """
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    roll_mean = excess.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Statistical significance (permutation test)
# ---------------------------------------------------------------------------

def permutation_test(
    strategy_returns: pd.Series,
    n_permutations: int = 1000,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    random_state: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric permutation test for the Sharpe Ratio.

    Null hypothesis: the observed Sharpe is no better than a strategy that
    randomly shuffles the daily returns (i.e., no exploitable pattern).

    The p-value is the fraction of shuffled Sharpes that are >= the observed
    Sharpe.  A low p-value (e.g. < 0.05) provides statistical evidence that
    the strategy's risk-adjusted return is not due to chance.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily net returns from the backtest.
    n_permutations : int
        Number of random shuffles (default 1000).
    risk_free_rate : float
        Annualised risk-free rate used in Sharpe calculation.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    observed_sharpe : float
        Sharpe Ratio of the actual strategy.
    p_value : float
        Fraction of permuted Sharpes >= observed_sharpe.
    sharpe_95th_pct : float
        95th percentile of the null distribution (Sharpe threshold for
        significance at the 5% level).
    """
    rng = np.random.default_rng(random_state)
    returns_arr = strategy_returns.dropna().values

    observed_sharpe = sharpe_ratio(
        pd.Series(returns_arr), risk_free_rate=risk_free_rate
    )

    null_sharpes = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(returns_arr)
        null_sharpes[i] = sharpe_ratio(
            pd.Series(shuffled), risk_free_rate=risk_free_rate
        )

    p_value = float((null_sharpes >= observed_sharpe).mean())
    sharpe_95th = float(np.percentile(null_sharpes, 95))

    return observed_sharpe, p_value, sharpe_95th


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> list:
    """
    Generate train/test index pairs for walk-forward (time-series) validation.

    Unlike k-fold cross-validation, each fold respects temporal ordering:
    the training set always precedes the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Full backtest DataFrame.
    n_splits : int
        Number of folds (default 5).

    Returns
    -------
    list of (train_df, test_df) tuples
        Each tuple is a non-overlapping temporal split of *df*.
    """
    n = len(df)
    fold_size = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train = df.iloc[: i * fold_size]
        test = df.iloc[i * fold_size: (i + 1) * fold_size]
        if len(test) > 0:
            splits.append((train.copy(), test.copy()))
    return splits


# ---------------------------------------------------------------------------
# Aggregate reporter
# ---------------------------------------------------------------------------

def compute_all_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    run_permutation_test: bool = False,
    n_permutations: int = 1000,
) -> Dict[str, object]:
    """
    Compute all performance metrics and return as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame containing 'Strategy_Return', 'Market_Return',
        'Equity_Curve', and 'BnH_Equity' columns.
    risk_free_rate : float
        Annualised risk-free rate (default 4.5%).
    run_permutation_test : bool
        If True, also run the permutation significance test (adds ~1-2 seconds).
    n_permutations : int
        Number of permutations for the significance test.

    Returns
    -------
    dict
        Metric names mapped to their computed values.
    """
    returns = df["Strategy_Return"].dropna()
    market_returns = df["Market_Return"].dropna()
    equity = df["Equity_Curve"]
    bnh_equity = df["BnH_Equity"]

    metrics: Dict[str, object] = {
        "Sharpe":          sharpe_ratio(returns, risk_free_rate),
        "Sortino":         sortino_ratio(returns, risk_free_rate),
        "Calmar":          calmar_ratio(equity),
        "Max_Drawdown":    max_drawdown(equity),
        "Total_Return":    total_return(equity),
        "CAGR":            cagr(equity),
        "Volatility":      annualised_volatility(returns),
        "BnH_Total_Return": total_return(bnh_equity),
        "BnH_Sharpe":      sharpe_ratio(market_returns, risk_free_rate),
        "Risk_Free_Rate":  risk_free_rate,
        "N_Bars":          len(returns),
        "N_Trades":        int((df["Signal"] != 0).sum()) if "Signal" in df.columns else None,
    }

    if run_permutation_test:
        obs_sr, pval, sr_95 = permutation_test(
            returns,
            n_permutations=n_permutations,
            risk_free_rate=risk_free_rate,
        )
        metrics["Permutation_P_Value"] = pval
        metrics["Null_Sharpe_95pct"] = sr_95
        metrics["Is_Significant_5pct"] = pval < 0.05

    return metrics