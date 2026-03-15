"""
backtester.py
-------------
Vectorised backtest engine for the EMA crossover strategy.
"""

import numpy as np
import pandas as pd


INITIAL_CAPITAL: float = 10_000.0


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Vectorised backtest: fully invested when Position=1, cash when Position=0.

    Look-ahead bias prevention
    --------------------------
    Signals generated at bar *t* are acted on at bar *t+1* via
    ``Position.shift(1)``.  This simulates: signal fires at today's close,
    order executes at tomorrow's open (approximated as next close).

    Transaction cost model
    ----------------------
    A round-trip cost of ``transaction_cost`` (fraction of capital deployed)
    is charged on every position change.  The cost is applied as a fractional
    deduction from the strategy return on the bar a trade occurs:

        Strategy_Return[t] -= |delta_position[t]| * transaction_cost

    For a binary 0/1 strategy ``|delta_position|`` is always 1 on a trade bar,
    so the deduction equals ``transaction_cost`` (e.g. 0.001 = 0.1 %).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' and 'Position' columns.
    initial_capital : float
        Starting cash balance.
    transaction_cost : float
        One-way cost per unit of position change (fraction, e.g. 0.001 = 0.1%).

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with columns:
        - 'Market_Return'    : daily pct change of the asset
        - 'Strategy_Return'  : daily pct change earned by the strategy
        - 'Gross_Return'     : strategy return before transaction costs
        - 'Transaction_Cost' : cost deducted on trade bars
        - 'Equity_Curve'     : cumulative portfolio value ($)
        - 'BnH_Equity'       : buy-and-hold equity curve for comparison
        - 'Drawdown'         : running drawdown from peak equity (fraction)
    """
    required = {"Close", "Position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[Backtester] Missing columns: {missing}")

    out = df.copy()

    # ── Daily asset return ──────────────────────────────────────────────
    out["Market_Return"] = out["Close"].pct_change()

    # ── Gross strategy return (no costs) ───────────────────────────────
    # Shift position by 1 bar: trade at bar t+1 after signal at bar t
    out["Gross_Return"] = out["Market_Return"] * out["Position"].shift(1)

    # ── Transaction cost ───────────────────────────────────────────────
    # Charge cost proportional to the SIZE of the position change.
    # For binary 0/1 sizing this equals transaction_cost on every entry/exit.
    # Using .shift(1) here because the position change that matters is the
    # change in the EXECUTED position (already shifted above).
    executed_position = out["Position"].shift(1)
    out["Transaction_Cost"] = executed_position.diff().abs() * transaction_cost

    # ── Net strategy return ────────────────────────────────────────────
    out["Strategy_Return"] = out["Gross_Return"] - out["Transaction_Cost"]

    # Drop the first row(s) that inevitably have NaN from pct_change / shift
    out = out.dropna(subset=["Strategy_Return", "Market_Return"])

    # ── Equity curves ──────────────────────────────────────────────────
    out["Equity_Curve"] = initial_capital * (1 + out["Strategy_Return"]).cumprod()
    out["BnH_Equity"] = initial_capital * (1 + out["Market_Return"]).cumprod()

    # ── Drawdown ───────────────────────────────────────────────────────
    rolling_max = out["Equity_Curve"].cummax()
    out["Drawdown"] = (out["Equity_Curve"] - rolling_max) / rolling_max

    return out
