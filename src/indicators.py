"""
indicators.py
-------------
Technical indicator calculations (EMA and ATR).
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average using the standard recursive formula.

    Uses ``adjust=False`` so the result matches the classic EMA definition:
        EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1},
    where alpha = 2 / (span + 1).

    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    span : int
        Number of periods (window) for the EMA.

    Returns
    -------
    pd.Series
        EMA values aligned to *series* index.
    """
    if span < 1:
        raise ValueError(f"EMA span must be >= 1, got {span}.")
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder smoothing).

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ATR values aligned to *df* index.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_emas(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """
    Append EMA_short and EMA_long columns to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    short : int
        Short EMA span.  Must be strictly less than *long*.
    long : int
        Long EMA span.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with 'EMA_short' and 'EMA_long' columns added.
    """
    if short >= long:
        raise ValueError(
            f"short ({short}) must be strictly less than long ({long})."
        )

    out = df.copy()
    out["EMA_short"] = ema(df["Close"], short)
    out["EMA_long"] = ema(df["Close"], long)
    return out