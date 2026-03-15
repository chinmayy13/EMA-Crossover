"""
signals.py
----------
Trading signal generation from EMA crossover logic.
"""

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce BUY / SELL / HOLD signals from EMA crossover.

    Requires 'EMA_short' and 'EMA_long' columns (add via indicators.add_emas).

    Signal logic
    ------------
    * Position = 1  when EMA_short > EMA_long  (long / fully invested)
    * Position = 0  when EMA_short <= EMA_long  (flat / cash)

    A BUY  event fires on the bar that transitions 0 -> 1.
    A SELL event fires on the bar that transitions 1 -> 0.

    Parameters
    ----------
    df : pd.DataFrame
        Price + indicator DataFrame with 'EMA_short' and 'EMA_long' columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* extended with:
        - 'Position'  : 1 (long) or 0 (flat)
        - 'Signal'    : +1 (buy event), -1 (sell event), 0 (hold)
    """
    if "EMA_short" not in df.columns or "EMA_long" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'EMA_short' and 'EMA_long' columns. "
            "Run indicators.add_emas() first."
        )

    out = df.copy()

    # Raw position: 1 when short EMA is above long EMA, else 0
    out["Position"] = np.where(out["EMA_short"] > out["EMA_long"], 1, 0)

    # Signal = change in position (crossover event)
    out["Signal"] = out["Position"].diff().fillna(0).astype(int)

    return out


def get_trade_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract individual buy/sell trade events for analysis or plotting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Signal' and 'Close' columns.

    Returns
    -------
    pd.DataFrame
        Rows where a trade occurred, with columns:
        - 'Close'  : fill price (close of signal bar)
        - 'Signal' : +1 or -1
        - 'Type'   : 'BUY' or 'SELL'
    """
    if "Signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'Signal' column.")

    events = df[df["Signal"] != 0][["Close", "Signal"]].copy()
    events["Type"] = events["Signal"].map({1: "BUY", -1: "SELL"})
    return events
