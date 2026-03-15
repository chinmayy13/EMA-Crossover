"""
data_loader.py
--------------
Historical OHLCV data ingestion via yfinance with local CSV caching.
"""

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf
import os


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _read_cache(cache_path) -> pd.DataFrame:
    """
    Read a cached CSV that may have a 3-row yfinance MultiIndex header:
        Row 0: Price,  Close, High, Low, Open, Volume
        Row 1: Ticker, AAPL,  AAPL, ...
        Row 2: Date,   (empty)
    Detects the format, flattens it, converts columns to numeric,
    and re-saves in clean flat format so future reads are instant.
    """
    # Peek at second row to detect MultiIndex format
    peek = pd.read_csv(cache_path, nrows=2, header=None)
    second_row = peek.iloc[1].dropna().tolist()

    # MultiIndex CSVs have the ticker symbol (a non-numeric string) in row 2
    is_multiindex = (
        len(second_row) > 1
        and isinstance(second_row[1], str)
        and not _is_numeric(str(second_row[1]))
    )

    if is_multiindex:
        raw = pd.read_csv(cache_path, header=[0, 1], index_col=0)
        # Flatten: keep only the first level (Price names: Close, High, ...)
        raw.columns = raw.columns.get_level_values(0)
        raw.columns.name = None
        # Drop the blank "Date" label row that appears as first data row
        raw = raw[pd.to_datetime(raw.index, errors="coerce").notna()]
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "Date"
        # Ensure all columns are numeric
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.dropna()
        # Re-save as clean flat CSV so this only runs once
        raw.to_csv(cache_path)
        print(f"[DataLoader] Converted MultiIndex cache to flat format: {cache_path.name}")
        return raw

    return pd.read_csv(cache_path, index_col=0, parse_dates=True)


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        try:
            pd.to_datetime(value)
            return True
        except Exception:
            return False

def download_data(ticker: str, start="2018-01-01"):

    cache_file = f"data/{ticker}.csv"

    if os.path.exists(cache_file):
        print(f"[DataLoader] Loading cached data for {ticker}")

        df = pd.read_csv(cache_file)

        # Detect and fix yfinance multiindex format
        if "Date" not in df.columns:
            df = pd.read_csv(cache_file, header=[0,1], index_col=0)
            df.columns = df.columns.get_level_values(0)
            df.reset_index(inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        return df

    raise FileNotFoundError(
        f"Cached data file not found: {cache_file}. "
        "Please add the CSV file to the data folder."
    )

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate downloaded market data and return a clean copy.

    Raises ``ValueError`` for missing required columns or insufficient rows.
    Drops any rows containing NaN values and returns the cleaned frame
    (original is NOT modified in-place).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV frame returned by :func:`download_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of *df*.
    """
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[DataLoader] Missing required columns: {missing}")

    if df.empty:
        raise ValueError("[DataLoader] Dataset is empty.")

    out = df.copy()

    nan_count = int(out.isna().sum().sum())
    if nan_count > 0:
        print(f"[DataLoader] Dropping {nan_count} NaN cell(s).")
        out = out.dropna()

    if len(out) < 100:
        raise ValueError(
            f"[DataLoader] Only {len(out)} rows after cleaning -- "
            "too few for reliable EMA backtesting (need >= 100)."
        )

    return out
