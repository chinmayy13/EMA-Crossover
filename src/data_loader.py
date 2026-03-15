"""
data_loader.py
--------------
Historical OHLCV data ingestion via yfinance with local CSV caching.
"""

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf


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

def download_data(
    ticker: str,
    start: str = "2018-01-01",
    end: str = None,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Download OHLCV data for *ticker* from Yahoo Finance and cache to CSV.

    On subsequent calls the cached file is returned immediately, avoiding
    network round-trips during development or optimisation loops.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "SPY", "AAPL", "BTC-USD".
    start : str
        Earliest date to fetch (YYYY-MM-DD).
    end : str or None
        Latest date to fetch (inclusive).  None -> today.
    cache_dir : str
        Directory for cached CSV files.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex.
    """
    cache_path = Path(cache_dir) / f"{ticker.replace('-', '_')}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = _read_cache(cache_path)
        print(f"[DataLoader] Loaded '{ticker}' from cache ({len(df)} rows)")
        return df

    print(f"[DataLoader] Downloading '{ticker}' from Yahoo Finance ...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            f"[DataLoader] No data returned for '{ticker}'. "
            "Check the ticker symbol and date range."
        )

    # yfinance >= 0.2.x may return a MultiIndex column when downloading a
    # single ticker with auto_adjust=True -- flatten it if so.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    raw.to_csv(cache_path)
    print(f"[DataLoader] Cached to {cache_path} ({len(raw)} rows)")
    return raw


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