"""Load and clean the raw yfinance OHLCV export for ^NSEI.

The raw CSV that yfinance writes has a three-row header artifact:

    Price,Adj Close,Close,High,Low,Open,Volume
    Ticker,^NSEI,^NSEI,^NSEI,^NSEI,^NSEI,^NSEI
    Date,,,,,,
    2007-09-17,4494.64,4494.64,4549.04,4482.85,4518.45,0
    ...

This module normalizes that into a clean, typed, date-sorted DataFrame with
columns [date, close, high, low, open, volume] - dropping the duplicate
adjusted-close column that yfinance emits.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["date", "close", "close_adj", "high", "low", "open", "volume"]
NUMERIC_COLUMNS = ["close", "high", "low", "open", "volume"]


def load_ohlcv(csv_path: str | Path) -> pd.DataFrame:
    """Return a clean OHLCV frame sorted ascending by date.

    Robust to the two yfinance header rows (Ticker / Date) that sit above the
    real data. Raises FileNotFoundError with a helpful message if the raw file
    is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. "
            f"Run: python Source/Ingestion/Fetch_Market_Data.py"
        )

    # First row is the real header (Price, Adj Close, Close, ...). Rename it to
    # our canonical schema, then drop the two metadata rows (Ticker / Date).
    df = pd.read_csv(csv_path)
    df.columns = RAW_COLUMNS

    # Drop the yfinance metadata rows: any row where `close` is non-numeric.
    df = df[pd.to_numeric(df["close"], errors="coerce").notna()].copy()

    df = df.drop(columns=["close_adj"])
    df["date"] = pd.to_datetime(df["date"])
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(float)

    df = df.sort_values("date").reset_index(drop=True)
    return df
