"""Download auxiliary macro/market series used as model features.

Run:
    python -m Source.Ingestion.fetch_macro
-> Data/Raw_Data/Macro/{INDIAVIX,GSPC,USDINR,CRUDE}.csv

Why these are legitimate (not look-ahead):
- INDIAVIX: NSE's own volatility index, published intraday; its close on day d is
  known at day d's close.
- GSPC (S&P 500): the US session for calendar day d closes ~02:00 IST on d+1 —
  i.e. BEFORE the Indian session of d+1 trades. The overnight US move is a real,
  well-documented input to the next Indian session and is genuinely available.
- USDINR / CRUDE: macro drivers for an oil-importing economy.

Alignment safety is enforced in Source/Features/Macro.py: every series is
merge_asof'd BACKWARD onto NSE dates (last value at or before the date) and then
shifted one row, so a feature at row t only carries information from <= t-1.
"""
from __future__ import annotations

from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Data" / "Raw_Data" / "Macro"

SERIES = {
    "INDIAVIX": "^INDIAVIX",
    "GSPC": "^GSPC",
    "USDINR": "INR=X",
    "CRUDE": "CL=F",
}


def fetch_macro() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ticker in SERIES.items():
        df = yf.download(ticker, start="2007-01-01", auto_adjust=True, progress=False)
        if df.empty:
            print(f"  {name} ({ticker}): EMPTY - skipped")
            continue
        if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
        df = df[["date", "close"]].dropna()
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  {name}: {len(df)} rows ({df['date'].min().date()} .. {df['date'].max().date()})")


if __name__ == "__main__":
    fetch_macro()
