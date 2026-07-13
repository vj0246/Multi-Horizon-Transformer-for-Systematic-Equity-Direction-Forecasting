"""Download daily OHLCV for the cross-sectional NSE universe.

Run from repo root:
    python -m Source.Ingestion.fetch_universe

Writes one CSV per ticker to Data/Raw_Data/Universe/<TICKER>.csv with columns
[date, close, high, low, open, volume] (already cleaned - no yfinance
multi-header rows). Tickers with insufficient history are still written; the
panel builder filters by universe.min_history_days.
"""
from __future__ import annotations

from pathlib import Path

import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]


def fetch_universe() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    out_dir = ROOT / cfg["universe"]["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    start = cfg["data"]["start_date"]

    for ticker in cfg["universe"]["tickers"]:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  {ticker}: EMPTY - skipped")
            continue
        if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
        df = df[["date", "close", "high", "low", "open", "volume"]]
        safe = ticker.replace(".NS", "").replace("&", "_")
        df.to_csv(out_dir / f"{safe}.csv", index=False)
        print(f"  {ticker}: {len(df)} rows ({df['date'].min().date()} .. {df['date'].max().date()})")


if __name__ == "__main__":
    fetch_universe()
