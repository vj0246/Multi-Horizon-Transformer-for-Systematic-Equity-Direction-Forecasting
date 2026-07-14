"""Fetch a CURRENT fundamentals snapshot for the universe (yfinance).

Run:
    python -m Source.Ingestion.fetch_fundamentals
-> Data/Processed_Data/fundamentals_snapshot.csv

IMPORTANT — read before using this anywhere near the model:
    yfinance exposes only *today's* fundamentals (trailing P/E, ROE, margins,
    market cap as of now). These are a valid, genuine snapshot for a LIVE view
    ("what does each name look like today"), but they are NOT point-in-time and
    therefore CANNOT be used as features in the 2007-2026 backtest: feeding
    today's P/E to a 2015 training window is look-ahead leakage. A leak-free
    fundamental factor needs a vendor with as-reported, as-of-date history
    (S&P Capital IQ, Refinitiv, or NSE/BSE corporate filings parsed by report
    date). Until such a source is wired in, fundamentals stay out of the model
    by design - this fetcher exists for display/context only, never as a
    training input.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Data" / "Processed_Data" / "fundamentals_snapshot.csv"
FIELDS = ["trailingPE", "priceToBook", "returnOnEquity", "profitMargins",
          "marketCap", "dividendYield", "beta"]


def fetch_fundamentals() -> pd.DataFrame:
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    rows = []
    for ticker in cfg["universe"]["tickers"]:
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:  # network / symbol issues shouldn't abort the batch
            print(f"  {ticker}: {e}")
            continue
        row = {"ticker": ticker.replace(".NS", "")}
        row.update({f: info.get(f) for f in FIELDS})
        rows.append(row)
        print(f"  {ticker}: PE={row.get('trailingPE')} ROE={row.get('returnOnEquity')}")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}  (SNAPSHOT ONLY - not backtest features)")
    return df


if __name__ == "__main__":
    fetch_fundamentals()
