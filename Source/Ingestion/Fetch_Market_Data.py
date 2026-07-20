"""Download the raw ^NSEI OHLCV export the pipeline trains on.

Writes the multi-header yfinance CSV that `data_loader.load_ohlcv` expects, to
the ticker/path configured in config.yaml rather than hardcoded ones.

Run:  python -m Source.Ingestion.Fetch_Market_Data
"""
from __future__ import annotations

from pathlib import Path

import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]


def fetch_data(ticker: str, start_date: str, out_csv: Path):
    data = yf.download(ticker, start=start_date, auto_adjust=False, progress=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_csv)
    return data


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    out = ROOT / cfg["data"]["raw_csv"]
    df = fetch_data(cfg["data"]["ticker"], cfg["data"]["start_date"], out)
    print(f"{cfg['data']['ticker']}: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
