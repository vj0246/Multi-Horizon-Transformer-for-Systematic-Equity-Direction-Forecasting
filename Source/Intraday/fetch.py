"""Intraday bar ingestion, behind a source-agnostic interface.

Why hourly and not 5-minute: the two constraints pull in opposite directions and
only one band satisfies both.

  bar / hold          indep obs   cost / typical move   win rate to break even
  daily / 20 days         32              2%                   50.8%
  hourly / 20 bars       253              4%                   52.1%   <- here
  5-min  / 20 bars       950             14%                   57.1%
  5-min  / 1 bar        huge             64%                   82.0%   impossible

Sampling finer buys statistical power and spends it on transaction costs: at
5-minute bars the typical move is ~15bps against a 9.58bps round trip, so the
required win rate (57%) is HARDER than daily despite far more data. Hourly bars
held ~20 bars (~3 days) is the band where there is enough independent data to
measure an edge and the cost is still a small fraction of the move.

`SOURCES` maps a name to a loader returning a normalised frame. yfinance is
implemented and verified (5,057 hourly bars over 730 days, free). To add a
different provider - a broker API, or an NSE site that publishes bars - write a
loader returning the same columns and register it; nothing downstream changes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Data" / "Raw_Data" / "Intraday"

COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def _normalise(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Flatten any provider's frame to COLUMNS, tz-aware, sorted, deduped."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):          # yfinance multi-ticker shape
        df.columns = [c[0] for c in df.columns]
    # reset FIRST: the index carries the timestamp under its own name, which must
    # be lowercased along with everything else
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    for cand in ("datetime", "date", "index", "timestamp"):
        if cand in df.columns:
            df = df.rename(columns={cand: "datetime"})
            break
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(tz)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"source is missing columns {missing}; got {list(df.columns)}")
    df = df[COLUMNS].dropna(subset=["close"])
    return df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def from_yfinance(ticker: str = "^NSEI", interval: str = "1h",
                  period: str = "730d") -> pd.DataFrame:
    """Free intraday bars. Yahoo caps history by interval: 1h -> 730d, 5m -> 60d."""
    from Source.Ingestion.session import download
    # retries on a fresh impersonating session each attempt: Yahoo answers a
    # throttled request with an EMPTY body, and yfinance caches that emptiness on
    # the client, so reusing one would re-read the same nothing.
    df = download(ticker, interval=interval, period=period, auto_adjust=False)
    return _normalise(df)


def from_csv(path: str | Path) -> pd.DataFrame:
    """Any provider that can export bars to CSV.

    Accepts a datetime column named datetime/date/timestamp plus OHLCV in any
    case. This is the drop-in point for a broker API or an NSE bar feed: dump to
    CSV in that shape and it flows through the rest of the pipeline unchanged.
    """
    return _normalise(pd.read_csv(path))


SOURCES = {"yfinance": from_yfinance, "csv": from_csv}


def load(source: str = "yfinance", **kw) -> pd.DataFrame:
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}; have {list(SOURCES)}")
    return SOURCES[source](**kw)


def save(df: pd.DataFrame, ticker: str, interval: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{ticker.replace('^', '')}_{interval}.csv"
    df.to_csv(path, index=False)
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download intraday bars.")
    ap.add_argument("--ticker", default="^NSEI")
    ap.add_argument("--interval", default="1h", help="1h (730d history) or 5m (60d)")
    ap.add_argument("--period", default="730d")
    ap.add_argument("--source", default="yfinance", choices=list(SOURCES))
    ap.add_argument("--csv", help="path, when --source csv")
    args = ap.parse_args()

    kw = {"path": args.csv} if args.source == "csv" else {
        "ticker": args.ticker, "interval": args.interval, "period": args.period}
    df = load(args.source, **kw)
    path = save(df, args.ticker, args.interval)
    span_days = (df["datetime"].max() - df["datetime"].min()).days or 1
    print(f"{len(df):,} bars | {df['datetime'].min():%Y-%m-%d %H:%M} -> "
          f"{df['datetime'].max():%Y-%m-%d %H:%M} | {len(df)/span_days:.1f} bars/day")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
