"""Authoritative NSE India fundamentals, archived daily to build a point-in-time panel.

WHAT THIS IS AND IS NOT
-----------------------
NSE serves only TODAY's values: P/E, sector P/E, market cap, 52-week range,
delivery %, annualised volatility. Like the yfinance snapshot in
fetch_fundamentals.py, these are a genuine live view and are **look-ahead
leakage if used as historical features** - feeding today's P/E into a 2015
training window tells the model the future.

The project has documented "no free point-in-time NSE fundamentals" as a hard
blocker throughout. This module is how that blocker gets removed: it APPENDS a
dated snapshot on every run, so the archive becomes a real point-in-time panel
going forward. It cannot backfill; it can only accumulate. Start it now and it
is worth something in a year.

Until `Data/Raw_Data/NSE_Fundamentals/snapshots.csv` spans enough history to
cover a train/test split, treat these columns as display-only. `as_of` exists so
that check is mechanical rather than a matter of trust.

WHY delivery_pct IS THE INTERESTING ONE
---------------------------------------
Delivery percentage - the share of traded quantity actually settled rather than
squared off intraday - is a genuine microstructure signal and is NOT available
from yfinance at all. High delivery means positional conviction; low delivery
means speculative churn. It is the one field here that is closer to a price
feature than a fundamental.

ACCESS NOTES
------------
NSE serves these endpoints only to browser-like clients and blocks many
datacenter IPs, so this typically works from an Indian machine and fails from a
US cloud host. Every call fails soft and a circuit breaker prevents repeatedly
waiting on timeouts once NSE has refused us.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from Source.Ingestion.session import IST, market_open as _market_open

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "Data" / "Raw_Data" / "NSE_Fundamentals" / "snapshots.csv"

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

_SESSION = None
_CACHE: dict = {}
_TTL_MARKET = 300          # 5 min during market hours
_TTL_OFFHOURS = 86400      # 24 h outside them
_DOWN_UNTIL = 0.0          # circuit breaker: skip NSE until this time after a failure
_COOLDOWN = 300

def _get_ttl() -> int:
    return _TTL_MARKET if _market_open() else _TTL_OFFHOURS


def _f(v):
    try:
        f = float(v)
        return f if f == f else None          # NaN check
    except (TypeError, ValueError):
        return None


def _session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    from curl_cffi import requests as creq
    s = creq.Session(impersonate="chrome")
    s.headers.update(HEADERS)
    s.get(BASE, timeout=8)                    # warm up cookies; raises so caller backs off
    _SESSION = s
    return s


def symbol_data(ticker: str) -> dict:
    """Authoritative fundamentals for one NSE symbol, or {} if NSE is unreachable."""
    global _SESSION, _DOWN_UNTIL
    key = (ticker or "").strip().upper().replace(".NS", "").replace(".BO", "")
    if not key:
        return {}
    now = time.time()
    if now < _DOWN_UNTIL:                     # recently failed - do not hang on it again
        return {}
    if key in _CACHE and now - _CACHE[key][0] < _get_ttl():
        return _CACHE[key][1]

    out: dict = {}
    try:
        s = _session()
        url = (f"{BASE}/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolData"
               f"&marketType=N&series=EQ&symbol={key}")
        r = s.get(url, timeout=8)
        eq = (r.json() or {}).get("equityResponse", [])
        if eq:
            sym = eq[0]
            sec = sym.get("secInfo") or {}
            ti = sym.get("tradeInfo") or {}
            pi = sym.get("priceInfo") or {}
            mc = _f(ti.get("totalMarketCap"))          # rupees
            out = {
                "pe": _f(sec.get("pdSymbolPe")),
                "sector_pe": _f(sec.get("pdSectorPe")),
                "sector": sec.get("sector") or None,
                "industry": sec.get("basicIndustry") or None,
                "market_cap": mc,
                "market_cap_cr": round(mc / 1e7) if mc else None,
                "year_high": _f(pi.get("yearHigh")),
                "year_low": _f(pi.get("yearLow")),
                "annual_volatility_pct": _f(pi.get("cmAnnualVolatility")),
                "delivery_pct": _f(ti.get("deliveryToTradedQuantity")),
            }
            out = {k: v for k, v in out.items() if v is not None}
    except Exception:
        _SESSION = None                        # drop a stale session
        _DOWN_UNTIL = now + _COOLDOWN          # back off rather than keep timing out
        return {}

    _CACHE[key] = (now, out)
    return out


def snapshot(tickers: list[str], pause_s: float = 0.35) -> "list[dict]":
    """One dated observation per symbol. `as_of` is what makes the archive usable."""
    stamp = datetime.now(IST).strftime("%Y-%m-%d")
    rows = []
    for t in tickers:
        d = symbol_data(t)
        if d:
            rows.append({"as_of": stamp, "symbol": t.upper().replace(".NS", ""), **d})
        time.sleep(pause_s)                    # NSE dislikes bursts
    return rows


def append_archive(rows: list[dict]) -> Path:
    """Append today's rows, replacing any existing rows for the same as_of+symbol.

    Idempotent per day, so re-running does not duplicate. Rows for PREVIOUS days
    are never touched - that history is the whole point and overwriting it would
    silently destroy the point-in-time property.
    """
    import pandas as pd

    new = pd.DataFrame(rows)
    if new.empty:
        return ARCHIVE
    if ARCHIVE.exists():
        old = pd.read_csv(ARCHIVE)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["as_of", "symbol"], keep="last")
    else:
        combined = new
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["as_of", "symbol"]).to_csv(ARCHIVE, index=False)
    return ARCHIVE


def archive_span() -> dict:
    """How much point-in-time history exists yet - the gate on using this at all."""
    import pandas as pd
    if not ARCHIVE.exists():
        return {"days": 0, "usable_as_features": False,
                "note": "no archive yet; run this daily to start accumulating"}
    df = pd.read_csv(ARCHIVE)
    days = df["as_of"].nunique()
    return {
        "days": int(days),
        "symbols": int(df["symbol"].nunique()),
        "start": str(df["as_of"].min()),
        "end": str(df["as_of"].max()),
        # a train/test split needs enough distinct dates to be meaningful at all
        "usable_as_features": bool(days >= 250),
        "note": (
            f"{days} distinct dates archived. These become legitimate model "
            "features only once the archive spans a real train/test split; until "
            "then they are display-only, because a snapshot applied backwards is "
            "look-ahead leakage."
        ),
    }


def main():
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="Archive a dated NSE fundamentals snapshot.")
    ap.add_argument("--tickers", nargs="*", help="default: the config universe")
    ap.add_argument("--status", action="store_true", help="report archive span and exit")
    args = ap.parse_args()

    if args.status:
        for k, v in archive_span().items():
            print(f"  {k}: {v}")
        return

    tickers = args.tickers
    if not tickers:
        cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
        uni = (cfg.get("cross_section", {}) or {}).get("universe") or []
        tickers = [t.replace(".NS", "") for t in uni]
    if not tickers:
        import glob
        tickers = [Path(p).stem for p in
                   glob.glob(str(ROOT / "Data" / "Raw_Data" / "Universe" / "*.csv"))]
    if not tickers:
        raise SystemExit("no tickers - pass --tickers or populate the universe")

    print(f"snapshotting {len(tickers)} symbols from NSE ...")
    rows = snapshot(tickers)
    if not rows:
        raise SystemExit("NSE returned nothing - it blocks datacenter IPs; "
                         "run this from an Indian residential connection")
    path = append_archive(rows)
    span = archive_span()
    print(f"{len(rows)}/{len(tickers)} symbols archived -> {path}")
    print(f"  archive now spans {span['days']} distinct date(s)")
    print(f"  {span['note']}")


if __name__ == "__main__":
    main()
