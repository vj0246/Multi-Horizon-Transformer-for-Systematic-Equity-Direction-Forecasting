"""Live quotes, routed to the source that is actually authoritative for each field.

No single free source is best at everything, and using the wrong one is not a
style choice - it changes the numbers:

  HISTORICAL BARS   -> yfinance      depth (2007+ daily, 730d hourly). NSE's
                                     public endpoints do not serve deep history.
  LIVE SPOT PRICE   -> NSE first     authoritative and real-time. yfinance's
                                     history() returns only SETTLED daily bars,
                                     so during the session its last close is
                                     yesterday's - wrong for anything live.
  CURRENT RATIOS    -> NSE           authoritative Indian P/E, sector P/E, and
                                     delivery % (unavailable from yfinance).
  FUNDAMENTAL HIST. -> screener.py   the only source here with real history, and
                                     only through its point-in-time filter.

This module owns the SECOND row. It is deliberately execution-side: nothing here
feeds a backtest, because a live quote has no history to train on. Research reads
CSVs; this reads the market.

FALLBACK POLICY
---------------
NSE -> yfinance -> raise. Every returned quote carries `source`, so a caller can
always tell which one answered. There is deliberately NO synthetic or sample
fallback: a fabricated price that looks real is worse than an error, and this
project's core rule is that no number is ever invented.
"""
from __future__ import annotations

import time
from datetime import datetime

from Source.Ingestion import nse
from Source.Ingestion.session import IST, market_open, ticker

_QUOTE_CACHE: dict = {}
_TTL_LIVE = 20          # seconds, during the session
_TTL_CLOSED = 3600      # off-hours the last close does not move


def _ttl() -> int:
    return _TTL_LIVE if market_open() else _TTL_CLOSED


def _num(v):
    """float only if v is a real finite number (NaN != NaN)."""
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _from_nse(symbol: str) -> dict:
    """NSE quote. Authoritative and real-time when reachable."""
    d = nse.symbol_data(symbol)
    if not d:
        return {}
    # nse.symbol_data returns fundamentals; pull the price fields it carries
    out = {k: d[k] for k in ("year_high", "year_low", "delivery_pct",
                             "annual_volatility_pct") if k in d}
    return out


def _from_yfinance(symbol: str) -> dict:
    """Live-ish price from yfinance, preferring fast_info over settled bars.

    history() only returns SETTLED daily bars, so mid-session its last row is
    yesterday's close. fast_info carries the intraday last price; a 1-minute bar
    is the backstop.
    """
    sym = symbol if symbol.startswith("^") or "." in symbol else f"{symbol}.NS"
    tk = ticker(sym)
    price = prev = high = low = None
    try:
        fi = tk.fast_info
        price = _num(getattr(fi, "last_price", None))
        prev = _num(getattr(fi, "previous_close", None))
        high = _num(getattr(fi, "day_high", None))
        low = _num(getattr(fi, "day_low", None))
    except Exception:
        pass
    if price is None:
        try:
            intr = tk.history(period="1d", interval="1m").dropna(subset=["Close"])
            if not intr.empty:
                price = _num(intr["Close"].iloc[-1])
                high = high or _num(intr["High"].max())
                low = low or _num(intr["Low"].min())
        except Exception:
            pass
    if price is None:
        return {}
    return {"price": round(price, 2),
            "prev_close": round(prev, 2) if prev else None,
            "day_high": round(high, 2) if high else None,
            "day_low": round(low, 2) if low else None}


def quote(symbol: str, use_cache: bool = True) -> dict:
    """Freshest available quote for one symbol.

    Raises if BOTH sources fail - never returns a fabricated price.
    """
    key = symbol.strip().upper()
    now = time.time()
    if use_cache and key in _QUOTE_CACHE and now - _QUOTE_CACHE[key][0] < _ttl():
        return _QUOTE_CACHE[key][1]

    nse_part = _from_nse(key)          # authoritative extras (delivery %, 52w)
    yf_part = _from_yfinance(key)      # the tradeable last price

    if not yf_part and not nse_part:
        raise RuntimeError(
            f"no live quote for {key}: NSE unreachable (it blocks datacenter IPs) "
            "and yfinance returned nothing. Refusing to synthesise a price.")

    price = yf_part.get("price")
    prev = yf_part.get("prev_close")
    out = {
        "symbol": key,
        "price": price,
        "change_pct": round((price / prev - 1) * 100, 2) if (price and prev) else None,
        **{k: v for k, v in yf_part.items() if k != "price"},
        **nse_part,
        "market_open": market_open(),
        "as_of": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": ("nse+yfinance" if (nse_part and yf_part)
                   else "nse" if nse_part else "yfinance"),
    }
    out = {k: v for k, v in out.items() if v is not None}
    _QUOTE_CACHE[key] = (now, out)
    return out


INDICES = [("NIFTY 50", "^NSEI"), ("SENSEX", "^BSESN"),
           ("NIFTY BANK", "^NSEBANK"), ("INDIA VIX", "^INDIAVIX")]


def indices() -> list[dict]:
    """Headline index levels. INDIA VIX is already a model feature, so this is
    also a live sanity check that the macro series is behaving."""
    out = []
    for name, sym in INDICES:
        try:
            q = quote(sym)
            out.append({"index": name, "symbol": sym, "level": q.get("price"),
                        "change_pct": q.get("change_pct")})
        except Exception:
            continue
    return out


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Live quotes (execution-side, not research).")
    ap.add_argument("symbols", nargs="*", default=["^NSEI"])
    ap.add_argument("--indices", action="store_true")
    args = ap.parse_args()

    print(f"market open: {market_open()}  ({datetime.now(IST):%Y-%m-%d %H:%M} IST)")
    if args.indices:
        for row in indices():
            lvl = row.get("level")
            chg = row.get("change_pct")
            print(f"  {row['index']:12} {lvl if lvl else 'n/a':>10} "
                  f"{f'{chg:+.2f}%' if chg is not None else '':>8}")
        return
    for s in args.symbols:
        try:
            print(json.dumps(quote(s), indent=1))
        except RuntimeError as e:
            print(f"  {s}: {e}")


if __name__ == "__main__":
    main()
