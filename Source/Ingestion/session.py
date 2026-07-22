"""Shared market session: NSE trading calendar + a rate-limit-resistant yfinance client.

Two problems this repository already has, both fixed here once instead of in each
of the eight modules that call yfinance.

1. YAHOO THROTTLING. Under load Yahoo returns an EMPTY body rather than an error,
   which yfinance surfaces as "no data, possibly delisted". A plain retry does
   not help because yfinance caches that empty result on the Ticker instance, so
   every retry re-reads the same nothing. Two things are needed: a
   browser-impersonating TLS session (curl_cffi) so we are throttled far less,
   and a FRESH client per attempt so a retry actually refetches.

2. TRADING HOLIDAYS. NSE closes ~15 extra days a year beyond weekends. Nothing in
   this project knew that, so "is the market open" was wrong on Diwali, and any
   count of trading days between two dates was quietly inflated.

Deliberately NOT ported from the source module this came from: its sample-data
fallback. In a demo app, silently serving embedded sample prices when the network
fails is good UX. Here it would be a correctness disaster - a backtest would
train on fabricated prices and report metrics as if they were real, which is the
one thing this project must never do. Fetches here fail loudly instead.
"""
from __future__ import annotations

import datetime as dt
import time

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

OPEN = dt.time(9, 15)
CLOSE = dt.time(15, 30)

# NSE trading holidays. Published yearly by the exchange; extend as they are announced.
NSE_HOLIDAYS: set[dt.date] = {
    # 2025
    dt.date(2025, 2, 26), dt.date(2025, 3, 14), dt.date(2025, 3, 31),
    dt.date(2025, 4, 10), dt.date(2025, 4, 14), dt.date(2025, 4, 18),
    dt.date(2025, 5, 1), dt.date(2025, 6, 7), dt.date(2025, 8, 15),
    dt.date(2025, 8, 16), dt.date(2025, 10, 2), dt.date(2025, 10, 21),
    dt.date(2025, 10, 22), dt.date(2025, 11, 5), dt.date(2025, 12, 25),
    # 2026
    dt.date(2026, 1, 26), dt.date(2026, 2, 17), dt.date(2026, 3, 3),
    dt.date(2026, 3, 20), dt.date(2026, 3, 30), dt.date(2026, 4, 3),
    dt.date(2026, 4, 14), dt.date(2026, 5, 1), dt.date(2026, 5, 25),
    dt.date(2026, 6, 26), dt.date(2026, 7, 17), dt.date(2026, 8, 15),
    dt.date(2026, 8, 25), dt.date(2026, 10, 2), dt.date(2026, 10, 9),
    dt.date(2026, 10, 29), dt.date(2026, 10, 30), dt.date(2026, 11, 25),
    dt.date(2026, 12, 25),
}

# Holidays are only known for the years listed above; outside that range the
# calendar silently degrades to weekends-only, so callers can check rather than
# assume.
CALENDAR_YEARS = {d.year for d in NSE_HOLIDAYS}


def is_trading_day(d: dt.date) -> bool:
    """Weekday and not an NSE holiday."""
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def market_open(now: dt.datetime | None = None) -> bool:
    """True during the NSE cash session, holidays and weekends excluded."""
    now = (now or dt.datetime.now(IST)).astimezone(IST)
    return is_trading_day(now.date()) and OPEN <= now.time() <= CLOSE


def trading_days(start: dt.date, end: dt.date) -> int:
    """Count trading days in [start, end]. Warns callers via calendar_covers()."""
    n, cur = 0, start
    while cur <= end:
        if is_trading_day(cur):
            n += 1
        cur += dt.timedelta(days=1)
    return n


def calendar_covers(start: dt.date, end: dt.date) -> bool:
    """Whether the holiday list actually spans this range, or is degrading to weekends."""
    return all(y in CALENDAR_YEARS for y in range(start.year, end.year + 1))


def impersonating_session():
    """A curl_cffi session that looks like Chrome, or None if unavailable."""
    try:
        from curl_cffi import requests as creq
        return creq.Session(impersonate="chrome")
    except Exception:
        return None


def ticker(symbol: str):
    """yfinance Ticker on an impersonating session, falling back to the default."""
    import yfinance as yf
    s = impersonating_session()
    return yf.Ticker(symbol, session=s) if s is not None else yf.Ticker(symbol)


def download(symbol: str, tries: int = 3, **kw):
    """yfinance download with retries and a FRESH session per attempt.

    The fresh session matters: yfinance caches an empty response on the client,
    so retrying with the same object re-reads the same nothing. Raises rather
    than returning empty - a silent empty frame downstream becomes a silent gap
    in a training set.
    """
    import yfinance as yf
    last = None
    for attempt in range(tries):
        try:
            s = impersonating_session()
            df = yf.download(symbol, session=s, progress=False, **kw) if s is not None \
                else yf.download(symbol, progress=False, **kw)
            if df is not None and not df.empty:
                return df
            last = "empty response (Yahoo throttling)"
        except Exception as e:                                # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"yfinance failed for {symbol} after {tries} attempts: {last}")
