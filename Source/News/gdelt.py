"""Historical news sentiment from GDELT - free, no key, and actually backfillable.

The existing NewsAPI path (build_sentiment.py) is stuck off by default for a
reason that cannot be engineered around: the free tier serves ~30 days, so a
sentiment feature built from it exists for 30 of ~4,600 training days. Training
on that means zero-filling 99% of history, which invents a feature rather than
adding one.

GDELT DOC 2.0 solves exactly that gap:
  - free, no API key
  - global news coverage including Indian outlets
  - history back to 2017
  - `timelinetone` returns an average tone series, which is what we want -
    per-article scoring is unnecessary when the target is a per-bar aggregate

Three operational realities, handled here rather than left to callers:
  1. RESOLUTION SCALES WITH THE WINDOW. A 60-day request returns ~60 points, i.e.
     DAILY tone, not 15-minute. Finer resolution needs smaller `chunk_days`,
     which means proportionally more requests - and see (2). Daily tone is
     sufficient for the daily track; the hourly track forward-fills it.
  2. It rate-limits aggressively (HTTP 429) with no documented budget, so a
     730-day backfill will usually NOT complete in one pass. Every chunk is
     therefore saved as it arrives and `resume` skips what is already on disk:
     run the command repeatedly to fill gaps rather than expecting one clean run.
  3. OR'd query terms MUST be parenthesised or the API returns a plain-text
     error with HTTP 200 - a malformed query looks like a successful empty
     result unless you check.

LEAKAGE: `attach_sentiment` merges asof-BACKWARD and then shifts, so a bar at
time t carries only sentiment published strictly before t. GDELT timestamps are
publication times, not event times, but a story published at 10:00 can still
describe a 09:50 move, so the shift is mandatory, not defensive.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Data" / "Processed_Data" / "gdelt_sentiment.csv"
API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Parentheses are required by the API for OR'd terms - without them GDELT
# returns HTTP 200 with a plain-text error body.
DEFAULT_QUERY = '("nifty 50" OR "sensex" OR "nse india" OR "indian stock market")'

MAX_RETRIES = 5
TIMEOUT_S = 45


def _get(params: dict) -> dict:
    """One GDELT call with backoff. Raises RuntimeError on persistent failure.

    Network faults (read timeout, connection reset) are retried like a 429 and
    then re-raised AS RuntimeError, so callers have exactly one exception type to
    handle. Letting a raw requests exception escape here once cost a partially
    completed backfill.
    """
    delay = 5.0
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API, params=params, timeout=TIMEOUT_S,
                             headers={"User-Agent": "mht-research/1.0"})
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(delay)
            delay *= 1.8
            continue
        if r.status_code == 200:
            body = r.text.lstrip()
            if not body.startswith("{"):
                # 200 with a text body = malformed query, not a transient fault
                raise RuntimeError(f"GDELT rejected the query: {body[:200]}")
            return r.json()
        last = r.status_code
        if r.status_code != 429 and r.status_code < 500:
            raise RuntimeError(f"GDELT HTTP {r.status_code}: {r.text[:200]}")
        time.sleep(delay)
        delay *= 1.8
    raise RuntimeError(f"GDELT unavailable after {MAX_RETRIES} attempts (last {last})")


def fetch_tone(start: datetime, end: datetime, query: str = DEFAULT_QUERY) -> pd.DataFrame:
    """Average news tone between two datetimes -> DataFrame[datetime, tone].

    Tone is roughly [-100, +100], negative = negative coverage. GDELT caps each
    response, so callers should window long spans (see `fetch_range`).
    """
    j = _get({
        "query": query, "mode": "timelinetone", "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    })
    tl = j.get("timeline") or []
    if not tl:
        return pd.DataFrame(columns=["datetime", "tone"])
    rows = tl[0].get("data") or []
    if not rows:
        return pd.DataFrame(columns=["datetime", "tone"])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    df = df.rename(columns={"value": "tone"})[["datetime", "tone"]]
    return df.sort_values("datetime").reset_index(drop=True)


def _merge_save(new: pd.DataFrame) -> pd.DataFrame:
    """Merge new points into the CSV on disk and persist. Never loses prior work."""
    frames = [new]
    if OUT.exists():
        frames.append(pd.read_csv(OUT, parse_dates=["datetime"]))
    df = (pd.concat(frames, ignore_index=True)
          .dropna(subset=["datetime"])
          .drop_duplicates("datetime")
          .sort_values("datetime")
          .reset_index(drop=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


def fetch_range(start: datetime, end: datetime, query: str = DEFAULT_QUERY,
                chunk_days: int = 60, pause_s: float = 5.0,
                resume: bool = True) -> pd.DataFrame:
    """Walk a long span in chunks, saving after every chunk.

    GDELT rate-limits hard and unpredictably, so a 730-day backfill will usually
    NOT complete in one pass. Two properties make that a non-problem:

      - every successful chunk is written to disk immediately, so a failure
        later never discards earlier work
      - `resume` skips chunks already covered on disk, so re-running the command
        fills the gaps instead of starting over

    Run it a few times rather than expecting one clean pass.
    """
    have = set()
    if resume and OUT.exists():
        prior = pd.read_csv(OUT, parse_dates=["datetime"])
        have = set(pd.to_datetime(prior["datetime"], utc=True).dt.date)
        print(f"  resuming: {len(have)} days already on disk")

    cur = start
    total_new = 0
    while cur < end:
        stop = min(cur + timedelta(days=chunk_days), end)
        span_days = {(cur + timedelta(days=i)).date()
                     for i in range((stop - cur).days + 1)}
        if resume and span_days and span_days <= have:
            print(f"  {cur:%Y-%m-%d} -> {stop:%Y-%m-%d}: already covered")
            cur = stop
            continue
        try:
            part = fetch_tone(cur, stop, query)
            if not part.empty:
                _merge_save(part)
                total_new += len(part)
            print(f"  {cur:%Y-%m-%d} -> {stop:%Y-%m-%d}: {len(part)} points (saved)")
        except RuntimeError as e:
            print(f"  {cur:%Y-%m-%d} -> {stop:%Y-%m-%d}: skipped ({e})")
        cur = stop
        time.sleep(pause_s)                      # be a good citizen; it rate-limits

    print(f"  {total_new} new points this pass")
    if OUT.exists():
        return pd.read_csv(OUT, parse_dates=["datetime"])
    return pd.DataFrame(columns=["datetime", "tone"])


def attach_sentiment(bars: pd.DataFrame, tone: pd.DataFrame, *,
                     time_col: str = "datetime", lag_bars: int = 1) -> pd.DataFrame:
    """Join tone onto bars with a STRICT backward lag.

    merge_asof(direction="backward") never looks forward, and the shift then
    pushes the reading one bar further back, so a bar at t uses only sentiment
    that existed before the bar opened. Both steps are required: without the
    shift, a story published inside bar t would be visible to the model
    predicting bar t.
    """
    if tone.empty:
        bars = bars.copy()
        bars["news_tone"] = 0.0
        bars["news_tone_chg"] = 0.0
        bars["has_news"] = 0.0
        return bars

    b = bars.sort_values(time_col).copy()
    t = tone.sort_values("datetime").copy()
    for frame, col in ((b, time_col), (t, "datetime")):
        if frame[col].dt.tz is None:
            frame[col] = frame[col].dt.tz_localize("UTC")

    merged = pd.merge_asof(b[[time_col]], t, left_on=time_col, right_on="datetime",
                           direction="backward")
    raw = merged["tone"].to_numpy()

    b["has_news"] = (~pd.isna(raw)).astype(float)
    b["news_tone"] = pd.Series(raw).shift(lag_bars).to_numpy()
    b["news_tone_chg"] = pd.Series(raw).diff().shift(lag_bars).to_numpy()
    for c in ("news_tone", "news_tone_chg"):
        b[c] = b[c].fillna(0.0)
    return b


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Backfill GDELT news tone.")
    ap.add_argument("--days", type=int, default=730, help="how far back to fetch")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--chunk-days", type=int, default=60,
                    help="smaller chunks = finer tone resolution, more requests")
    ap.add_argument("--no-resume", action="store_true",
                    help="refetch everything instead of filling gaps")
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"GDELT tone {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    df = fetch_range(start, end, args.query, chunk_days=args.chunk_days,
                     resume=not args.no_resume)
    if df.empty:
        raise SystemExit("no data returned - check the query parenthesisation")
    span = (df["datetime"].max() - df["datetime"].min()).days or 1
    print(f"{len(df)} points on disk -> {OUT}")
    print(f"  {df['datetime'].min():%Y-%m-%d} -> {df['datetime'].max():%Y-%m-%d} "
          f"({len(df)/span:.2f} points/day)")
    print(f"  tone mean {df['tone'].mean():+.3f}  sd {df['tone'].std():.3f}")
    covered = df["datetime"].dt.date.nunique()
    if covered < args.days * 0.9:
        print(f"  NOTE: {covered}/{args.days} days covered. GDELT rate-limits hard; "
              "re-run this command to fill the gaps (it resumes).")


if __name__ == "__main__":
    main()
