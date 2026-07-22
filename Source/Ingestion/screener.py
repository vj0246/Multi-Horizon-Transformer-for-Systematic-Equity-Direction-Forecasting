"""Historical fundamentals from Screener.in, stamped with when they became PUBLIC.

This is the first fundamentals source in the project with real history: ~11 years
of annual P&L, ~13 quarters of results, and ~12 quarters of shareholding
(promoter / FII / DII). NSE and the yfinance snapshot both serve only today's
values and are therefore unusable as features; this is not.

THE REPORTING LAG IS THE WHOLE POINT
------------------------------------
A quarter labelled "Mar 2026" ENDS 2026-03-31 but is not ANNOUNCED until weeks
later. Using its revenue as a feature on 2026-04-01 is look-ahead leakage: the
market did not know the number yet. This is the single easiest way to
manufacture a fake edge from fundamental data, and it is why every row here
carries `available_from` rather than just a period label.

Lags applied (SEBI LODR deadlines, taken at the limit rather than the typical
case, because assuming a company reported early is exactly the optimistic
assumption that produces phantom alpha):

  quarterly results   +45 days after period end   (Reg 33)
  annual results      +60 days after year end     (Reg 33)
  shareholding        +21 days after quarter end  (Reg 31)

Callers MUST filter on `available_from <= t`, never on the period label. The
`point_in_time()` helper does this correctly; use it rather than hand-rolling.

WHAT THIS DOES NOT FIX
----------------------
Survivorship. Screener publishes pages for currently-listed companies, so names
that delisted, merged or collapsed are simply absent. A panel built from it still
over-represents winners, and the cross-sectional results carry that bias no
matter how carefully the reporting lag is handled.

Keyless and public. Best-effort: any failure returns empty and a circuit breaker
avoids hammering an unreachable host.
"""
from __future__ import annotations

import html
import re
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "Data" / "Raw_Data" / "Fundamentals" / "screener_panel.csv"

BASE = "https://www.screener.in"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# SEBI LODR disclosure deadlines, taken at the limit (see module docstring).
LAG_QUARTERLY = 45
LAG_ANNUAL = 60
LAG_SHAREHOLDING = 21

_SESSION = None
_PAGE_CACHE: dict = {}
_DOWN_UNTIL = 0.0
_COOLDOWN = 300
_TTL = 86400

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def _key(ticker: str) -> str:
    return (ticker or "").strip().upper().replace(".NS", "").replace(".BO", "")


def _num(s):
    """First number in a Screener cell ('₹ 7,57,881 Cr.' -> 757881.0)."""
    if not s:
        return None
    s = str(s).replace(",", "").replace("%", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _period_end(label: str) -> date | None:
    """'Mar 2026' -> 2026-03-31. Returns None for TTM and unparsable labels."""
    m = re.match(r"([A-Za-z]{3})\s+(\d{4})", (label or "").strip())
    if not m:
        return None                       # 'TTM' has no period end - deliberately dropped
    mon, yr = _MONTHS.get(m.group(1)), int(m.group(2))
    if not mon:
        return None
    nxt = date(yr + (mon == 12), (mon % 12) + 1, 1)
    return nxt - timedelta(days=1)


def _session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    from curl_cffi import requests as creq
    s = creq.Session(impersonate="chrome")
    s.headers.update(HEADERS)
    _SESSION = s
    return s


def _get_page(key: str) -> str:
    global _SESSION, _DOWN_UNTIL
    if not key:
        return ""
    now = time.time()
    if now < _DOWN_UNTIL:
        return ""
    if key in _PAGE_CACHE and now - _PAGE_CACHE[key][0] < _TTL:
        return _PAGE_CACHE[key][1]
    text = ""
    try:
        s = _session()
        for path in (f"/company/{key}/consolidated/", f"/company/{key}/"):
            r = s.get(BASE + path, timeout=12)
            if r.status_code == 200 and "Stock P/E" in r.text:
                text = r.text
                break
    except Exception:
        _SESSION = None
        _DOWN_UNTIL = now + _COOLDOWN
        return ""
    _PAGE_CACHE[key] = (now, text)
    return text


def _table(block: str):
    """(columns, {row_label: [raw cells]}) from a Screener data table."""
    thead = re.search(r"<thead.*?</thead>", block, re.S)
    if not thead:
        return [], {}
    cols = [re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<th[^>]*>(.*?)</th>", thead.group(0), re.S)]
    cols = [c for c in cols if c]
    rows = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", block, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if not cells:
            continue
        label = html.unescape(re.sub(r"<[^>]+>", "", cells[0])).replace("\xa0", " ").strip()
        rows[label] = [re.sub(r"<[^>]+>", "", c).strip() for c in cells[1:]]
    return cols, rows


def _section(txt: str, sec_id: str) -> str:
    m = re.search(rf'id="{sec_id}".*?</section>', txt, re.S)
    return m.group(0) if m else ""


def _series(rows: dict, names: tuple[str, ...]) -> list:
    """First row whose label starts with any of `names`.

    Banks and NBFCs use Revenue/Financing Profit where most names use
    Sales/Operating Profit, so each field lists its alternatives in order.
    """
    for nm in names:
        for label, vals in rows.items():
            if label.replace("+", "").strip().lower().startswith(nm.lower()):
                return vals
    return []


_QUARTER_ROWS = {
    "revenue_cr": ("Sales", "Revenue"),
    "net_income_cr": ("Net Profit",),
    "operating_income_cr": ("Operating Profit", "Financing Profit"),
    "eps": ("EPS in Rs", "EPS"),
}
_ANNUAL_ROWS = {
    "revenue_cr": ("Sales", "Revenue"),
    "net_income_cr": ("Net Profit",),
    "operating_income_cr": ("Operating Profit", "Financing Profit"),
    "opm_pct": ("OPM %", "Financing Margin %"),
    "eps": ("EPS in Rs", "EPS"),
}
_SHARE_ROWS = {
    "promoters_pct": ("Promoters",),
    "fiis_pct": ("FIIs",),
    "diis_pct": ("DIIs",),
    "government_pct": ("Government",),
    "public_pct": ("Public",),
}


def _extract(txt: str, section: str, spec: dict, lag_days: int, kind: str) -> list[dict]:
    cols, rows = _table(_section(txt, section))
    if not cols:
        return []
    keyed = {k: _series(rows, names) for k, names in spec.items()}
    out = []
    for i, label in enumerate(cols):
        end = _period_end(label)
        if end is None:                   # TTM and friends have no period end
            continue
        rec = {
            "period": label,
            "period_end": end.isoformat(),
            # the ONLY date a backtest may filter on
            "available_from": (end + timedelta(days=lag_days)).isoformat(),
            "statement": kind,
        }
        has = False
        for k, vals in keyed.items():
            v = _num(vals[i]) if i < len(vals) else None
            rec[k] = v
            has = has or v is not None
        if has:
            out.append(rec)
    return out


def fundamentals_history(ticker: str) -> list[dict]:
    """All historical rows for one symbol, each stamped with `available_from`."""
    key = _key(ticker)
    txt = _get_page(key)
    if not txt:
        return []
    rows = (_extract(txt, "quarters", _QUARTER_ROWS, LAG_QUARTERLY, "quarterly")
            + _extract(txt, "profit-loss", _ANNUAL_ROWS, LAG_ANNUAL, "annual")
            + _extract(txt, "shareholding", _SHARE_ROWS, LAG_SHAREHOLDING, "shareholding"))
    for r in rows:
        r["symbol"] = key
    return rows


def point_in_time(panel, as_of: str | date, statement: str = "quarterly"):
    """Rows genuinely public on `as_of`. The correct way to consume this panel.

    Filters on `available_from`, never the period label - filtering on the label
    would hand the model results the market had not seen, which is the leak this
    module exists to prevent.
    """
    import pandas as pd
    df = panel if isinstance(panel, pd.DataFrame) else pd.DataFrame(panel)
    if df.empty:
        return df
    when = pd.Timestamp(as_of)
    df = df[df["statement"] == statement].copy()
    df["available_from"] = pd.to_datetime(df["available_from"])
    visible = df[df["available_from"] <= when]
    if visible.empty:
        return visible
    return (visible.sort_values("available_from")
            .groupby("symbol", as_index=False).last())


def build_panel(tickers: list[str], pause_s: float = 1.0) -> "list[dict]":
    rows = []
    for i, t in enumerate(tickers, 1):
        got = fundamentals_history(t)
        rows.extend(got)
        print(f"  [{i}/{len(tickers)}] {_key(t)}: {len(got)} rows")
        time.sleep(pause_s)
    return rows


def save_panel(rows: list[dict]) -> Path:
    import pandas as pd
    if not rows:
        return ARCHIVE
    new = pd.DataFrame(rows)
    if ARCHIVE.exists():
        new = pd.concat([pd.read_csv(ARCHIVE), new], ignore_index=True)
    new = new.drop_duplicates(subset=["symbol", "statement", "period"], keep="last")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["symbol", "statement", "period_end"]).to_csv(ARCHIVE, index=False)
    return ARCHIVE


def main():
    import argparse
    import glob
    import pandas as pd
    import yaml

    ap = argparse.ArgumentParser(description="Build a point-in-time fundamentals panel.")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="cap symbols (Screener is slow)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        if not ARCHIVE.exists():
            print("  no panel yet")
            return
        df = pd.read_csv(ARCHIVE)
        print(f"  {len(df)} rows | {df['symbol'].nunique()} symbols")
        for kind, g in df.groupby("statement"):
            print(f"  {kind:13} {g['period_end'].min()} -> {g['period_end'].max()} "
                  f"({g['period'].nunique()} periods)")
        return

    tickers = args.tickers
    if not tickers:
        cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
        uni = (cfg.get("cross_section", {}) or {}).get("universe") or []
        tickers = [t.replace(".NS", "") for t in uni]
    if not tickers:
        tickers = [Path(p).stem for p in
                   glob.glob(str(ROOT / "Data" / "Raw_Data" / "Universe" / "*.csv"))]
    if args.limit:
        tickers = tickers[:args.limit]
    if not tickers:
        raise SystemExit("no tickers")

    print(f"building panel for {len(tickers)} symbols ...")
    rows = build_panel(tickers)
    if not rows:
        raise SystemExit("Screener unreachable")
    path = save_panel(rows)
    df = pd.DataFrame(rows)
    print(f"{len(rows)} rows -> {path}")
    print(f"  earliest period_end {df['period_end'].min()}")
    print("  every row carries available_from = period_end + the SEBI disclosure "
          "lag; filter on that, never on the period label")


if __name__ == "__main__":
    main()
