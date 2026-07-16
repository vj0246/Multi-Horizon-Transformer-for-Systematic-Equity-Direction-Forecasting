"""Macro / market-state features from auxiliary series and universe breadth.

These add information the ^NSEI price series does not contain. Every feature is
stationary or bounded, and every external series is aligned with a strict
no-look-ahead rule:

    merge_asof(direction="backward")  -> last value at or before the NSE date
    .shift(1)                         -> row t carries only information from <= t-1

Combined with the 60-day window ending at t-1 (see dataset.make_windows), a
feature at row t-1 is known well before the decision is taken at close t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MACRO_DIR = ROOT / "Data" / "Raw_Data" / "Macro"


def _load_series(name: str) -> pd.DataFrame | None:
    path = MACRO_DIR / f"{name}.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return s[["date", "close"]].rename(columns={"close": name.lower()})


def _asof_lagged(df: pd.DataFrame, aux: pd.DataFrame, col: str) -> pd.Series:
    """Last aux value at or before each NSE date, then shifted one row."""
    merged = pd.merge_asof(
        df[["date"]].sort_values("date"), aux.sort_values("date"),
        on="date", direction="backward",
    )
    return merged[col].shift(1).to_numpy()


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach India VIX / S&P 500 / USDINR / crude derived features."""
    df = df.copy()

    vix = _load_series("INDIAVIX")
    if vix is not None:
        v = pd.Series(_asof_lagged(df, vix, "indiavix"), index=df.index)
        df["vix_chg"] = np.log(v / v.shift(1))
        df["vix_z60"] = (v - v.rolling(60).mean()) / v.rolling(60).std()

    spx = _load_series("GSPC")
    if spx is not None:
        s = pd.Series(_asof_lagged(df, spx, "gspc"), index=df.index)
        df["spx_ret"] = np.log(s / s.shift(1))          # overnight US move
        df["spx_ret_5"] = np.log(s / s.shift(5))

    fx = _load_series("USDINR")
    if fx is not None:
        f = pd.Series(_asof_lagged(df, fx, "usdinr"), index=df.index)
        df["usdinr_ret"] = np.log(f / f.shift(1))

    oil = _load_series("CRUDE")
    if oil is not None:
        o = pd.Series(_asof_lagged(df, oil, "crude"), index=df.index)
        df["crude_ret"] = np.log(o / o.shift(1))

    return df.replace([np.inf, -np.inf], np.nan)


def add_breadth_features(df: pd.DataFrame, universe_dir: Path, min_names: int = 20) -> pd.DataFrame:
    """Market breadth computed from the constituent universe.

    breadth_above_ma20: fraction of names trading above their own 20-day MA.
    xs_dispersion:      cross-sectional std of that day's returns.
    Both use only same-date closes, and are then shifted one row so a feature at
    row t reflects breadth as of <= t-1.
    """
    df = df.copy()
    files = sorted(universe_dir.glob("*.csv")) if universe_dir.exists() else []
    if len(files) < min_names:
        return df

    closes = {}
    for p in files:
        s = pd.read_csv(p, parse_dates=["date"]).sort_values("date").set_index("date")["close"]
        closes[p.stem] = s
    wide = pd.DataFrame(closes).sort_index()

    above = (wide > wide.rolling(20).mean()).sum(axis=1) / wide.notna().sum(axis=1)
    disp = np.log(wide / wide.shift(1)).std(axis=1)

    br = pd.DataFrame({"date": above.index, "breadth_above_ma20": above.to_numpy(),
                       "xs_dispersion": disp.to_numpy()})
    for col in ("breadth_above_ma20", "xs_dispersion"):
        df[col] = _asof_lagged(df, br[["date", col]], col)
    return df.replace([np.inf, -np.inf], np.nan)
