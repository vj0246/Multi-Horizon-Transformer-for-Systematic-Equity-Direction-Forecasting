"""Intraday features - the part that genuinely differs from the daily track.

The architecture does not need to change at higher frequency (LightGBM and the
Transformer already agree at daily, so the model is not the constraint). What
changes is the INPUT: intraday bars carry structure that daily bars average
away, and none of it exists in the daily feature set.

  overnight gap     ~17 hours of world news lands between the 15:30 close and
                    the 09:15 open. This is the single largest intraday signal
                    candidate and is invisible daily.
  session position  first and last bars behave differently - opening auction
                    imbalance, closing rebalance flow.
  intraday range    where the bar closed inside its own high/low, a standard
                    pressure proxy.
  volume profile    volume relative to the same time-of-day historically, not to
                    an overall mean; intraday volume is strongly U-shaped, so a
                    flat mean would flag every open and close as anomalous.

All rolling statistics are causal (past-only), and every feature at bar t is
computed from data available before t closes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE = ["ret", "ret_5", "ret_20", "vol_20", "vol_60", "range_pos",
        "gap", "gap_z", "bar_of_day", "minutes_from_open",
        "rel_volume", "close_vs_vwap"]


def add_intraday_features(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    d = df.copy().sort_values("datetime").reset_index(drop=True)
    if d["datetime"].dt.tz is None:
        d["datetime"] = d["datetime"].dt.tz_localize("UTC")
    d["datetime"] = d["datetime"].dt.tz_convert(tz)
    d["session"] = d["datetime"].dt.date

    # ---- returns and realised vol (causal) ---------------------------------
    d["ret"] = np.log(d["close"]).diff()
    d["ret_5"] = np.log(d["close"]).diff(5)
    d["ret_20"] = np.log(d["close"]).diff(20)
    d["vol_20"] = d["ret"].rolling(20).std()
    d["vol_60"] = d["ret"].rolling(60).std()

    # ---- where the bar closed inside its own range -------------------------
    span = (d["high"] - d["low"]).replace(0, np.nan)
    d["range_pos"] = ((d["close"] - d["low"]) / span).fillna(0.5)

    # ---- overnight gap: only defined on the first bar of a session ----------
    first = d.groupby("session")["datetime"].transform("min") == d["datetime"]
    prev_close = d["close"].shift(1)
    raw_gap = np.log(d["open"] / prev_close)
    d["gap"] = np.where(first, raw_gap, 0.0)
    gap_sd = pd.Series(np.where(first, raw_gap, np.nan)).rolling(60, min_periods=10).std()
    d["gap_z"] = np.where(first, d["gap"] / gap_sd.replace(0, np.nan), 0.0)
    d["gap_z"] = pd.Series(d["gap_z"]).fillna(0.0)

    # ---- session position --------------------------------------------------
    d["bar_of_day"] = d.groupby("session").cumcount()
    open_time = d.groupby("session")["datetime"].transform("min")
    d["minutes_from_open"] = (d["datetime"] - open_time).dt.total_seconds() / 60.0

    # ---- volume relative to the SAME time of day, not to a flat mean -------
    # intraday volume is U-shaped; a flat mean would mark every open and close
    # as an outlier and carry a pure time-of-day signal instead of a surprise.
    tod = d["datetime"].dt.strftime("%H:%M")
    d["_tod_mean"] = (d.groupby(tod)["volume"]
                      .transform(lambda s: s.shift(1).expanding(min_periods=5).mean()))
    d["rel_volume"] = (d["volume"] / d["_tod_mean"].replace(0, np.nan)).fillna(1.0)

    # ---- close vs session VWAP (causal within the session) -----------------
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    cum_pv = (tp * d["volume"]).groupby(d["session"]).cumsum()
    cum_v = d["volume"].groupby(d["session"]).cumsum().replace(0, np.nan)
    d["close_vs_vwap"] = (d["close"] / (cum_pv / cum_v) - 1.0).fillna(0.0)

    d = d.drop(columns=["_tod_mean"])
    return d.replace([np.inf, -np.inf], np.nan)


def make_targets(df: pd.DataFrame, horizons: int) -> pd.DataFrame:
    """Direction targets in BAR units: close[t+h] > close[t]."""
    d = df.copy()
    for h in range(1, horizons + 1):
        d[f"target_{h}"] = (d["close"].shift(-h) > d["close"]).astype("float32")
    return d


def resolve_features(use_sentiment: bool) -> list[str]:
    return BASE + (["news_tone", "news_tone_chg", "has_news"] if use_sentiment else [])
