"""Volatility and volume features for the Nifty 50 direction model.

Rolling realized volatility (std of daily returns) plus volume-magnitude
features. Extracted from the research notebook into a reusable module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_volatility(daily_ret: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Realized volatility = rolling std of daily returns, per window."""
    out = {}
    for w in windows:
        out[f"roll_vol_{w}"] = daily_ret.rolling(window=w).std()
    return pd.DataFrame(out)


def log_volume(volume: pd.Series) -> pd.Series:
    """Log-stabilized volume magnitude, log(volume + 1)."""
    return np.log(volume + 1)


def rolling_mean_volume(volume: pd.Series, window: int = 5) -> pd.Series:
    """Average traded volume over `window` days (market participation trend)."""
    return volume.rolling(window=window).mean()


def realized_vol_percentile(daily_ret: pd.Series, short: int = 20, long: int = 252) -> pd.DataFrame:
    """Volatility-regime framework: short/long realized vol and a rolling percentile rank."""
    out = pd.DataFrame(index=daily_ret.index)
    out[f"realized_vol_{short}"] = daily_ret.rolling(short).std()
    out[f"realized_vol_{long}"] = daily_ret.rolling(long).std()
    out["vol_regime_percentile"] = (
        out[f"realized_vol_{long}"]
        .rolling(long)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    return out


def add_volatility_features(
    df: pd.DataFrame,
    vol_windows: list[int],
    volume_mean_window: int,
) -> pd.DataFrame:
    """Attach volatility/volume features. Requires `daily_ret` and `volume` columns."""
    df = df.copy()
    df = df.join(rolling_volatility(df["daily_ret"], vol_windows))
    df["log_volume"] = log_volume(df["volume"])
    df["vol_roll_mean_5"] = rolling_mean_volume(df["volume"], volume_mean_window)
    return df
