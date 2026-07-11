"""Return-based features for the Nifty 50 direction model.

Extracted from the research notebook into a reusable module. Every feature is
constructed to be stationary or bounded - no raw price levels leak in here.
"""
from __future__ import annotations

import pandas as pd


def daily_return(close: pd.Series) -> pd.Series:
    """Simple daily percentage return: (P_t - P_{t-1}) / P_{t-1}."""
    return close.pct_change()


def rolling_mean_returns(daily_ret: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Average daily return over each rolling window (directional momentum)."""
    out = {}
    for w in windows:
        out[f"roll_mean_ret_{w}"] = daily_ret.rolling(window=w).mean()
    return pd.DataFrame(out)


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Cumulative percentage price change over `period` days."""
    return close.pct_change(periods=period)


def ma_diff(close: pd.Series, window: int = 10) -> pd.Series:
    """Mean-reversion signal: distance of price from its moving average, (P - MA) / MA."""
    ma = close.rolling(window=window).mean()
    return (close - ma) / ma


def add_return_features(
    df: pd.DataFrame,
    return_windows: list[int],
    momentum_period: int,
    ma_diff_window: int,
) -> pd.DataFrame:
    """Attach all return-derived features to a copy of `df`. Requires a `close` column."""
    df = df.copy()
    df["daily_ret"] = daily_return(df["close"])
    df = df.join(rolling_mean_returns(df["daily_ret"], return_windows))
    df[f"momentum_{momentum_period}"] = momentum(df["close"], momentum_period)
    df[f"ma_diff_{ma_diff_window}"] = ma_diff(df["close"], ma_diff_window)
    return df
