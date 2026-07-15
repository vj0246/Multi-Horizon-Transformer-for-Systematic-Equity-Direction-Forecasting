"""Volatility-targeted position sizing.

Scales each period's exposure inversely to the strategy's own recent realized
volatility, so the book runs at a roughly constant risk budget instead of a
fixed unit position. The trailing-vol estimate is lagged by one period so the
size for period t uses only information available at t-1 (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_weights(
    period_returns: np.ndarray,
    target_vol_annual: float,
    periods_per_year: float,
    lookback: int = 6,
    max_leverage: float = 2.0,
) -> np.ndarray:
    """Per-period leverage that targets `target_vol_annual`.

    weight_t = clip( target_vol / annualized_trailing_vol_{t-1}, 0, max_leverage ).
    Early periods without enough history get weight 1 (unscaled).
    """
    r = pd.Series(np.asarray(period_returns, dtype=float))
    ann = np.sqrt(periods_per_year)
    trailing_vol = (r.rolling(lookback).std().shift(1) * ann).to_numpy()  # shift => no look-ahead
    # trailing_vol is NaN during warmup and 0 for a flat window; the mask maps both
    # to weight 1.0, so ratio's inf/nan there never survive.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = target_vol_annual / trailing_vol
    w = np.where(trailing_vol > 0, ratio, 1.0)
    return np.clip(w, 0.0, max_leverage)


def apply_vol_target(
    period_returns: np.ndarray,
    cfg: dict,
    periods_per_year: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scaled_returns, weights) applying the config risk budget to a
    per-period return stream (e.g. a strategy's net returns)."""
    rk = cfg.get("risk", {})
    w = vol_target_weights(
        period_returns,
        target_vol_annual=float(rk.get("target_vol_annual", 0.15)),
        periods_per_year=periods_per_year,
        lookback=int(rk.get("lookback", 6)),
        max_leverage=float(rk.get("max_leverage", 2.0)),
    )
    return w * np.asarray(period_returns, dtype=float), w
