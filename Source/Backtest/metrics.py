"""Quant evaluation metrics for the alpha-signal backtest.

All strategy metrics operate on NON-OVERLAPPING holding-period returns to avoid
the autocorrelation inflation that overlapping 20-day windows would introduce.
Transaction costs are charged per side in basis points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def total_cost_bps(cfg: dict) -> float:
    """Per-side cost in basis points (apply_costs doubles it to a round trip).

    Uses the itemized Indian cost stack when backtest.cost_model == "india",
    otherwise the legacy flat fees + slippage model.
    """
    bt = cfg["backtest"]
    if bt.get("cost_model", "flat") == "india":
        from Source.Backtest.costs import india_cost_breakdown
        return india_cost_breakdown(cfg)["per_side_bps"]
    return float(bt.get("transaction_cost_bps", 0) + bt.get("slippage_bps", 0))


def annualized_sharpe(period_returns: np.ndarray, periods_per_year: float) -> float:
    """Sharpe of per-period returns, annualized by sqrt(periods_per_year)."""
    period_returns = np.asarray(period_returns, dtype=float)
    if period_returns.size == 0:
        return 0.0
    sd = period_returns.std()
    if sd == 0:
        return 0.0
    return float(period_returns.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (fraction, negative)."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def apply_costs(positions: np.ndarray, gross_returns: np.ndarray, cost_bps: float) -> np.ndarray:
    """Net per-trade returns after round-trip transaction costs.

    Each trade opens and closes a `position`, so cost = |position| * 2 * bps.
    """
    positions = np.asarray(positions, dtype=float)
    cost = np.abs(positions) * 2.0 * (cost_bps / 1e4)
    return positions * gross_returns - cost


def equity_curve(net_returns: np.ndarray) -> np.ndarray:
    """Compound net per-period returns into an equity curve starting at 1.0."""
    return np.cumprod(1.0 + np.asarray(net_returns, dtype=float))


def non_overlapping(idx: np.ndarray, signal: np.ndarray, fwd_ret: np.ndarray, holding: int):
    """Subsample every `holding` steps so trades don't overlap. Returns (signal, fwd_ret) slices."""
    order = np.argsort(idx)
    sig = signal[order][::holding]
    ret = fwd_ret[order][::holding]
    return sig, ret


def strategy_report(
    signal: np.ndarray,
    fwd_ret: np.ndarray,
    idx: np.ndarray,
    cfg: dict,
    mode: str = "quantile",
) -> dict:
    """Full non-overlapping strategy report for one signal vector.

    mode:
      "sign"     -> long if signal>0 else short
      "quantile" -> long top pct, short bottom pct, flat middle
      "long"     -> long only when signal in top pct, else flat
    """
    bt = cfg["backtest"]
    holding = bt["holding_period"]
    ppy = bt["periods_per_year"] / holding
    cost_bps = total_cost_bps(cfg)

    sig, ret = non_overlapping(idx, signal, fwd_ret, holding)

    if mode == "sign":
        pos = np.sign(sig)
    elif mode == "long":
        thr = np.percentile(signal, bt["quantile_upper"])
        pos = (sig >= thr).astype(float)
    else:  # quantile long-short
        up = np.percentile(signal, bt["quantile_upper"])
        lo = np.percentile(signal, bt["quantile_lower"])
        pos = np.zeros_like(sig, dtype=float)
        pos[sig >= up] = 1.0
        pos[sig <= lo] = -1.0

    net = apply_costs(pos, ret, cost_bps)
    gross = pos * ret
    eq = equity_curve(net)

    return {
        "mode": mode,
        "sharpe_net": annualized_sharpe(net, ppy),
        "sharpe_gross": annualized_sharpe(gross, ppy),
        "mean_return": float(net.mean()) if net.size else 0.0,
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "max_drawdown": max_drawdown(eq),
        "avg_exposure": float(np.mean(np.abs(pos))) if pos.size else 0.0,
        "n_trades": int(pos.size),
        "hit_rate": float(np.mean(net > 0)) if net.size else 0.0,
        "equity_curve": [round(float(v), 5) for v in eq],
    }


def buy_and_hold_report(fwd_no: np.ndarray, cfg: dict) -> dict:
    """Passive long-only Nifty benchmark over the same non-overlapping periods.

    The strategy must beat this to add value - the index has a strong upward
    drift, so any long-biased signal looks good until compared here. Charges a
    single round-trip cost for the whole hold (enter once, exit once).
    """
    bt = cfg["backtest"]
    ppy = bt["periods_per_year"] / bt["holding_period"]
    eq = equity_curve(fwd_no)                       # always long, position = +1
    roundtrip = 2 * total_cost_bps(cfg) / 1e4
    net_total = float(eq[-1] * (1 - roundtrip) - 1) if eq.size else 0.0
    return {
        "mode": "buy_and_hold",
        "sharpe_net": annualized_sharpe(fwd_no, ppy),
        "total_return": net_total,
        "max_drawdown": max_drawdown(eq),
        "n_trades": int(fwd_no.size),
        "equity_curve": [round(float(v), 5) for v in eq],
    }


def per_horizon_classification(probs: np.ndarray, y_true: np.ndarray) -> list[dict]:
    """AUC / accuracy / majority-baseline for every horizon."""
    out = []
    for h in range(y_true.shape[1]):
        yt = y_true[:, h]
        p = probs[:, h]
        try:
            auc = float(roc_auc_score(yt, p))
        except ValueError:
            auc = float("nan")
        acc = float(np.mean((p > 0.5).astype(int) == yt))
        p_up = float(yt.mean())
        out.append({
            "horizon": h + 1,
            "auc": auc,
            "accuracy": acc,
            "baseline": max(p_up, 1 - p_up),
            "class_balance_up": p_up,
        })
    return out


def per_horizon_ic(probs: np.ndarray, fwd_returns_by_h: dict[int, np.ndarray]) -> list[dict]:
    """Spearman IC between each horizon's probability and its realized forward return."""
    out = []
    for h in range(probs.shape[1]):
        r = fwd_returns_by_h[h + 1]
        p = probs[:, h]
        mask = ~np.isnan(p) & ~np.isnan(r)
        if mask.sum() < 3:
            out.append({"horizon": h + 1, "ic": float("nan"), "p_value": float("nan")})
            continue
        ic, pv = spearmanr(p[mask], r[mask])
        out.append({"horizon": h + 1, "ic": float(ic), "p_value": float(pv)})
    return out


def decile_attribution(prob: np.ndarray, fwd_ret: np.ndarray) -> list[dict]:
    """Mean forward return per signal decile - tests monotonicity of the signal."""
    dfe = pd.DataFrame({"p": prob, "r": fwd_ret}).dropna()
    if len(dfe) < 10:
        return []
    dfe["decile"] = pd.qcut(dfe["p"], 10, labels=False, duplicates="drop")
    g = dfe.groupby("decile")["r"].mean()
    return [{"decile": int(d), "mean_return": float(v)} for d, v in g.items()]


def yearly_sharpe(dates: np.ndarray, net_returns: np.ndarray, periods_per_year: float) -> list[dict]:
    """Per-calendar-year annualized Sharpe of the strategy's net returns."""
    res = pd.DataFrame({"date": pd.to_datetime(dates), "r": net_returns})
    res["year"] = res["date"].dt.year
    out = []
    for yr, grp in res.groupby("year"):
        out.append({"year": int(yr), "sharpe": annualized_sharpe(grp["r"].to_numpy(), periods_per_year),
                    "n": int(len(grp))})
    return out
