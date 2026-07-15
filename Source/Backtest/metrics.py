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
    mode: str = "timing",
    threshold_ref: np.ndarray | None = None,
    holding: int | None = None,
) -> dict:
    """Full non-overlapping strategy report for one signal vector.

    mode:
      "sign"           -> long if signal>0 else short
      "quantile"       -> long top pct, short bottom pct, flat middle (cross-
                          sectional construct; kept for reference on a single index)
      "timing"/"long"  -> long/flat market timing: long when signal in top pct,
                          in cash otherwise. The honest single-index framing.

    threshold_ref: distribution percentile thresholds are computed from. Pass the
    VALIDATION-period signal so the entry rule is fixed without peeking at test
    data; defaults to `signal` itself (legacy in-sample behavior).
    holding: override holding period in days (defaults to config holding_period).
    """
    bt = cfg["backtest"]
    holding = holding or bt["holding_period"]
    ppy = bt["periods_per_year"] / holding
    cost_bps = total_cost_bps(cfg)
    ref = np.asarray(threshold_ref, dtype=float) if threshold_ref is not None else signal

    sig, ret = non_overlapping(idx, signal, fwd_ret, holding)

    if mode == "sign":
        pos = np.sign(sig)
    elif mode in ("timing", "long"):
        thr = np.percentile(ref, bt["quantile_upper"])
        pos = (sig >= thr).astype(float)
    else:  # quantile long-short
        up = np.percentile(ref, bt["quantile_upper"])
        lo = np.percentile(ref, bt["quantile_lower"])
        pos = np.zeros_like(sig, dtype=float)
        pos[sig >= up] = 1.0
        pos[sig <= lo] = -1.0

    net = apply_costs(pos, ret, cost_bps)
    gross = pos * ret
    return report_from_returns(net, gross, np.abs(pos), ppy, mode, holding)


def report_from_returns(net, gross, abs_pos, ppy: float, mode: str, holding: int) -> dict:
    """Assemble the canonical strategy-report dict from per-period return streams.

    Shared by strategy_report and any transformed book (e.g. the vol-targeted
    variant) so the exported schema stays in one place. `gross_returns` + `abs_pos`
    let the frontend recompute net = gross - abs_pos * 2 * (bps/1e4) at any cost.
    """
    net = np.asarray(net, dtype=float)
    gross = np.asarray(gross, dtype=float)
    abs_pos = np.asarray(abs_pos, dtype=float)
    eq = equity_curve(net)
    return {
        "mode": mode,
        "holding_days": int(holding),
        "sharpe_net": annualized_sharpe(net, ppy),
        "sharpe_gross": annualized_sharpe(gross, ppy),
        "mean_return": float(net.mean()) if net.size else 0.0,
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "max_drawdown": max_drawdown(eq),
        "avg_exposure": float(np.mean(abs_pos)) if abs_pos.size else 0.0,
        "n_trades": int(net.size),
        "hit_rate": float(np.mean(net > 0)) if net.size else 0.0,
        "net_returns": [round(float(v), 6) for v in net],
        "equity_curve": [round(float(v), 5) for v in eq],
        "gross_returns": [round(float(v), 6) for v in gross],
        "abs_pos": [round(float(v), 3) for v in abs_pos],
        "periods_per_year": float(ppy),
    }


def bootstrap_sharpe_ci(
    period_returns: np.ndarray,
    periods_per_year: float,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> list[float]:
    """Bootstrap confidence interval for the annualized Sharpe.

    With only ~30 non-overlapping trades a point-estimate Sharpe is mostly noise;
    the CI makes that uncertainty explicit instead of hiding it.
    """
    r = np.asarray(period_returns, dtype=float)
    if r.size < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(r, size=r.size, replace=True)
        sd = s.std()
        stats[i] = 0.0 if sd == 0 else s.mean() / sd * np.sqrt(periods_per_year)
    lo, hi = np.percentile(stats, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return [float(lo), float(hi)]


def calibrate_probs(
    logits_val: np.ndarray, y_val: np.ndarray, logits_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Platt-scale each horizon's logit into a calibrated probability.

    The BCE-trained logits are fine for ranking but their sigmoid is not a
    trustworthy P(up). A per-horizon logistic fit on the VALIDATION set (never
    test) maps logit -> calibrated probability. Rank metrics (AUC/IC) are
    unchanged; thresholds and reliability become meaningful.
    """
    from sklearn.linear_model import LogisticRegression

    n_h = logits_val.shape[1]
    cal_val = np.zeros_like(logits_val, dtype=float)
    cal_test = np.zeros_like(logits_test, dtype=float)
    for h in range(n_h):
        try:
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(logits_val[:, h:h + 1], y_val[:, h])
            cal_val[:, h] = lr.predict_proba(logits_val[:, h:h + 1])[:, 1]
            cal_test[:, h] = lr.predict_proba(logits_test[:, h:h + 1])[:, 1]
        except ValueError:  # degenerate single-class horizon: fall back to sigmoid
            cal_val[:, h] = 1 / (1 + np.exp(-logits_val[:, h]))
            cal_test[:, h] = 1 / (1 + np.exp(-logits_test[:, h]))
    return cal_val, cal_test


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability-diagram bins: mean predicted probability vs observed up-frequency."""
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = which == b
        if m.sum() == 0:
            continue
        out.append({
            "bin_mid": float((edges[b] + edges[b + 1]) / 2),
            "predicted": float(p[m].mean()),
            "observed": float(y[m].mean()),
            "count": int(m.sum()),
        })
    return out


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
