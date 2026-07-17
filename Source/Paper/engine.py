"""Paper-trading book: a long/flat index timer, marked to market daily.

Honest by construction. This paper-trades the SAME primary strategy the backtest
reports (rolling-threshold long/flat timing on the ensemble signal), on real
NSE closes, charged the real India futures round-trip cost. It cannot show a
profit the backtest says is not there - it is the forward, out-of-sample proof,
not a demo rigged to look good.

Mechanics (one call per trading day):
- If in a position, the day's index return accrues to equity.
- Every `holding_period` days the position is re-decided from the fresh signal
  vs its rolling threshold. Entering or exiting charges `per_side_bps` once.
- Buy-and-hold equity is tracked alongside as the benchmark.

State is a plain dict so it round-trips through JSON between daily runs.
"""
from __future__ import annotations


def new_state(start_equity: float = 100.0) -> dict:
    return {
        "start_equity": start_equity,
        "equity": start_equity,
        "bh_equity": start_equity,
        "position": 0,            # 0 = cash, 1 = long
        "days_held": 0,
        "history": [],            # [{date, close, position, equity, bh_equity}]
        "trades": [],             # [{date, action, price, equity}]
        "last_date": None,
    }


def step(state: dict, date: str, close: float, prev_close: float | None,
         target_position: int, per_side_bps: float, holding_period: int) -> dict:
    """Advance the book one trading day. Idempotent per date (re-runs are ignored)."""
    if state["last_date"] == date:
        return state                                   # already processed today

    ret = (close / prev_close - 1.0) if prev_close else 0.0
    cost_frac = per_side_bps / 1e4

    # 1) accrue the day's move on yesterday's position, always accrue for buy-hold
    if state["position"] == 1:
        state["equity"] *= (1 + ret)
    state["bh_equity"] *= (1 + ret)
    if state["position"] == 1:
        state["days_held"] += 1

    # 2) rebalance decision only at the horizon boundary (or when currently flat)
    rebalance = state["position"] == 0 or state["days_held"] >= holding_period
    if rebalance and target_position != state["position"]:
        # close an open position / open a new one - one side of cost each event
        if state["position"] == 1:                     # exit
            state["equity"] *= (1 - cost_frac)
            state["trades"].append({"date": date, "action": "SELL", "price": close,
                                    "equity": round(state["equity"], 4)})
        if target_position == 1:                        # enter
            state["equity"] *= (1 - cost_frac)
            state["trades"].append({"date": date, "action": "BUY", "price": close,
                                    "equity": round(state["equity"], 4)})
        state["position"] = target_position
        state["days_held"] = 0
    elif rebalance and state["position"] == 1:
        state["days_held"] = 0                          # held through, reset the clock

    state["history"].append({
        "date": date, "close": round(close, 2), "position": state["position"],
        "equity": round(state["equity"], 4), "bh_equity": round(state["bh_equity"], 4),
    })
    state["last_date"] = date
    return state


def summary(state: dict, periods_per_year: float = 252.0) -> dict:
    """Return-and-risk summary of the paper book vs buy-and-hold."""
    import numpy as np
    h = state["history"]
    if len(h) < 2:
        return {"n_days": len(h)}
    eq = np.array([r["equity"] for r in h], dtype=float)
    bh = np.array([r["bh_equity"] for r in h], dtype=float)
    daily = eq[1:] / eq[:-1] - 1
    peak = np.maximum.accumulate(eq)
    sd = daily.std()
    return {
        "n_days": len(h),
        "total_return": float(eq[-1] / eq[0] - 1),
        "buy_hold_return": float(bh[-1] / bh[0] - 1),
        "excess_return": float(eq[-1] / eq[0] - bh[-1] / bh[0]),
        "sharpe": float(daily.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "max_drawdown": float(((eq - peak) / peak).min()),
        "n_trades": len(state["trades"]),
        "current_position": "LONG" if state["position"] == 1 else "CASH",
        "time_in_market": float(np.mean([r["position"] for r in h])),
    }
