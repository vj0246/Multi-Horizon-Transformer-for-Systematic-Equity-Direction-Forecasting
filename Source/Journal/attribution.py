"""Trade journal and P&L attribution - "learn from mistakes", honestly.

The intuitive version of this idea is reinforcement learning: watch the P&L,
punish the losers, stop repeating them. That does not work on this data, for two
separate reasons, and the second is the important one.

  1. SAMPLE SIZE. The paper book has made 13 round trips in 674 out-of-sample
     days. Tabular Q-learning needs ~10,000 decisions; DQN/PPO need ~1e6. We are
     three to five orders of magnitude short.

  2. THE PREMISE IS WRONG. Reinforcement assumes a loss is a mistake. With no
     measured edge, most losses are noise realising against you. An agent that
     learns not to repeat a random loss has fitted noise - it will do WORSE than
     the rule it replaced while appearing to have learned something. That is the
     failure mode this whole repository exists to avoid.

So the journal does the part that is real: it decomposes every closed trade into
components that CAN be attributed, and separates the ones that are diagnostic
from the one that is not.

  signal error   direction was wrong                  -> diagnostic, but noisy
  cost drag      direction right, fees ate the move   -> deterministic, real
  exposure       flat while the market moved          -> deterministic, real
  noise          |move| below the signal's resolution -> NOT a mistake

The cost and exposure components are deterministic - they are arithmetic, not
inference, and can be acted on immediately. The direction component is the one
that needs a significance test before anyone calls it a lesson, which is why
`summary()` reports a binomial test on hit rate rather than a raw win count.
"""
from __future__ import annotations

import math


def _binom_p(hits: int, n: int, p0: float = 0.5) -> float:
    """Two-sided binomial p-value for hits/n against a fair coin.

    Exact rather than normal-approximated: with ~13 trades the normal
    approximation is badly wrong, and this is precisely the regime where an
    over-confident p-value would turn noise into a 'lesson learned'.
    """
    if n == 0:
        return float("nan")

    def pmf(k):
        return math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k)

    obs = pmf(hits)
    return float(min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= obs + 1e-12)))


def build_trades(history: list[dict], per_side_bps: float) -> list[dict]:
    """Pair BUY/SELL events from the daily book into closed round trips."""
    trades, open_leg = [], None
    for row in history:
        pos = row.get("position", 0)
        if open_leg is None and pos == 1:
            open_leg = row
        elif open_leg is not None and pos == 0:
            trades.append(_close(open_leg, row, per_side_bps))
            open_leg = None
    if open_leg is not None:                       # still open - report separately
        trades.append({**_close(open_leg, history[-1], per_side_bps), "open": True})
    return trades


def _close(entry: dict, exit_: dict, per_side_bps: float) -> dict:
    gross = exit_["close"] / entry["close"] - 1.0
    cost = 2 * per_side_bps / 1e4                  # both sides of the round trip
    net = gross - cost
    return {
        "entry_date": entry["date"], "exit_date": exit_["date"],
        "entry_price": float(entry["close"]), "exit_price": float(exit_["close"]),
        "days_held": int(exit_.get("i", 0) - entry.get("i", 0)),
        "gross_return": float(gross),
        "cost_return": float(-cost),
        "net_return": float(net),
        "direction_correct": bool(gross > 0),
        "open": False,
    }


def classify(trade: dict, noise_floor: float) -> str:
    """Label a closed trade by what actually drove its result.

    `noise_floor` is the move size below which the signal has no resolution -
    typically one daily standard deviation scaled to the holding period. Trades
    inside it are labelled `noise`, NOT mistakes: calling them errors and
    'learning' from them is exactly how a system fits randomness.
    """
    g, n = trade["gross_return"], trade["net_return"]
    if abs(g) < noise_floor:
        return "noise" if n < 0 else "noise_win"
    if g > 0 and n < 0:
        return "cost_drag"          # right call, fees ate it - deterministic
    if g > 0:
        return "win"
    return "signal_error"           # genuinely wrong direction, beyond noise


def summary(trades: list[dict], noise_floor: float) -> dict:
    closed = [t for t in trades if not t.get("open")]
    if not closed:
        return {"n_trades": 0, "note": "no closed trades yet"}

    for t in closed:
        t["category"] = classify(t, noise_floor)

    cats: dict[str, int] = {}
    for t in closed:
        cats[t["category"]] = cats.get(t["category"], 0) + 1

    n = len(closed)
    hits = sum(t["direction_correct"] for t in closed)
    gross = sum(t["gross_return"] for t in closed)
    costs = sum(t["cost_return"] for t in closed)
    p = _binom_p(hits, n)

    # deterministic components: arithmetic, actionable without a significance test
    cost_drag = [t for t in closed if t["category"] == "cost_drag"]
    noise_trades = [t for t in closed if t["category"].startswith("noise")]

    return {
        "n_trades": n,
        "hit_rate": hits / n,
        "hit_rate_pvalue": p,
        "hit_rate_is_significant": bool(p < 0.05),
        "gross_return_sum": float(gross),
        "cost_return_sum": float(costs),
        "cost_share_of_gross": float(abs(costs) / abs(gross)) if gross else None,
        "categories": cats,
        "noise_floor": float(noise_floor),
        "actionable": {
            "cost_drag_trades": len(cost_drag),
            "cost_drag_cost": float(sum(t["cost_return"] for t in cost_drag)),
            "note": (
                "Cost drag is deterministic - these trades had the direction right "
                "and lost to fees. Fewer, longer holds fix it without any model "
                "change. This is the only component that can be acted on without "
                "a significance test."
            ),
        },
        "not_actionable": {
            "noise_trades": len(noise_trades),
            "note": (
                f"{len(noise_trades)} of {n} trades moved less than the "
                f"{noise_floor:.2%} noise floor. Their sign carries no "
                "information. Treating them as mistakes to learn from is how a "
                "system fits randomness."
            ),
        },
        "verdict": (
            f"Hit rate {hits}/{n} = {hits/n:.1%}, binomial p = {p:.3f}. "
            + ("Distinguishable from a coin flip."
               if p < 0.05 else
               f"NOT distinguishable from a coin flip - with {n} trades, nothing "
               "here supports 'the model learned from a mistake'. The only "
               "reliable lessons in this journal are the deterministic ones "
               "(cost drag, exposure).")
        ),
    }
