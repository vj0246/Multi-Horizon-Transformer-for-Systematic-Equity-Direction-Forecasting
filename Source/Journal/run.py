"""Trade journal, bandit diagnostic, and commentary -> journal.json.

Answers "monitor the P&L and learn from mistakes" with the parts of that idea the
data can actually support:

  attribution  decompose every closed trade into signal / cost / noise, and test
               the hit rate against a coin flip before calling anything a lesson
  bandit       Thompson sampling over FIXED validated rules, reporting posterior
               overlap so its argmax is never mistaken for a decision
  commentary   plain-English explanation of the above (optional LLM, off by
               default, deterministic fallback otherwise)

Run:  python -m Source.Journal.run
      python -m Source.Journal.run --llm     (needs GROQ_API_KEY or ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from Source.Backtest.costs import india_cost_breakdown
from Source.Journal import attribution, bandit

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "frontend" / "public" / "data" / "paper_trading.json"
STRATS = ROOT / "frontend" / "public" / "data" / "strategies.json"
OUT = ROOT / "frontend" / "public" / "data" / "journal.json"


def _noise_floor(curve: list[dict], holding: int) -> float:
    """One standard deviation of the holding-period move.

    Trades that move less than this are inside the signal's resolution; their
    sign is not information, so they must not be counted as mistakes.
    """
    closes = np.array([r["buy_hold"] for r in curve], dtype=float)
    if len(closes) < 30:
        return 0.0
    daily = np.diff(np.log(closes))
    return float(np.std(daily) * np.sqrt(holding))


def build_journal(cfg) -> dict:
    paper = json.loads(PAPER.read_text(encoding="utf-8"))
    per_side = india_cost_breakdown(cfg, "futures")["per_side_bps"]
    holding = cfg["backtest"]["holding_period"]

    curve = paper["equity_curve"]
    for i, r in enumerate(curve):
        r["i"] = i
        r["close"] = r["buy_hold"]              # index level, up to a constant factor

    trades = attribution.build_trades(curve, per_side)
    floor = _noise_floor(curve, holding)
    summ = attribution.summary(trades, floor)
    return {"trades": trades, "summary": summ, "noise_floor": floor}


def build_bandit() -> dict:
    """Replay a bandit over the published strategy variants as a separability test."""
    if not STRATS.exists():
        return {"skipped": "strategies.json not generated"}
    strat = json.loads(STRATS.read_text(encoding="utf-8"))
    arms = {k: v["net_returns"] for k, v in strat.items()
            if isinstance(v, dict) and v.get("net_returns")}
    if len(arms) < 2:
        return {"skipped": "need at least two arms with net_returns"}
    n = min(len(v) for v in arms.values())
    return bandit.replay({k: v[:n] for k, v in arms.items()})


def _fallback_commentary(j: dict, b: dict) -> dict:
    """Deterministic commentary. No API key, no network, always available."""
    s = j["summary"]
    n = s.get("n_trades", 0)
    sep = (b.get("separation") or {}).get("verdict", "not evaluated")
    return {
        "source": "template",
        "headline": (
            f"{n} closed trades: hit rate {s.get('hit_rate', 0):.0%}, "
            f"not distinguishable from chance"
            if not s.get("hit_rate_is_significant")
            else f"{n} closed trades: hit rate {s.get('hit_rate', 0):.0%}, significant"
        ),
        "what_happened": (
            f"The book closed {n} round trips. Gross return summed to "
            f"{s.get('gross_return_sum', 0):+.2%} and costs removed "
            f"{abs(s.get('cost_return_sum', 0)):.2%}."
        ),
        "what_it_means": s.get("verdict", ""),
        "caveats": [
            f"{n} trades is far too few to establish skill either way.",
            f"{s.get('not_actionable', {}).get('noise_trades', 0)} trades fell inside "
            "the noise floor and carry no directional information.",
            f"Bandit separability: {sep}",
        ],
    }


def _llm_commentary(j: dict, b: dict, cfg) -> dict:
    """LLM commentary on HISTORY ONLY. Structurally cannot emit a signal."""
    from Source.Advisor import prompts
    from Source.Advisor.client import complete_json

    # The guardrail that matters is this filter, not the prompt text: the model
    # is never given a forward prediction, so it has nothing to trade on.
    payload = {
        "closed_trades": j["summary"],
        "bandit_separation": (b.get("separation") or {}).get("verdict"),
        "note": "historical results only; no forward prediction is included",
    }

    def validate(o: dict) -> None:
        for k in ("headline", "what_happened", "what_it_means", "caveats"):
            if k not in o:
                raise ValueError(f"missing key {k}")
        if not isinstance(o["caveats"], list) or not o["caveats"]:
            raise ValueError("caveats must be a non-empty list")
        if len(str(o["headline"]).split()) > 25:
            raise ValueError("headline longer than 25 words")

    adv = cfg.get("advisor", {})
    r = complete_json(
        prompts.SYSTEM,
        prompts.USER_TEMPLATE.format(payload=json.dumps(payload, indent=1)),
        validate=validate,
        provider=adv.get("provider", "groq"),
        model=adv.get("model"),
    )
    return {"source": "llm", "provider": r["provider"], "model": r["model"],
            "prompt_version": prompts.PROMPT_VERSION, "usage": r["usage"],
            **r["json"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="generate commentary with an LLM (needs an API key)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))

    if not PAPER.exists():
        raise SystemExit("paper_trading.json missing - run Source.Paper.run first")

    j = build_journal(cfg)
    b = build_bandit()

    if args.llm:
        try:
            commentary = _llm_commentary(j, b, cfg)
        except Exception as e:                        # noqa: BLE001
            print(f"  LLM commentary failed ({e}); using deterministic fallback")
            commentary = _fallback_commentary(j, b)
    else:
        commentary = _fallback_commentary(j, b)

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": j["trades"][-1]["exit_date"] if j["trades"] else None,
        "attribution": j["summary"],
        "trades": j["trades"],
        "bandit": b,
        "commentary": commentary,
        "disclaimer": (
            "Commentary explains HISTORICAL results only. It is never shown a "
            "forward prediction, so it cannot and does not give trading advice. "
            "The underlying model has no validated edge."
        ),
    }, indent=2, default=str), encoding="utf-8")

    s = j["summary"]
    print(f"journal: {s.get('n_trades')} closed trades | hit rate "
          f"{s.get('hit_rate', 0):.1%} (p={s.get('hit_rate_pvalue', float('nan')):.3f}) "
          f"| categories {s.get('categories')}")
    if "separation" in b:
        print(f"  bandit: {b['separation']['verdict']}")
    print(f"  commentary: {commentary['source']}")


if __name__ == "__main__":
    main()
