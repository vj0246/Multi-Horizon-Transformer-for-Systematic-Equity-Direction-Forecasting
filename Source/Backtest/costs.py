"""India-specific transaction cost model for the Nifty 50 backtest.

The Nifty 50 is an index, not a tradable security, so a real directional
strategy trades either **index futures** (default here - allows the short leg
the quantile long-short needs) or a **delivery ETF** (long-only). This module
itemizes every statutory Indian charge for a full round trip (one buy leg + one
sell leg) and returns the total in basis points.

Rates reflect the Indian regime as of FY2024-25 (post the Oct-2024 F&O STT
hike). They are configurable in config.yaml under `backtest.india`. All figures
are per-notional basis points unless stated. These are modeled estimates, not a
broker-exact contract note.
"""
from __future__ import annotations

# Instrument-dependent statutory charges (basis points).
#   STT  - Securities Transaction Tax
#   Stamp duty - charged on the BUY leg only (uniform 2020 regime)
INSTRUMENTS = {
    "futures": {           # NSE index futures
        "stt_buy_bps": 0.0,
        "stt_sell_bps": 2.0,   # 0.02% on sell (raised from 0.0125% in Oct 2024)
        "stamp_buy_bps": 2.0,  # 0.002% on buy
    },
    "delivery": {          # Nifty ETF held to delivery (long-only)
        "stt_buy_bps": 10.0,   # 0.1% on both buy and sell
        "stt_sell_bps": 10.0,
        "stamp_buy_bps": 1.5,  # 0.015% on buy
    },
}


def india_cost_breakdown(cfg: dict) -> dict:
    """Round-trip Indian cost breakdown (buy leg + sell leg), in basis points.

    Returns each component plus `roundtrip_bps` (full open+close) and
    `per_side_bps` (roundtrip / 2, the value the backtester charges per position
    via metrics.apply_costs, which doubles it back to a round trip).
    """
    ind = cfg["backtest"]["india"]
    inst = ind.get("instrument", "futures")
    if inst not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{inst}'. Use one of {list(INSTRUMENTS)}.")
    stat = INSTRUMENTS[inst]

    brokerage_bps = float(ind.get("brokerage_bps", 0.3))
    slippage_bps = float(ind.get("slippage_bps", 3.0))
    exch_bps = float(ind.get("exchange_txn_bps", 0.19))
    sebi_bps = float(ind.get("sebi_bps", 0.01))
    gst_pct = float(ind.get("gst_pct", 18.0)) / 100.0

    # Two legs (buy + sell) for the round trip.
    brokerage = brokerage_bps * 2
    exchange = exch_bps * 2
    sebi = sebi_bps * 2
    slippage = slippage_bps * 2
    stt = stat["stt_buy_bps"] + stat["stt_sell_bps"]
    stamp = stat["stamp_buy_bps"]                 # buy leg only
    gst = gst_pct * (brokerage + exchange)        # GST applies to brokerage + exchange charges

    roundtrip = brokerage + exchange + sebi + slippage + stt + stamp + gst
    return {
        "instrument": inst,
        "brokerage_bps": round(brokerage, 4),
        "exchange_txn_bps": round(exchange, 4),
        "sebi_bps": round(sebi, 4),
        "stt_bps": round(stt, 4),
        "stamp_duty_bps": round(stamp, 4),
        "gst_bps": round(gst, 4),
        "slippage_bps": round(slippage, 4),
        "roundtrip_bps": round(roundtrip, 4),
        "per_side_bps": round(roundtrip / 2, 4),
    }
