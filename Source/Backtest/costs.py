"""Comprehensive India transaction-cost model (FY2024-25 regime).

Every statutory charge an Indian trader actually pays, itemized for a full round
trip (one buy leg + one sell leg), returned in basis points of notional. Rates
are the post-Oct-2024 F&O regime at a discount broker (Zerodha-style flat/near-
zero brokerage); they are modeled averages, not a broker-exact contract note.

Per-side statutory rates (bps of turnover), sourced from the NSE/SEBI schedule
and broker charge lists:

  instrument   brokerage   exchange_txn   STT(buy/sell)   stamp(buy)   SEBI
  delivery     0.0         0.297          10.0 / 10.0     1.5          0.01
  intraday     0.4         0.297          0.0  / 2.5      0.3          0.01
  futures      0.4         0.173          0.0  / 2.0      0.2          0.01
  options*     flat/lot    3.503(prem)    0.0  / 10.0     0.3          0.01

  STT: delivery 0.1% both sides; intraday 0.025% sell; futures 0.02% sell
       (raised from 0.0125% Oct-2024); options 0.1% on premium sell.
  Exchange txn (NSE): delivery/intraday 0.00297%; futures 0.00173%;
       options 0.03503% of PREMIUM.
  Stamp duty (buy only, 2020 uniform): delivery 0.015%; intraday 0.003%;
       futures 0.002%; options 0.003%.
  SEBI turnover: 0.0001% (Rs 10/crore) all segments.
  GST: 18% on (brokerage + exchange txn + SEBI).
  DP charge: delivery SELL only, ~Rs 15.93 flat per scrip incl GST -> added in
       bps only when a notional is supplied (backtest.india.notional_inr).

*Options are premium-based, a different notional base, so their bps figure is
 approximate; the tradable strategies here use futures / delivery.

Prior versions of this file carried a 10x error: futures stamp at 2.0 bps rather
than 0.2 bps (0.002%). Fixed here; it lowers the futures round trip ~1.8 bps.
"""
from __future__ import annotations

# Per-side statutory charges in basis points (except stt split buy/sell).
INSTRUMENTS = {
    "delivery": {
        "brokerage_bps": 0.0, "exchange_txn_bps": 0.297, "sebi_bps": 0.01,
        "stt_buy_bps": 10.0, "stt_sell_bps": 10.0, "stamp_buy_bps": 1.5,
        "default_slippage_bps": 4.0, "dp_on_sell": True,
    },
    "intraday": {
        "brokerage_bps": 0.4, "exchange_txn_bps": 0.297, "sebi_bps": 0.01,
        "stt_buy_bps": 0.0, "stt_sell_bps": 2.5, "stamp_buy_bps": 0.3,
        "default_slippage_bps": 3.0, "dp_on_sell": False,
    },
    "futures": {
        "brokerage_bps": 0.4, "exchange_txn_bps": 0.173, "sebi_bps": 0.01,
        "stt_buy_bps": 0.0, "stt_sell_bps": 2.0, "stamp_buy_bps": 0.2,
        "default_slippage_bps": 3.0, "dp_on_sell": False,
    },
    "options": {
        "brokerage_bps": 0.4, "exchange_txn_bps": 3.503, "sebi_bps": 0.01,
        "stt_buy_bps": 0.0, "stt_sell_bps": 10.0, "stamp_buy_bps": 0.3,
        "default_slippage_bps": 25.0, "dp_on_sell": False,
    },
}
DP_CHARGE_INR = 15.93            # per-scrip delivery-sell demat charge incl GST


def india_cost_breakdown(cfg: dict, instrument: str | None = None) -> dict:
    """Round-trip Indian cost breakdown (buy leg + sell leg), in basis points.

    Statutory rates come from INSTRUMENTS; slippage and GST% (and an optional
    notional for the flat DP charge) come from config `backtest.india`. Returns
    each component plus `roundtrip_bps` (open+close) and `per_side_bps`
    (roundtrip / 2, what metrics.apply_costs charges per position and doubles).
    """
    ind = cfg["backtest"].get("india", {})
    inst = instrument or ind.get("instrument", "futures")
    if inst not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{inst}'. Use one of {list(INSTRUMENTS)}.")
    s = INSTRUMENTS[inst]

    # slippage: config overrides the instrument default; GST% configurable
    slippage_bps = float(ind.get("slippage_bps", s["default_slippage_bps"]))
    gst_pct = float(ind.get("gst_pct", 18.0)) / 100.0

    brokerage = s["brokerage_bps"] * 2
    exchange = s["exchange_txn_bps"] * 2
    sebi = s["sebi_bps"] * 2
    slippage = slippage_bps * 2
    stt = s["stt_buy_bps"] + s["stt_sell_bps"]
    stamp = s["stamp_buy_bps"]                       # buy leg only
    gst = gst_pct * (brokerage + exchange + sebi)    # GST on brokerage+txn+SEBI

    dp = 0.0
    if s.get("dp_on_sell") and ind.get("notional_inr"):
        dp = DP_CHARGE_INR / float(ind["notional_inr"]) * 1e4   # flat -> bps

    roundtrip = brokerage + exchange + sebi + slippage + stt + stamp + gst + dp
    return {
        "instrument": inst,
        "brokerage_bps": round(brokerage, 4),
        "exchange_txn_bps": round(exchange, 4),
        "sebi_bps": round(sebi, 4),
        "stt_bps": round(stt, 4),
        "stamp_duty_bps": round(stamp, 4),
        "gst_bps": round(gst, 4),
        "dp_charge_bps": round(dp, 4),
        "slippage_bps": round(slippage, 4),
        "roundtrip_bps": round(roundtrip, 4),
        "per_side_bps": round(roundtrip / 2, 4),
    }
