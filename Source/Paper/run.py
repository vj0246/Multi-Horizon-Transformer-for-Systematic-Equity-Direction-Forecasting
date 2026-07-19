"""Paper-trading runner - one frozen model, everything after its training cutoff.

The frozen model (scripts/save_paper_model.py) trained only on train+val, so every
date from the validation cutoff onward - the whole test period AND every live day
since - is a true out-of-sample read. This scores those dates with that single
frozen model and marks a long/flat book (the deployed timing_rolling rule) to
market daily, charged the real India futures round trip.

  --refresh   re-download ^NSEI + macro + universe first (what the daily cron does)
  (default)   use the CSVs on disk

Writes Data/Paper/state.json and frontend/public/data/paper_trading.json. Falls
back to the cached test predictions only if the frozen model is absent.

Run:  python -m Source.Paper.run            (score with current data)
      python -m Source.Paper.run --refresh  (fetch fresh data, then score)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml

from Source.Backtest.costs import india_cost_breakdown
from Source.Backtest.run import ensemble_signal
from Source.Paper import engine
from Source.Pipeline.data_loader import load_ohlcv
from Source.Pipeline.dataset import build_dataset, build_features, resolve_feature_cols

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Data" / "Processed_Data" / "paper_model"
STATE = ROOT / "Data" / "Paper" / "state.json"
OUT = ROOT / "frontend" / "public" / "data" / "paper_trading.json"


def _rolling_positions(signal: np.ndarray, roll_w: int, q_up: float) -> np.ndarray:
    pos = np.zeros(len(signal), dtype=int)
    for i in range(len(signal)):
        past = signal[max(0, i - roll_w):i]
        if len(past) >= 20:
            pos[i] = int(signal[i] >= np.percentile(past, q_up))
    return pos


def _refresh_data(cfg):
    """Re-download ^NSEI (in the raw yfinance format load_ohlcv parses), macro
    and the universe, so forward days have current features."""
    import yfinance as yf
    print("refreshing ^NSEI / macro / universe ...")
    try:
        d = yf.download(cfg["data"]["ticker"], start=cfg["data"]["start_date"],
                        auto_adjust=False, progress=False)
        if not d.empty:
            (ROOT / cfg["data"]["raw_csv"]).parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(ROOT / cfg["data"]["raw_csv"])       # multi-header, as the loader expects
            print(f"  ^NSEI -> {str(d.index.max().date())}")
    except Exception as e:
        print("  index fetch skipped:", e)
    from Source.Ingestion import fetch_macro, fetch_universe
    try:
        fetch_macro.fetch_macro()
    except Exception as e:
        print("  macro fetch skipped:", e)
    try:
        fetch_universe.fetch_universe()
    except Exception as e:
        print("  universe fetch skipped:", e)


def _frozen_signal_series(cfg):
    """Score every 60-day window with the frozen ensemble -> (dates, closes, signal)."""
    import joblib

    from Source.Models.transformer import build_model
    meta = json.loads((MODEL / "meta.json").read_text(encoding="utf-8"))
    scaler = joblib.load(MODEL / "scaler.pkl")
    feat_cols = meta["feature_cols"]
    lookback = cfg["sequence"]["lookback"]
    mu, sd = np.array(meta["mu"]), np.array(meta["sd"])

    df = build_features(load_ohlcv(ROOT / cfg["data"]["raw_csv"]), cfg)
    # guard: feature set must match the frozen model
    if resolve_feature_cols(cfg) != feat_cols:
        raise SystemExit("feature set changed since the model was frozen - re-run "
                         "scripts/save_paper_model.py")
    feats = df[feat_cols].to_numpy(dtype="float32")
    n_feat = len(feat_cols)

    # inference only: never compiled, so the saved optimizer state is skipped
    # deliberately (no shape-mismatch warning) instead of half-restored.
    models = []
    for i in range(meta["n_seeds"]):
        m, _ = build_model(cfg, num_features=n_feat)
        m.load_weights(str(MODEL / f"seed_{i}.weights.h5"))
        models.append(m)

    idx = np.arange(lookback, len(df))
    W = np.stack([feats[t - lookback:t] for t in idx]).astype("float32")
    W = scaler.transform(W.reshape(-1, n_feat)).reshape(W.shape).astype("float32")
    logit = np.mean([m.predict(W, verbose=0, batch_size=256) for m in models], axis=0)
    sig = ensemble_signal(logit, mu, sd)

    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()[idx]
    closes = df["close"].to_numpy()[idx]
    # rolling threshold seeded with the model's train+val signal history
    full = np.concatenate([np.array(meta["signal_history"]), sig])
    pos = _rolling_positions(full, int(cfg["backtest"].get("rolling_threshold_window", 250)),
                             cfg["backtest"]["quantile_upper"])[len(meta["signal_history"]):]
    # OOS cutoff is FROZEN in the model meta (its last training date), so paper
    # never trades an in-sample day and the curve's start never drifts.
    cutoff_date = meta["oos_cutoff"]
    return dates, closes, sig, pos, cutoff_date, meta


def build_paper(cfg) -> dict:
    holding = cfg["backtest"]["holding_period"]
    per_side = india_cost_breakdown(cfg, "futures")["per_side_bps"]
    dates, closes, sig, pos, cutoff, meta = _frozen_signal_series(cfg)

    st = engine.new_state(100.0)
    prev = None
    for i in range(len(dates)):
        if dates[i] <= cutoff:                        # only paper-trade OOS dates
            prev = float(closes[i])
            continue
        engine.step(st, str(dates[i]), float(closes[i]), prev, int(pos[i]), per_side, holding)
        prev = float(closes[i])
    st["meta"] = {
        "strategy": "timing_rolling (primary)",
        "horizon": cfg["backtest"]["primary_horizon"],
        "cost_roundtrip_bps": round(2 * per_side, 2),
        "seeded_from": f"frozen model trained through {cutoff} (all later days are OOS)",
        "instrument": "Nifty 50 futures (paper)",
        "model_trained_through": meta.get("trained_through"),
    }
    return st


def write(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    summ = engine.summary(state)
    OUT.write_text(json.dumps({
        "as_of": state.get("last_date"),
        "meta": state.get("meta", {}),
        "summary": summ,
        "disclaimer": (
            "PAPER TRADING - simulated, no real money. Runs the backtest's primary "
            "strategy forward on real NSE closes, charged real India futures costs. "
            "The model has NO validated edge (see the results above), so this is an "
            "honest live demonstration, not a profit engine. Not investment advice."
        ),
        "equity_curve": [{"date": r["date"], "strategy": r["equity"],
                          "buy_hold": r["bh_equity"], "position": r["position"]}
                         for r in state["history"]],
        "trades": state["trades"][-40:],
    }, indent=2), encoding="utf-8")
    print(f"paper book: {summ.get('n_days')} OOS days | strat {summ.get('total_return', 0)*100:+.1f}% "
          f"vs BH {summ.get('buy_hold_return', 0)*100:+.1f}% | Sharpe {summ.get('sharpe', 0):+.2f} "
          f"| {summ.get('n_trades')} trades | now {summ.get('current_position')} | as of {state.get('last_date')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    if args.refresh:
        _refresh_data(cfg)
    if (MODEL / "meta.json").exists():
        state = build_paper(cfg)
    else:
        raise SystemExit("frozen model missing - run scripts/save_paper_model.py first")
    write(state)


if __name__ == "__main__":
    main()
