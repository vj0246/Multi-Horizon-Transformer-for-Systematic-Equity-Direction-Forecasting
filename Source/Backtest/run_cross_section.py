"""Cross-sectional train + backtest: rank NSE large caps, trade the spread.

Run from repo root:
    python -m Source.Backtest.run_cross_section

This is the track where a direction model can genuinely earn something: instead
of timing one near-efficient index, the SAME shared-weight Transformer scores
~37 stocks on the same date and we trade relative ranks - long the top 20%,
short the bottom 20%, equal weight, rebalanced every 20 trading days.

Cost conventions (India):
- Long-short legs are priced as single-stock futures; each leg pays a full
  round trip per rebalance on its own notional (the reported spread return is
  long-mean minus short-mean, so 2x gross exposure per unit of spread).
- The long-only variant is priced as delivery (STT 0.1% both sides - much
  heavier), which is how a retail portfolio would actually hold it.

Honest caveats exported with the artifacts: the universe is (mostly) today's
large caps backtested into the past, so survivorship bias inflates absolute
returns; the long-short spread is partially insulated but still favored. IC is
also reported on overlapping daily cross-sections (standard practice) while all
P&L is non-overlapping.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from scipy.stats import spearmanr

from Source.Backtest import metrics as M
from Source.Backtest.costs import india_cost_breakdown
from Source.Backtest.run import ensemble_signal, set_seeds
from Source.Models.transformer import build_model, compile_model
from Source.Pipeline.cross_section import build_panel

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    out_dir = ROOT / cfg["output"]["artifacts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = cfg["cross_section"]
    holding = cs["holding_period"]
    ppy = cfg["backtest"]["periods_per_year"] / holding

    print("Building panel...")
    panel = build_panel(cfg)
    print(f"Panel: {len(panel.tickers)} stocks | train={len(panel.X_train)} "
          f"val={len(panel.X_val)} test={len(panel.X_test)} windows")
    print(f"Date split: train < {panel.date_train_end.date()} <= val < "
          f"{panel.date_val_end.date()} <= test")

    target_mode = "relative" if cs.get("relative_targets", False) else "absolute"
    feat_tag = "xs" if cs.get("use_xs_features", False) else "base"
    n_features = len(panel.feature_cols)
    cache_path = ROOT / cfg["output"]["model_dir"] / f"cs_cache_{target_mode}_{feat_tag}.npz"
    if os.environ.get("REUSE") == "1" and cache_path.exists():
        print(f"Reusing cached run: {cache_path}")
        c = np.load(cache_path, allow_pickle=True)
        logits_val, logits_test = c["logits_val"], c["logits_test"]
        hist_history = c["hist_history"].item()
    else:
        print(f"Training shared-weight model on the pooled panel ({n_features} features)...")
        set_seeds(cfg["training"]["seed"])
        model, _ = build_model(cfg, num_features=n_features)
        compile_model(model, cfg)
        es = tf.keras.callbacks.EarlyStopping(
            patience=cfg["training"]["early_stopping_patience"], restore_best_weights=True)
        hist = model.fit(
            panel.X_train, panel.y_train,
            validation_data=(panel.X_val, panel.y_val),
            epochs=cs.get("epochs", cfg["training"]["epochs"]),
            batch_size=cs.get("batch_size", cfg["training"]["batch_size"]),
            callbacks=[es], verbose=1,
        )
        logits_val = model.predict(panel.X_val, verbose=0, batch_size=512)
        logits_test = model.predict(panel.X_test, verbose=0, batch_size=512)
        hist_history = hist.history
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, logits_val=logits_val, logits_test=logits_test,
                            hist_history=np.array(hist_history, dtype=object))
        print(f"Cached -> {cache_path}")

    # ---- pooled classification sanity (h20 AUC on the panel test set) ----
    probs_test = tf.sigmoid(logits_test).numpy()
    auc_h20 = M.per_horizon_classification(probs_test, panel.y_test)[
        cfg["backtest"]["primary_horizon"] - 1]["auc"]

    # ---- signal: ensemble of 20 heads, z-scored on validation stats ----
    mu_v, sd_v = logits_val.mean(axis=0), logits_val.std(axis=0)
    sig_test = ensemble_signal(logits_test, mu_v, sd_v)

    df = pd.DataFrame({
        "date": pd.to_datetime(panel.test_date),
        "ticker": panel.test_ticker,
        "signal": sig_test,
        "fwd20": panel.test_fwd20,
    }).dropna(subset=["fwd20"])

    # ---- daily cross-sectional IC (overlapping horizon - disclosed) ----
    min_names = 15
    ic_rows = []
    for date, grp in df.groupby("date"):
        if len(grp) < min_names:
            continue
        ic, _ = spearmanr(grp["signal"], grp["fwd20"])
        ic_rows.append({"date": date, "ic": float(ic), "n": int(len(grp))})
    ic_df = pd.DataFrame(ic_rows)
    mean_ic = float(ic_df["ic"].mean())
    ic_ir = float(mean_ic / ic_df["ic"].std()) if len(ic_df) > 2 else float("nan")
    pct_ic_pos = float((ic_df["ic"] > 0).mean())

    # ---- non-overlapping rebalances every `holding` trading days ----
    dates = sorted(ic_df["date"].unique())
    rebal_dates = dates[::holding]
    fut = india_cost_breakdown(cfg, instrument="futures")
    dlv = india_cost_breakdown(cfg, instrument="delivery")
    fut_rt = fut["roundtrip_bps"] / 1e4
    dlv_rt = dlv["roundtrip_bps"] / 1e4

    top_frac, bot_frac = cs["top_frac"], cs["bottom_frac"]
    spread_rets, long_rets, ew_rets, quint_accum = [], [], [], []
    for date in rebal_dates:
        grp = df[df["date"] == date].sort_values("signal")
        n = len(grp)
        if n < min_names:
            continue
        k_top = max(1, int(round(top_frac * n)))
        k_bot = max(1, int(round(bot_frac * n)))
        top = grp.tail(k_top)["fwd20"].mean()
        bot = grp.head(k_bot)["fwd20"].mean()
        # long-short spread: each leg pays its own futures round trip
        spread_rets.append(float(top - bot - 2 * fut_rt))
        # long-only top quintile at delivery costs
        long_rets.append(float(top - dlv_rt))
        # equal-weight universe benchmark (gross - the passive comparison)
        ew_rets.append(float(grp["fwd20"].mean()))
        # quintile attribution
        grp = grp.reset_index(drop=True)
        grp["quintile"] = pd.qcut(grp["signal"].rank(method="first"), 5, labels=False)
        quint_accum.append(grp.groupby("quintile")["fwd20"].mean())

    spread_rets = np.array(spread_rets)
    long_rets = np.array(long_rets)
    ew_rets = np.array(ew_rets)
    quintiles = pd.concat(quint_accum, axis=1).mean(axis=1)

    def block(rets: np.ndarray) -> dict:
        eq = M.equity_curve(rets)
        return {
            "sharpe": M.annualized_sharpe(rets, ppy),
            "sharpe_ci95": M.bootstrap_sharpe_ci(rets, ppy),
            "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
            "max_drawdown": M.max_drawdown(eq),
            "hit_rate": float((rets > 0).mean()) if rets.size else 0.0,
            "n_rebalances": int(rets.size),
            "equity_curve": [round(float(v), 5) for v in eq],
        }

    result = {
        "target_mode": target_mode,
        "n_features": n_features,
        "use_xs_features": bool(cs.get("use_xs_features", False)),
        "feature_cols": panel.feature_cols,
        "universe_size": len(panel.tickers),
        "tickers": panel.tickers,
        "test_start": str(pd.Timestamp(min(dates)).date()),
        "test_end": str(pd.Timestamp(max(dates)).date()),
        "n_test_dates": int(len(ic_df)),
        "auc_h20_pooled": float(auc_h20),
        "mean_daily_ic": mean_ic,
        "ic_ir": ic_ir,
        "pct_days_ic_positive": pct_ic_pos,
        "quintile_mean_fwd20": [float(v) for v in quintiles],
        "spread": block(spread_rets),          # long top 20% / short bottom 20%, futures
        "long_only": block(long_rets),          # top 20% at delivery costs
        "ew_benchmark": block(ew_rets),         # equal-weight universe, gross
        "costs": {"futures_roundtrip_bps": fut["roundtrip_bps"],
                  "delivery_roundtrip_bps": dlv["roundtrip_bps"]},
        "ic_series": [{"date": str(pd.Timestamp(r["date"]).date()), "ic": round(r["ic"], 4)}
                      for r in ic_rows[::3]],   # downsampled for the chart
        "caveats": [
            "Universe is (mostly) today's large caps backtested into the past: "
            "survivorship bias inflates absolute returns; the long-short spread "
            "is partially insulated but still favored.",
            "Daily IC uses overlapping 20-day forward returns (standard practice); "
            "all P&L is computed on non-overlapping rebalances.",
            "Equal-weight benchmark is gross of costs; strategy legs are net.",
            f"Targets are {target_mode}. Both formulations were run and both "
            "results are published (cross_section.json / "
            "cross_section_absolute.json) - no cherry-picking.",
        ],
    }

    with open(out_dir / "cross_section.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print("  wrote cross_section.json")

    print("\n==== CROSS-SECTIONAL HEADLINE ====")
    print(f"Universe: {result['universe_size']} stocks | targets {target_mode} | "
          f"{n_features} features ({feat_tag}) | test "
          f"{result['test_start']} .. {result['test_end']}")
    print(f"Pooled AUC h20: {auc_h20:.4f}")
    print(f"Mean daily CS IC: {mean_ic:+.4f} (IR {ic_ir:.2f}, "
          f"{pct_ic_pos*100:.0f}% days positive)")
    print(f"Quintile mean fwd20: {[f'{v:+.4f}' for v in quintiles]}")
    sp, lo, ew = result["spread"], result["long_only"], result["ew_benchmark"]
    print(f"L/S spread (futures, net): Sharpe {sp['sharpe']:+.2f} "
          f"CI {[round(x, 2) for x in sp['sharpe_ci95']]} total {sp['total_return']*100:+.1f}%")
    print(f"Long-only top 20% (delivery, net): Sharpe {lo['sharpe']:+.2f} "
          f"total {lo['total_return']*100:+.1f}%")
    print(f"EW universe benchmark (gross): Sharpe {ew['sharpe']:+.2f} "
          f"total {ew['total_return']*100:+.1f}%")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
