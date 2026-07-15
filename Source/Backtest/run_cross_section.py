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

# Determinism flags must be set before TensorFlow is imported.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISM", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from Source.Backtest import metrics as M  # noqa: E402
from Source.Backtest.costs import india_cost_breakdown  # noqa: E402
from Source.Backtest.run import ensemble_signal  # noqa: E402
from Source.Models.ensemble import train_ensemble  # noqa: E402
from Source.Models.transformer import build_model, compile_model  # noqa: E402
from Source.Pipeline.cross_section import build_panel, SECTORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    from Source.device import configure_devices
    configure_devices(cfg)
    out_dir = ROOT / cfg["output"]["artifacts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = cfg["cross_section"]
    holding = cs["holding_period"]
    ppy = cfg["backtest"]["periods_per_year"] / holding

    panel = build_panel(cfg)
    target_mode = "relative" if cs.get("relative_targets", False) else "absolute"
    objective = cs.get("objective", "classification")
    feat_tag = "xs" if cs.get("use_xs_features", False) else "base"
    n_features = len(panel.feature_cols)
    n_seeds = int(cfg["training"].get("n_seeds", 1))

    # Latest 60-day window per stock -> forward, out-of-sample signal (no outcome).
    # Emitted by build_panel from its own load (no second universe load).
    Xl, tickl, asofl = panel.X_latest, panel.latest_tickers, panel.latest_asof

    sp = cfg["split"]
    split_tag = f"tr{sp['train_frac']}_v{sp['val_frac']}"
    cache_path = (ROOT / cfg["output"]["model_dir"]
                  / f"cs_cache_{objective}_{target_mode}_{feat_tag}_s{n_seeds}_{split_tag}.npz")
    if os.environ.get("REUSE") == "1" and cache_path.exists():
        c = np.load(cache_path, allow_pickle=True)
        logits_val, logits_test, logits_latest = c["logits_val"], c["logits_test"], c["logits_latest"]
        hist_history = c["hist_history"].item()
    else:
        seeds = [cfg["training"]["seed"] + i for i in range(n_seeds)]
        avg, _, _, hist_history = train_ensemble(
            build_compile=lambda: (
                compile_model(build_model(cfg, num_features=n_features)[0], cfg, objective=objective),
                None,
            ),
            X_train=panel.X_train, y_train=panel.y_train,
            X_val=panel.X_val, y_val=panel.y_val,
            predict_sets={"val": panel.X_val, "test": panel.X_test, "latest": Xl},
            seeds=seeds,
            epochs=cs.get("epochs", cfg["training"]["epochs"]),
            batch_size=cs.get("batch_size", cfg["training"]["batch_size"]),
            patience=cfg["training"]["early_stopping_patience"],
        )
        logits_val, logits_test, logits_latest = avg["val"], avg["test"], avg["latest"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, logits_val=logits_val, logits_test=logits_test,
                            logits_latest=logits_latest,
                            hist_history=np.array(hist_history, dtype=object))

    # For regression the outputs are excess-return estimates; for classification
    # they are logits. The ranking signal (ensemble, z-scored on validation) and
    # every downstream metric are identical either way.
    ph = cfg["backtest"]["primary_horizon"]
    mu_v, sd_v = logits_val.mean(axis=0), logits_val.std(axis=0)
    sig_test = ensemble_signal(logits_test, mu_v, sd_v)

    df = pd.DataFrame({
        "date": pd.to_datetime(panel.test_date),
        "ticker": panel.test_ticker,
        "signal": sig_test,
        "score_h20": logits_test[:, ph - 1],
        "fwd20": panel.test_fwd20,
    }).dropna(subset=["fwd20"])

    # ---- realized cross-sectional excess return + h20 AUC (works for both heads) ----
    df["excess20"] = df["fwd20"] - df.groupby("date")["fwd20"].transform("median")
    _m = df["excess20"].notna() & (df.groupby("date")["ticker"].transform("count") >= 2)
    try:
        from sklearn.metrics import roc_auc_score
        auc_h20 = float(roc_auc_score((df.loc[_m, "excess20"] > 0).astype(int),
                                      df.loc[_m, "score_h20"]))
    except ValueError:
        auc_h20 = float("nan")

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
    # Pooled rank IC: signal vs realized excess across the whole test panel.
    pooled_ic = float(spearmanr(df.loc[_m, "signal"], df.loc[_m, "excess20"]).correlation)

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
        "objective": objective,
        "pooled_ic": pooled_ic,
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

    # ---- latest per-stock signal across all 20 horizons (forward, no outcome) ----
    if len(tickl) > 0:
        if objective == "classification":
            _, probs_l = M.calibrate_probs(logits_val, panel.y_val, logits_latest)
        else:  # regression outputs are excess estimates; squash monotonically for display
            probs_l = 1.0 / (1.0 + np.exp(-(logits_latest - logits_latest.mean(0)) / (logits_latest.std(0) + 1e-9)))
        ens_l = ensemble_signal(logits_latest, mu_v, sd_v)
        rank_pct = np.argsort(np.argsort(ens_l)) / max(1, len(ens_l) - 1)

        rows = []
        for i in range(len(tickl)):
            rows.append({
                "ticker": tickl[i].replace("_", "&"),
                "sector": SECTORS.get(tickl[i], "Other"),
                "ensemble_score": float(ens_l[i]),
                "rank_pct": float(rank_pct[i]),
                "probs": [round(float(p), 4) for p in probs_l[i]],
            })
        rows.sort(key=lambda r: -r["ensemble_score"])

        names = [r["ticker"] for r in rows]
        k = max(1, round(cs["top_frac"] * len(names)))
        longs, shorts = names[:k], names[-k:]
        _keep = ("sharpe", "sharpe_ci95", "total_return", "max_drawdown")
        risk_profiles = [
            {"key": "conservative", "label": "Conservative",
             "construction": "Equal-weight the entire universe (no selection)",
             **{m: result["ew_benchmark"][m] for m in _keep},
             "long": names, "short": []},
            {"key": "balanced", "label": "Balanced",
             "construction": f"Long the top {k} names by signal (top {int(cs['top_frac']*100)}%)",
             **{m: result["long_only"][m] for m in _keep},
             "long": longs, "short": []},
            {"key": "aggressive", "label": "Aggressive",
             "construction": f"Long top {k} / short bottom {k} (market-neutral spread)",
             **{m: result["spread"][m] for m in _keep},
             "long": longs, "short": shorts},
        ]
        signals = {
            "as_of": max(asofl) if asofl else None,
            "objective": objective,
            "horizons": cfg["sequence"]["horizons"],
            "n_stocks": len(rows),
            "disclaimer": (
                "RESEARCH DEMONSTRATION ONLY - NOT INVESTMENT ADVICE. This model has "
                "no validated predictive edge (see the results above: information "
                "coefficients are within noise and no strategy beats a passive "
                "equal-weight benchmark after costs). These are the model's raw "
                "outputs on the most recent historical window, not a recommendation "
                "to buy or sell any security. The universe carries survivorship bias. "
                "Do not trade on this."
            ),
            "stocks": rows,
            "risk_profiles": risk_profiles,
        }
        with open(out_dir / "stock_signals.json", "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2, allow_nan=True)

    print(f"cross-section done ({result['universe_size']} stocks, {n_seeds}-seed "
          f"ensemble) -> {out_dir}")


if __name__ == "__main__":
    main()
