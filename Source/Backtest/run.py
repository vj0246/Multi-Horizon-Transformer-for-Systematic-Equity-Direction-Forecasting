"""End-to-end train + backtest driver. Produces the JSON artifacts the site reads.

Run from repo root:
    python -m Source.Backtest.run

Outputs land in `frontend/public/data/` (configurable). Every number the
frontend displays is produced here from a real trained model - nothing is
hand-authored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from Source.Backtest import metrics as M
from Source.Models.transformer import build_model, compile_model
from Source.Pipeline.data_loader import load_ohlcv
from Source.Pipeline.dataset import build_dataset, build_features, make_windows

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_DETERMINISM_SET = False


def set_seeds(seed: int) -> None:
    global _DETERMINISM_SET
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Make results reproducible across runs so published numbers are stable.
    if not _DETERMINISM_SET:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        _DETERMINISM_SET = True


def fwd_log_return(close: np.ndarray, idx: np.ndarray, h: int) -> np.ndarray:
    """log(close[t+h] / close[t]) for each t in idx (NaN if out of range)."""
    out = np.full(len(idx), np.nan, dtype=float)
    n = len(close)
    for i, t in enumerate(idx):
        if t + h < n:
            out[i] = np.log(close[t + h] / close[t])
    return out


def train_once(ds, cfg, epochs=None, verbose=0):
    set_seeds(cfg["training"]["seed"])
    model, attn_model = build_model(cfg)
    compile_model(model, cfg)
    es = tf.keras.callbacks.EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"], restore_best_weights=True
    )
    hist = model.fit(
        ds.X_train, ds.y_train,
        validation_data=(ds.X_val, ds.y_val),
        epochs=epochs or cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        callbacks=[es], verbose=verbose,
    )
    return model, attn_model, hist


def walk_forward(df_feat, cfg) -> list[dict]:
    """Rolling-origin retrain: expanding train window, evaluate next block out-of-sample."""
    X, y, idx = make_windows(df_feat, cfg)
    close = df_feat["close"].to_numpy()
    n = len(X)
    wf = cfg["backtest"]["walk_forward"]
    n_folds = wf["n_folds"]
    start = int(wf["min_train_frac"] * n)
    fold_size = (n - start) // n_folds
    ph = cfg["backtest"]["primary_horizon"]
    holding = cfg["backtest"]["holding_period"]
    ppy = cfg["backtest"]["periods_per_year"] / holding
    from sklearn.preprocessing import StandardScaler

    folds = []
    for k in range(n_folds):
        tr_end = start + k * fold_size
        te_end = tr_end + fold_size if k < n_folds - 1 else n
        if te_end - tr_end < 20:
            continue
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xte, yte, idxte = X[tr_end:te_end], y[tr_end:te_end], idx[tr_end:te_end]

        nf = Xtr.shape[-1]
        sc = StandardScaler().fit(Xtr.reshape(-1, nf))
        Xtr = sc.transform(Xtr.reshape(-1, nf)).reshape(Xtr.shape).astype("float32")
        Xte = sc.transform(Xte.reshape(-1, nf)).reshape(Xte.shape).astype("float32")

        set_seeds(cfg["training"]["seed"])
        model, _ = build_model(cfg)
        compile_model(model, cfg)
        es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        # small validation tail carved from train for early stopping
        vcut = int(0.9 * len(Xtr))
        model.fit(Xtr[:vcut], ytr[:vcut], validation_data=(Xtr[vcut:], ytr[vcut:]),
                  epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
                  callbacks=[es], verbose=0)

        logits = model.predict(Xte, verbose=0)
        signal = logits[:, ph - 1]
        fwd = fwd_log_return(close, idxte, ph)
        m = ~np.isnan(fwd)
        rep = M.strategy_report(signal[m], fwd[m], idxte[m], cfg, mode="quantile")
        auc = M.per_horizon_classification(
            tf.sigmoid(logits).numpy(), yte
        )[ph - 1]["auc"]
        folds.append({
            "fold": k + 1,
            "train_size": int(tr_end),
            "test_size": int(te_end - tr_end),
            "sharpe_net": rep["sharpe_net"],
            "total_return": rep["total_return"],
            "max_drawdown": rep["max_drawdown"],
            "auc_h20": auc,
        })
    return folds


def main():
    cfg = load_config()
    out_dir = ROOT / cfg["output"]["artifacts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_raw = load_ohlcv(ROOT / cfg["data"]["raw_csv"])
    ds = build_dataset(df_raw, cfg)
    df = ds.df
    close = df["close"].to_numpy()
    ph = cfg["backtest"]["primary_horizon"]
    holding = cfg["backtest"]["holding_period"]
    ppy = cfg["backtest"]["periods_per_year"] / holding

    print(f"Samples: train={len(ds.X_train)} val={len(ds.X_val)} test={len(ds.X_test)}")

    # Heavy artifacts (predictions, attention, walk-forward, training history) are
    # cached so presentation-only tweaks can regenerate JSON without retraining.
    # Set REUSE=1 to load the cache; anything else trains from scratch.
    cache_path = ROOT / cfg["output"]["model_dir"] / "run_cache.npz"
    reuse = os.environ.get("REUSE") == "1" and cache_path.exists()

    if reuse:
        print(f"Reusing cached run: {cache_path}")
        c = np.load(cache_path, allow_pickle=True)
        logits_test = c["logits_test"]
        probs_val = c["probs_val"]
        avg_attention = c["avg_attention"]
        hist_history = c["hist_history"].item()
        wf = list(c["wf"])
    else:
        print("Training main model...")
        model, attn_model, hist = train_once(ds, cfg, verbose=1)
        logits_test = model.predict(ds.X_test, verbose=0)
        probs_val = tf.sigmoid(model.predict(ds.X_val, verbose=0)).numpy()
        attn = attn_model.predict(ds.X_val, verbose=0)      # (N, heads, seq, seq)
        avg_attention = np.mean(attn, axis=(0, 1, 2))       # (seq,)
        hist_history = hist.history
        print("Walk-forward validation...")
        wf = walk_forward(build_features(df_raw, cfg), cfg)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            logits_test=logits_test, probs_val=probs_val,
            avg_attention=avg_attention,
            hist_history=np.array(hist_history, dtype=object),
            wf=np.array(wf, dtype=object),
        )
        print(f"Cached run -> {cache_path}")

    probs_test = tf.sigmoid(logits_test).numpy()

    # ---- per-horizon classification + IC (test) ----
    horizon_cls = M.per_horizon_classification(probs_test, ds.y_test)
    fwd_by_h = {h: fwd_log_return(close, ds.idx_test, h) for h in range(1, ds.horizons + 1)}
    horizon_ic = M.per_horizon_ic(probs_test, fwd_by_h)
    horizons_json = [{**c, **{k: v for k, v in ic.items() if k != "horizon"}}
                     for c, ic in zip(horizon_cls, horizon_ic)]

    # ---- strategies on primary horizon (test) ----
    signal = logits_test[:, ph - 1]
    fwd_primary = fwd_by_h[ph]
    mask = ~np.isnan(fwd_primary)
    sig_m, fwd_m, idx_m = signal[mask], fwd_primary[mask], ds.idx_test[mask]
    strategies = {
        mode: M.strategy_report(sig_m, fwd_m, idx_m, cfg, mode=mode)
        for mode in ("sign", "quantile", "long")
    }
    # Passive buy-and-hold Nifty benchmark over the same non-overlapping periods.
    _, fwd_no_bh = M.non_overlapping(idx_m, sig_m, fwd_m, holding)
    strategies["buy_and_hold"] = M.buy_and_hold_report(fwd_no_bh, cfg)

    # ---- threshold sweep (test, prob-based long-only) ----
    # Thresholds are drawn from the actual probability distribution (40th..95th
    # percentile) so the sweep is always populated regardless of calibration level.
    p = probs_test[:, ph - 1][mask]
    thresholds = np.unique(np.quantile(p, np.linspace(0.40, 0.95, 24)))
    sweep = []
    for th in thresholds:
        s = (p > th).astype(float)
        if s.mean() == 0:
            continue
        sig_no, ret_no = M.non_overlapping(idx_m, s, fwd_m, holding)
        net = M.apply_costs(sig_no, ret_no, M.total_cost_bps(cfg))
        sweep.append({"threshold": round(float(th), 4),
                      "sharpe": M.annualized_sharpe(net, ppy),
                      "trade_freq": float(s.mean())})

    # ---- decile attribution + yearly sharpe ----
    decile = M.decile_attribution(p, fwd_m)
    sig_no, ret_no = M.non_overlapping(idx_m, sig_m, fwd_m, holding)
    pos_no = np.where(sig_no >= np.percentile(sig_m, cfg["backtest"]["quantile_upper"]), 1.0,
                      np.where(sig_no <= np.percentile(sig_m, cfg["backtest"]["quantile_lower"]), -1.0, 0.0))
    net_no = M.apply_costs(pos_no, ret_no, M.total_cost_bps(cfg))
    dates_no = df.iloc[idx_m]["date"].to_numpy()[np.argsort(idx_m)][::holding]
    yearly = M.yearly_sharpe(dates_no, net_no, ppy)

    # ---- attention (days back) ----
    attention_json = [{"days_back": int(len(avg_attention) - i),
                       "weight": float(w)} for i, w in enumerate(avg_attention)]

    # ---- price series for context (downsampled) ----
    price = df[["date", "close"]].copy()
    price["date"] = price["date"].dt.strftime("%Y-%m-%d")
    price = price.iloc[::5]  # every 5th day to keep payload small
    price_json = price.to_dict(orient="records")

    # ---- training history ----
    training_json = {
        "loss": [float(x) for x in hist_history.get("loss", [])],
        "val_loss": [float(x) for x in hist_history.get("val_loss", [])],
        "auc": [float(x) for x in hist_history.get("auc", [])],
        "val_auc": [float(x) for x in hist_history.get("val_auc", [])],
    }

    # ---- cost breakdown + buy-and-hold comparison ----
    from Source.Backtest.costs import india_cost_breakdown
    cost_model = cfg["backtest"].get("cost_model", "flat")
    cost_breakdown = india_cost_breakdown(cfg) if cost_model == "india" else None
    bh = strategies["buy_and_hold"]
    q = strategies["quantile"]

    # ---- summary ----
    aucs = [h["auc"] for h in horizons_json if not np.isnan(h["auc"])]
    ics = [h["ic"] for h in horizons_json if not np.isnan(h["ic"])]
    summary = {
        "ticker": cfg["data"]["ticker"],
        "date_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_end": df["date"].max().strftime("%Y-%m-%d"),
        "n_trading_days": int(len(df)),
        "n_samples": int(len(ds.X_train) + len(ds.X_val) + len(ds.X_test)),
        "split": {"train": int(len(ds.X_train)), "val": int(len(ds.X_val)), "test": int(len(ds.X_test))},
        "n_features": len(ds.feature_cols),
        "horizons": ds.horizons,
        "lookback": cfg["sequence"]["lookback"],
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "mean_ic": float(np.mean(ics)) if ics else None,
        "primary_horizon": ph,
        "strategy_sharpe_net": q["sharpe_net"],
        "strategy_total_return": q["total_return"],
        "strategy_max_drawdown": q["max_drawdown"],
        "cost_model": cost_model,
        "cost_breakdown": cost_breakdown,               # itemized India round-trip (or null)
        "roundtrip_cost_bps": cost_breakdown["roundtrip_bps"] if cost_breakdown else 2 * M.total_cost_bps(cfg),
        "per_side_cost_bps": M.total_cost_bps(cfg),
        "instrument": cfg["backtest"].get("india", {}).get("instrument") if cost_model == "india" else None,
        "buy_hold_sharpe": bh["sharpe_net"],
        "buy_hold_total_return": bh["total_return"],
        "strategy_excess_return": q["total_return"] - bh["total_return"],
        "use_sentiment": cfg["features"].get("use_sentiment", False),
        "model": cfg["model"],
        "walk_forward_mean_sharpe": float(np.mean([f["sharpe_net"] for f in wf])) if wf else None,
    }

    artifacts = {
        "summary.json": summary,
        "horizons.json": horizons_json,
        "strategies.json": strategies,
        "threshold_sweep.json": sweep,
        "decile.json": decile,
        "yearly.json": yearly,
        "attention.json": attention_json,
        "walkforward.json": wf,
        "price.json": price_json,
        "training.json": training_json,
        "features.json": ds.feature_cols,
    }
    for name, obj in artifacts.items():
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, allow_nan=True)
        print(f"  wrote {name}")

    print("\n==== HEADLINE ====")
    print(f"Cost model: {cost_model} ({summary['instrument']}) "
          f"round-trip {summary['roundtrip_cost_bps']:.2f} bps")
    print(f"Mean AUC (test, 20 horizons): {summary['mean_auc']:.4f}")
    print(f"Mean IC  (test): {summary['mean_ic']:.4f}")
    print(f"Quantile L/S Sharpe (net, h={ph}): {summary['strategy_sharpe_net']:.3f}")
    print(f"Strategy total return: {summary['strategy_total_return']*100:.2f}%  "
          f"vs buy-hold {summary['buy_hold_total_return']*100:.2f}%  "
          f"(excess {summary['strategy_excess_return']*100:.2f}%)")
    print(f"Buy-hold Sharpe: {summary['buy_hold_sharpe']:.3f}")
    print(f"Walk-forward mean Sharpe: {summary['walk_forward_mean_sharpe']}")
    print("Artifacts ->", out_dir)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
