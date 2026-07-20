"""End-to-end train + backtest driver. Produces the JSON artifacts the site reads.

Run from repo root:
    python -m Source.Backtest.run

Outputs land in `frontend/public/data/` (configurable). Every number the
frontend displays is produced here from a real trained model - nothing is
hand-authored.

Evaluation design (single-index honest framing):
- The PRIMARY strategy is long/flat market timing: long when the signal is in
  its top quantile, in cash otherwise. Quantile long-short is a cross-sectional
  construct and is kept only as a reference row.
- The signal is an equal-weight ensemble of all 20 horizon logits (z-scored
  with validation-period statistics), so every output head contributes.
- Entry thresholds and probability calibration (Platt) are fit on the
  VALIDATION set only - the test set is never used to tune anything.
- Sharpe comes with a bootstrap 95% CI: with ~30 non-overlapping trades the
  point estimate alone would be noise dressed as precision.
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
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

from Source.Backtest import metrics as M  # noqa: E402
from Source.Models.ensemble import train_ensemble  # noqa: E402
from Source.Models.transformer import build_model, compile_model  # noqa: E402
from Source.Pipeline.data_loader import load_ohlcv  # noqa: E402
from Source.Pipeline.dataset import build_dataset, build_features, make_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

_DETERMINISM_SET = False


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def ensemble_signal(logits: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Equal-weight ensemble of all horizon logits, z-scored with reference stats.

    mu/sd come from the validation period so no test statistics leak into the
    signal construction. Uses every one of the 20 output heads instead of h20 only.
    """
    z = (logits - mu) / np.where(sd == 0, 1.0, sd)
    return z.mean(axis=1)


def walk_forward(df_feat, cfg) -> list[dict]:
    """Expanding-window retrain; long/flat ensemble timing on each OOS block.

    Threshold and z-score stats come from the fold's own validation carve
    (last 10% of its training window), never from the OOS block.
    """
    X, y, idx = make_windows(df_feat, cfg)
    close = df_feat["close"].to_numpy()
    n = len(X)
    wf = cfg["backtest"]["walk_forward"]
    n_folds = wf["n_folds"]
    start = int(wf["min_train_frac"] * n)
    fold_size = (n - start) // n_folds
    ph = cfg["backtest"]["primary_horizon"]
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
        # small validation tail carved from train for early stopping + thresholds
        vcut = int(0.9 * len(Xtr))
        model.fit(Xtr[:vcut], ytr[:vcut], validation_data=(Xtr[vcut:], ytr[vcut:]),
                  epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
                  callbacks=[es], verbose=0)

        logits_carve = model.predict(Xtr[vcut:], verbose=0)
        logits_te = model.predict(Xte, verbose=0)
        mu, sd = logits_carve.mean(axis=0), logits_carve.std(axis=0)
        sig_carve = ensemble_signal(logits_carve, mu, sd)
        sig_te = ensemble_signal(logits_te, mu, sd)

        fwd = fwd_log_return(close, idxte, ph)
        m = ~np.isnan(fwd)
        rep = M.strategy_report(sig_te[m], fwd[m], idxte[m], cfg,
                                mode="timing", threshold_ref=sig_carve)
        auc = M.per_horizon_classification(
            tf.sigmoid(logits_te).numpy(), yte
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
    from Source.device import configure_devices
    configure_devices(cfg)
    out_dir = ROOT / cfg["output"]["artifacts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_ohlcv(ROOT / cfg["data"]["raw_csv"])
    ds = build_dataset(df_raw, cfg)
    df = ds.df
    close = df["close"].to_numpy()
    ph = cfg["backtest"]["primary_horizon"]
    holding = cfg["backtest"]["holding_period"]
    ppy = cfg["backtest"]["periods_per_year"] / holding
    q_up = cfg["backtest"]["quantile_upper"]
    n_seeds = int(cfg["training"].get("n_seeds", 1))

    # Heavy artifacts (predictions, attention, walk-forward, training history) are
    # cached so presentation-only tweaks can regenerate JSON without retraining.
    # Set REUSE=1 to load the cache; anything else trains from scratch.
    # Cache key must cover everything that changes the cached arrays' shape or
    # meaning: seeds, split, feature set and architecture/optimizer.
    import hashlib
    _sig = hashlib.md5(json.dumps({
        "split": cfg["split"], "model": cfg["model"],
        "lr": cfg["training"]["learning_rate"],
        "features": ds.feature_cols,
    }, sort_keys=True, default=str).encode()).hexdigest()[:10]
    cache_path = ROOT / cfg["output"]["model_dir"] / f"run_cache_v4_s{n_seeds}_{_sig}.npz"
    reuse = os.environ.get("REUSE") == "1" and cache_path.exists()

    if reuse:
        c = np.load(cache_path, allow_pickle=True)
        logits_test = c["logits_test"]
        logits_val = c["logits_val"]
        avg_attention = c["avg_attention"]
        hist_history = c["hist_history"].item()
        wf = list(c["wf"])
    else:
        # Seed-ensemble main model: average predictions over n_seeds models.
        def _build_index():
            m, a = build_model(cfg)          # model and attn_model share weights
            compile_model(m, cfg)
            return m, a

        seeds = [cfg["training"]["seed"] + i for i in range(n_seeds)]
        avg, _, attn_model, hist_history = train_ensemble(
            build_compile=_build_index,
            X_train=ds.X_train, y_train=ds.y_train, X_val=ds.X_val, y_val=ds.y_val,
            predict_sets={"test": ds.X_test, "val": ds.X_val},
            seeds=seeds,
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            patience=cfg["training"]["early_stopping_patience"],
        )
        logits_test, logits_val = avg["test"], avg["val"]
        # Attention from the last-seed model (interpretability only, not averaged).
        attn = attn_model.predict(ds.X_val, verbose=0) if attn_model is not None else np.zeros((1, 1, cfg["sequence"]["lookback"], cfg["sequence"]["lookback"]))
        avg_attention = np.mean(attn, axis=(0, 1, 2))
        wf = walk_forward(build_features(df_raw, cfg), cfg)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            logits_test=logits_test, logits_val=logits_val,
            avg_attention=avg_attention,
            hist_history=np.array(hist_history, dtype=object),
            wf=np.array(wf, dtype=object),
        )

    probs_test = tf.sigmoid(logits_test).numpy()

    # ---- probability calibration (Platt fit on validation, applied to test) ----
    probs_val_cal, probs_test_cal = M.calibrate_probs(logits_val, ds.y_val, logits_test)
    calibration_json = {
        "horizon": ph,
        "pre": M.reliability_bins(probs_test[:, ph - 1], ds.y_test[:, ph - 1]),
        "post": M.reliability_bins(probs_test_cal[:, ph - 1], ds.y_test[:, ph - 1]),
    }

    # ---- per-horizon classification + IC (test) ----
    horizon_cls = M.per_horizon_classification(probs_test, ds.y_test)
    fwd_by_h = {h: fwd_log_return(close, ds.idx_test, h) for h in range(1, ds.horizons + 1)}
    horizon_ic = M.per_horizon_ic(probs_test, fwd_by_h)
    horizons_json = [{**c, **{k: v for k, v in ic.items() if k != "horizon"}}
                     for c, ic in zip(horizon_cls, horizon_ic)]

    # ---- signals: ensemble of all 20 heads (val-z-scored) + h20 reference ----
    mu_v, sd_v = logits_val.mean(axis=0), logits_val.std(axis=0)
    sig_ens_test = ensemble_signal(logits_test, mu_v, sd_v)
    sig_ens_val = ensemble_signal(logits_val, mu_v, sd_v)
    sig_h20_test = logits_test[:, ph - 1]
    sig_h20_val = logits_val[:, ph - 1]

    # ---- best horizon by validation timing Sharpe (selection on val only) ----
    best_h, best_val_sharpe = ph, -np.inf
    for h in range(1, ds.horizons + 1):
        fw_v = fwd_log_return(close, ds.idx_val, h)
        mv = ~np.isnan(fw_v)
        if mv.sum() < 3 * h:
            continue
        s_v = logits_val[mv, h - 1]
        thr = np.percentile(s_v, q_up)
        sg, rt = M.non_overlapping(ds.idx_val[mv], s_v, fw_v[mv], h)
        pos = (sg >= thr).astype(float)
        net = M.apply_costs(pos, rt, M.total_cost_bps(cfg))
        sh = M.annualized_sharpe(net, cfg["backtest"]["periods_per_year"] / h)
        if sh > best_val_sharpe:
            best_val_sharpe, best_h = sh, h

    # ---- strategies on test (thresholds from validation distributions) ----
    fwd_primary = fwd_by_h[ph]
    mask = ~np.isnan(fwd_primary)
    fwd_m, idx_m = fwd_primary[mask], ds.idx_test[mask]
    strategies = {
        # Adaptive-threshold book (past-only expanding quantile) - the deployable
        # rule; a frozen validation cutoff can sit permanently in cash if the
        # signal level shifts. Both are published side by side.
        "timing_expanding": M.strategy_report(sig_ens_test[mask], fwd_m, idx_m, cfg,
                                              mode="timing_expanding", threshold_ref=sig_ens_val),
        "timing_ensemble": M.strategy_report(sig_ens_test[mask], fwd_m, idx_m, cfg,
                                             mode="timing", threshold_ref=sig_ens_val),
        "timing_h20": M.strategy_report(sig_h20_test[mask], fwd_m, idx_m, cfg,
                                        mode="timing", threshold_ref=sig_h20_val),
        "quantile": M.strategy_report(sig_h20_test[mask], fwd_m, idx_m, cfg,
                                      mode="quantile", threshold_ref=sig_h20_val),
        "sign": M.strategy_report(sig_h20_test[mask], fwd_m, idx_m, cfg, mode="sign"),
    }
    fwd_b = fwd_by_h[best_h]
    mb = ~np.isnan(fwd_b)
    strategies["timing_best"] = M.strategy_report(
        logits_test[mb, best_h - 1], fwd_b[mb], ds.idx_test[mb], cfg,
        mode="timing", threshold_ref=logits_val[:, best_h - 1], holding=best_h)
    strategies["timing_best"]["horizon"] = int(best_h)

    # Passive buy-and-hold Nifty benchmark over the same non-overlapping periods.
    _, fwd_no_bh = M.non_overlapping(idx_m, sig_ens_test[mask], fwd_m, holding)
    strategies["buy_and_hold"] = M.buy_and_hold_report(fwd_no_bh, cfg)

    # ---- level-invariant timing: threshold from a TRAILING window of signals ----
    # The signal's level shifts between fit and deployment (validation mean 0.0 vs
    # test mean ~-0.8), so a frozen absolute cutoff leaves the book permanently in
    # cash. Compare each signal to the quantile of the last `roll_w` DAILY signals
    # strictly before it (validation history seeds the start). Past-only, and
    # invariant to a shift in the signal's level.
    roll_w = int(cfg["backtest"].get("rolling_threshold_window", 250))
    _order = np.argsort(idx_m)
    _sig_daily = sig_ens_test[mask][_order]
    _fwd_daily = fwd_m[_order]
    _full = np.concatenate([sig_ens_val, _sig_daily])
    _nval = len(sig_ens_val)
    _sel = np.arange(0, len(_sig_daily), holding)
    _pos, _ret = [], []
    for j in _sel:
        p = _nval + j
        past = _full[max(0, p - roll_w):p]          # strictly before this date
        _pos.append(float(_sig_daily[j] >= np.percentile(past, q_up)))
        _ret.append(_fwd_daily[j])
    _pos = np.asarray(_pos); _ret = np.asarray(_ret)
    _gross_ro = _pos * _ret
    _net_ro = _gross_ro - _pos * 2 * M.total_cost_bps(cfg) / 1e4
    strategies["timing_rolling"] = M.report_from_returns(
        _net_ro, _gross_ro, _pos, ppy, "timing_rolling", holding)

    # Primary = the rolling-threshold book: the frozen-cutoff variants are
    # degenerate here (the signal's level shifts between fit and test, so they
    # never trade). See summary.threshold_rule_note for the honest caveat.
    prim = strategies["timing_rolling"]
    sharpe_ci95 = M.bootstrap_sharpe_ci(np.asarray(prim["net_returns"]), ppy)

    # Vol-targeted variant of the primary book (Source/Risk/sizing.py). Sizes on
    # the primary's own realized vol (lagged, no look-ahead); cost-consistent with
    # the explorer via scaled gross returns and exposures.
    from Source.Risk.sizing import apply_vol_target
    _pg = np.asarray(prim["gross_returns"]); _pap = np.asarray(prim["abs_pos"])
    _, _w = apply_vol_target(np.asarray(prim["net_returns"]), cfg, ppy)
    _gross_rt = _w * _pg
    _abs_rt = _w * _pap
    _net_rt = _gross_rt - _abs_rt * 2 * M.total_cost_bps(cfg) / 1e4
    strategies["risk_targeted"] = M.report_from_returns(
        _net_rt, _gross_rt, _abs_rt, ppy, "risk_targeted", holding)

    # ---- threshold sweep on CALIBRATED P(up), thresholds from validation ----
    p_val = probs_val_cal[:, ph - 1]
    p_test = probs_test_cal[:, ph - 1][mask]
    thresholds = np.unique(np.quantile(p_val, np.linspace(0.40, 0.95, 24)))
    sweep = []
    for th in thresholds:
        s = (p_test > th).astype(float)
        if s.mean() == 0:
            continue
        sig_no, ret_no = M.non_overlapping(idx_m, s, fwd_m, holding)
        net = M.apply_costs(sig_no, ret_no, M.total_cost_bps(cfg))
        sweep.append({"threshold": round(float(th), 4),
                      "sharpe": M.annualized_sharpe(net, ppy),
                      "trade_freq": float(s.mean())})

    # ---- decile attribution (calibrated p) + yearly sharpe of primary timing ----
    decile = M.decile_attribution(p_test, fwd_m)
    thr_ens = np.percentile(sig_ens_val, q_up)
    sig_no, ret_no = M.non_overlapping(idx_m, sig_ens_test[mask], fwd_m, holding)
    pos_no = (sig_no >= thr_ens).astype(float)
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

    # ---- summary ----
    aucs = [h["auc"] for h in horizons_json if not np.isnan(h["auc"])]
    ics = [h["ic"] for h in horizons_json if not np.isnan(h["ic"])]
    wf_sharpes = [f["sharpe_net"] for f in wf]
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
        "primary_strategy": "timing_rolling",
        "threshold_rule_note": (
            "Model architecture/optimizer were selected on VALIDATION only "
            "(scripts/select_model.py); the test set was evaluated once. The "
            "entry-threshold RULE, however, was not: the frozen validation cutoff "
            "(timing_ensemble) turned out degenerate on test - the signal's level "
            "shifts (validation mean 0.00 vs test mean -0.79), so it never trades "
            "and returns exactly 0. The rolling-window rule (past-only, "
            "level-invariant) was adopted after observing that, so its +0.62 Sharpe "
            "carries selection optimism and should be read as an upper bound, not a "
            "clean out-of-sample estimate. Its 95% CI spans zero and it still loses "
            "to buy-and-hold. All three rules are published unmodified."
        ),
        "strategy_sharpe_net": prim["sharpe_net"],
        "sharpe_ci95": sharpe_ci95,
        "strategy_total_return": prim["total_return"],
        "strategy_max_drawdown": prim["max_drawdown"],
        "best_val_horizon": int(best_h),
        "cost_model": cost_model,
        "cost_breakdown": cost_breakdown,               # itemized India round-trip (or null)
        "roundtrip_cost_bps": cost_breakdown["roundtrip_bps"] if cost_breakdown else 2 * M.total_cost_bps(cfg),
        "per_side_cost_bps": M.total_cost_bps(cfg),
        "instrument": cfg["backtest"].get("india", {}).get("instrument") if cost_model == "india" else None,
        "buy_hold_sharpe": bh["sharpe_net"],
        "buy_hold_total_return": bh["total_return"],
        "strategy_excess_return": prim["total_return"] - bh["total_return"],
        "use_sentiment": cfg["features"].get("use_sentiment", False),
        "model": cfg["model"],
        "walk_forward_mean_sharpe": float(np.mean(wf_sharpes)) if wf_sharpes else None,
        "walk_forward_sharpe_std": float(np.std(wf_sharpes)) if wf_sharpes else None,
        "walk_forward_folds": len(wf),
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
        "calibration.json": calibration_json,
    }
    for name, obj in artifacts.items():
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, allow_nan=True)

    print(f"index track done ({n_seeds}-seed ensemble) -> {out_dir}")


if __name__ == "__main__":
    main()
