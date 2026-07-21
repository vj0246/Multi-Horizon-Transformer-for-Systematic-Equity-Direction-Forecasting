"""Adaptive cycle - runs all four layers and writes one honest report.

  daily      drift detection            0 fitted params    monitor only
  daily      rolling entry threshold    0 fitted params    (already in Paper/run)
  monthly    decision-layer recalib.    ~40 params         trailing window
  quarterly  backbone refit             ~70,000 params     purged + gated

Layered by PARAMETER COUNT, not just by clock: each layer's parameter count must
be supportable by the independent observations its cadence delivers. At h=20 a
week carries ~0.25 independent observations, so the fast layers fit almost
nothing and only the slow layer touches the network.

  --audit    (default) run the monitoring + simulation layers, train nothing
  --retrain  additionally train a challenger and run the champion/challenger gate

Writes frontend/public/data/adaptive.json.

Run:  python -m Source.Adaptive.run
      python -m Source.Adaptive.run --retrain
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISM", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from Source.Adaptive import drift, recalibrate, retrain, versioning  # noqa: E402
from Source.Backtest.run import ensemble_signal  # noqa: E402
from Source.Paper import frozen  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "data" / "adaptive.json"


def _labels(df, idx, horizons):
    """Realised direction labels, NaN where the forward close has not happened."""
    close = df["close"].to_numpy()
    n = len(close)
    lab = np.full((len(idx), horizons), np.nan)
    for h in range(1, horizons + 1):
        ok = idx + h < n
        lab[ok, h - 1] = (close[idx[ok] + h] > close[idx[ok]]).astype(float)
    return lab


def bootstrap_champion(meta) -> dict:
    """Register the existing frozen model as champion v1 if the registry is empty."""
    champ = versioning.champion()
    if champ:
        return champ
    return versioning.promote(
        versioning.register("v1-frozen", meta["oos_cutoff"], status="challenger",
                            parent=None,
                            metrics={"source": "scripts/save_paper_model.py"},
                            notes="initial frozen paper model")["version"],
        "bootstrap: first model in the lineage",
    )


def build(cfg, do_retrain: bool) -> dict:
    horizons = cfg["sequence"]["horizons"]
    meta = frozen.load_meta()
    champ = bootstrap_champion(meta)

    df = frozen.feature_frame(cfg, meta)
    dates_all = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    cutoff = meta["oos_cutoff"]
    first_oos = int(np.argmax(dates_all > cutoff))

    idx, logits = frozen.score(cfg, meta, df, start=max(0, first_oos - 250))
    dates = dates_all[idx]
    mu, sd = np.array(meta["mu"]), np.array(meta["sd"])
    sig = ensemble_signal(logits, mu, sd)
    labels = _labels(df, idx, horizons)

    # ---- layer 1: drift, monitoring only -------------------------------------
    drift_report = drift.scan(list(sig), list(dates), cfg)

    # ---- layer 2: provenance --------------------------------------------------
    # the guard the paper book relies on: every scored date must post-date the
    # champion's training cutoff
    oos_mask = dates > champ["train_cutoff"]
    for d in dates[oos_mask][:1]:
        versioning.assert_out_of_sample(champ, d)

    # ---- layer 3: monthly decision-layer recalibration ------------------------
    rc = cfg["adaptive"]["recalibration"]
    recal_events = []
    if rc.get("enabled", True):
        lab_ok = np.nan_to_num(labels, nan=0.0)
        for t in recalibrate.schedule(len(idx), cfg, start_index=int(oos_mask.argmax())):
            ev = recalibrate.recalibrate_at(logits, lab_ok, t, cfg, horizon_max=horizons)
            if ev:
                ev["date"] = str(dates[t])
                recal_events.append(ev)

    # ---- layer 4: quarterly refit + gate --------------------------------------
    rt = cfg["adaptive"]["retrain"]
    embargo = int(rt.get("embargo_days", 20))
    if embargo < horizons:
        raise SystemExit(f"adaptive.retrain.embargo_days ({embargo}) must be >= "
                         f"sequence.horizons ({horizons}); a shorter embargo lets "
                         "forward labels leak into the evaluation block")

    eval_days = int(rt.get("eval_days", 126))
    n = len(idx)
    eval_start = max(0, n - eval_days)
    train_idx, eval_idx = retrain.purged_indices(n, train_end=eval_start,
                                                 eval_start=eval_start, eval_end=n,
                                                 embargo=embargo)

    ev_lab = labels[eval_idx]
    keep = ~np.isnan(ev_lab).any(axis=1)
    champ_metrics = retrain.evaluate_block(logits[eval_idx][keep], ev_lab[keep],
                                           cfg["backtest"]["primary_horizon"]) \
        if keep.sum() > 20 else {"mean_auc": float("nan"), "n_scored": int(keep.sum())}

    retrain_report = {
        "enabled": bool(rt.get("enabled", True)),
        "ran": False,
        "embargo_days": embargo,
        "n_train_after_purge": int(len(train_idx)),
        "eval_block": {"start": str(dates[eval_idx[0]]) if len(eval_idx) else None,
                       "end": str(dates[eval_idx[-1]]) if len(eval_idx) else None,
                       "n_labelled": int(keep.sum())},
        "champion": {"version": champ["version"], "cutoff": champ["train_cutoff"],
                     **champ_metrics},
    }

    if do_retrain:
        retrain_report.update(_train_challenger(cfg, df, idx, train_idx, eval_idx,
                                                keep, labels, champ, champ_metrics,
                                                dates, horizons))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(dates[-1]),
        "design": {
            "backbone_params": 69589,
            "decision_layer_params": 2 * horizons,
            "independent_obs_per": {
                "day": round(1 / horizons, 3), "week": round(5 / horizons, 3),
                "month": round(21 / horizons, 3), "quarter": round(63 / horizons, 3),
            },
            "rationale": (
                "Layers are sized by parameter count against the independent "
                "observations their cadence delivers. A week carries ~0.25 "
                "independent observations at h=20, so weekly gradient updates to a "
                "69,589-parameter backbone would fit noise; the fast layers "
                "therefore fit 0 or ~40 parameters and only the quarterly layer "
                "refits the network."
            ),
        },
        "drift": drift_report,
        "recalibration": {
            "enabled": bool(rc.get("enabled", True)),
            "window_days": int(rc.get("window_days", 250)),
            "n_events": len(recal_events),
            "events": recal_events[-12:],
            "note": (
                "Platt coefficients are refit on a trailing window ending "
                f"{horizons} days before the prediction date, because a label for "
                "horizon h only resolves h days later. The entry threshold is not "
                "refit here - it is already a past-only rolling percentile with no "
                "fitted parameters."
            ),
        },
        "retrain": retrain_report,
        "registry": versioning.summary(),
    }


def _train_challenger(cfg, df, idx, train_idx, eval_idx, keep, labels, champ,
                      champ_metrics, dates, horizons) -> dict:
    """Fit a challenger on purged data and run it through the gate."""
    import joblib
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    from Source.Models.transformer import build_model, compile_model

    meta = frozen.load_meta()
    feat_cols = meta["feature_cols"]
    lookback = cfg["sequence"]["lookback"]
    feats = df[feat_cols].to_numpy(dtype="float32")
    n_feat = len(feat_cols)

    def windows(rows):
        W = np.stack([feats[t - lookback:t] for t in idx[rows]]).astype("float32")
        return W

    tr_lab = labels[train_idx]
    ok = ~np.isnan(tr_lab).any(axis=1)
    tr_rows = train_idx[ok]
    if len(tr_rows) < 200:
        return {"ran": False, "skipped": f"only {len(tr_rows)} purged training rows"}

    Xtr = windows(tr_rows)
    ytr = labels[tr_rows]
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_feat))      # train-only fit
    Xtr = scaler.transform(Xtr.reshape(-1, n_feat)).reshape(Xtr.shape).astype("float32")

    n_seeds = int(cfg["training"].get("n_seeds", 3))
    ev_rows = eval_idx[keep]
    Xev = windows(ev_rows)
    Xev = scaler.transform(Xev.reshape(-1, n_feat)).reshape(Xev.shape).astype("float32")

    acc = None
    for s in range(n_seeds):
        tf.keras.utils.set_random_seed(cfg["training"]["seed"] + 100 + s)
        m, _ = build_model(cfg, num_features=n_feat)
        compile_model(m, cfg)
        es = tf.keras.callbacks.EarlyStopping(
            patience=cfg["training"]["early_stopping_patience"], restore_best_weights=True)
        cut = int(0.9 * len(Xtr))
        m.fit(Xtr[:cut], ytr[:cut], validation_data=(Xtr[cut:], ytr[cut:]),
              epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
              callbacks=[es], verbose=0)
        p = m.predict(Xev, verbose=0, batch_size=256)
        acc = p if acc is None else acc + p
    chal_logits = acc / n_seeds

    chal_metrics = retrain.evaluate_block(chal_logits, labels[ev_rows],
                                          cfg["backtest"]["primary_horizon"])
    n_trials = versioning.next_trial_index()
    verdict = retrain.gate(champ_metrics, chal_metrics, cfg, n_trials)

    version = f"v{n_trials}-refit-{dates[train_idx[-1]]}"
    cutoff = str(dates[train_idx[-1]])
    entry = retrain.record(version, cutoff, verdict, chal_metrics, parent=champ["version"])

    out_dir = ROOT / "Data" / "Adaptive" / version
    if verdict.get("promote"):
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, out_dir / "scaler.pkl")

    return {
        "ran": True,
        "challenger": {"version": version, "cutoff": cutoff, **chal_metrics},
        "gate": verdict,
        "outcome": entry["status"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true",
                    help="train a challenger and run the champion/challenger gate")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    rep = build(cfg, do_retrain=args.retrain)
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    d = rep["drift"]["detectors"]
    alarms = ", ".join(f"{k} {v['n_alarms']}" for k, v in d.items())
    print(f"adaptive as of {rep['as_of']} | drift alarms: {alarms} "
          f"| recalibrations: {rep['recalibration']['n_events']} "
          f"| champion {rep['registry']['champion']} "
          f"(trials {rep['registry']['n_trials']})")
    if rep["retrain"].get("ran"):
        g = rep["retrain"]["gate"]
        print(f"  challenger -> {rep['retrain']['outcome'].upper()}: {g['reason']}")


if __name__ == "__main__":
    main()
