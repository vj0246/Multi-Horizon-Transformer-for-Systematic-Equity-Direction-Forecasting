"""Current forward predictions, stated with their statistical qualification.

Scores the most recent 60-day window with the FROZEN paper model and reports,
per horizon, the calibrated P(up) alongside the only thing that decides whether
that number means anything: the horizon's out-of-sample AUC, its overlap-
corrected confidence interval, and whether it survives multiple-testing
correction across all 20 horizons.

The point is precision about uncertainty. A 54% probability from a horizon whose
AUC confidence interval straddles 0.5 is not a 54% edge - it is noise with a
decimal point, and the artifact says so explicitly per row.

Writes frontend/public/data/predictions.json.

Run:  python -m Source.Insights.build
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from Source.Backtest.run import ensemble_signal
from Source.Evaluation.suite import _auc_se, auc_pvalue, multiple_testing
from Source.Pipeline.data_loader import load_ohlcv
from Source.Pipeline.dataset import build_features, resolve_feature_cols

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Data" / "Processed_Data" / "paper_model"
OUT = ROOT / "frontend" / "public" / "data" / "predictions.json"


def _latest_logits(cfg, meta):
    """Frozen-ensemble logits for the most recent window + the full signal series."""
    import joblib

    from Source.Models.transformer import build_model
    if resolve_feature_cols(cfg) != meta["feature_cols"]:
        raise SystemExit("feature set changed since the model was frozen - re-run "
                         "scripts/save_paper_model.py")
    scaler = joblib.load(MODEL / "scaler.pkl")
    lookback = cfg["sequence"]["lookback"]

    df = build_features(load_ohlcv(ROOT / cfg["data"]["raw_csv"]), cfg)
    feats = df[meta["feature_cols"]].to_numpy(dtype="float32")
    n_feat = len(meta["feature_cols"])

    idx = np.arange(lookback, len(df))
    W = np.stack([feats[t - lookback:t] for t in idx]).astype("float32")
    W = scaler.transform(W.reshape(-1, n_feat)).reshape(W.shape).astype("float32")

    logits = None
    for i in range(meta["n_seeds"]):
        m, _ = build_model(cfg, num_features=n_feat)      # inference only, never compiled
        m.load_weights(str(MODEL / f"seed_{i}.weights.h5"))
        p = m.predict(W, verbose=0, batch_size=256)
        logits = p if logits is None else logits + p
    logits /= meta["n_seeds"]

    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()[idx]
    closes = df["close"].to_numpy()[idx]
    return dates, closes, logits


def _skill_table(cfg):
    """Per-horizon OOS skill from the published backtest, with honest error bars.

    AUC standard errors use the EFFECTIVE sample size (n / horizon): 20-day
    labels built from daily windows overlap 20-fold, so treating them as
    independent would shrink every interval by ~4.5x and manufacture skill.
    """
    horizons = json.loads((ROOT / "frontend" / "public" / "data" / "horizons.json")
                          .read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "frontend" / "public" / "data" / "summary.json")
                         .read_text(encoding="utf-8"))
    n_test = summary["split"]["test"]

    rows = []
    for r in horizons:
        h = int(r["horizon"])
        auc = float(r["auc"])
        eff_n = n_test / h
        # class-balanced synthetic label vector purely to size the SE
        y = np.zeros(n_test)
        y[:int(round(float(r["class_balance_up"]) * n_test))] = 1
        se = _auc_se(auc, y, overlap=h)
        rows.append({
            "horizon": h, "auc": auc, "ic": float(r["ic"]),
            "auc_se": se,
            "auc_ci95": [auc - 1.96 * se, auc + 1.96 * se],
            "eff_n": eff_n,
            # NOT horizons.json's p_value: that is the Spearman IC p-value over
            # raw overlapping samples, which treats ~640 correlated labels as
            # independent and reports significance that is not there. Test the
            # AUC against 0.5 using the overlap-corrected SE instead.
            "p_value": auc_pvalue(auc, y, overlap=h),
            "ic_pvalue_uncorrected": float(r["p_value"]),
        })
    pvals = np.array([r["p_value"] for r in rows], dtype=float)
    mt = multiple_testing(list(pvals))
    # per-horizon reject flags: Bonferroni is a flat threshold; BH rejects the
    # k smallest p-values, where k is the count the suite computed.
    bh_cut = np.sort(pvals)[mt["n_significant_bh"] - 1] if mt["n_significant_bh"] else -np.inf
    for r in rows:
        r["significant_bonferroni"] = bool(r["p_value"] <= mt["bonferroni_threshold"])
        r["significant_bh"] = bool(r["p_value"] <= bh_cut)
    return rows, mt


def build(cfg) -> dict:
    meta = json.loads((MODEL / "meta.json").read_text(encoding="utf-8"))
    dates, closes, logits = _latest_logits(cfg, meta)
    skill, mt = _skill_table(cfg)

    latest = logits[-1]
    platt = meta.get("platt")
    if platt:
        prob = [float(1 / (1 + np.exp(-(p["a"] * z + p["b"]))))
                for p, z in zip(platt, latest)]
    else:                                             # pre-Platt frozen model
        prob = [float(1 / (1 + np.exp(-z))) for z in latest]

    mu, sd = np.array(meta["mu"]), np.array(meta["sd"])
    sig = ensemble_signal(logits, mu, sd)

    # the deployed rule: long only when today's signal clears the trailing quantile
    roll_w = int(cfg["backtest"].get("rolling_threshold_window", 250))
    q_up = cfg["backtest"]["quantile_upper"]
    past = np.concatenate([np.array(meta["signal_history"]), sig])[-roll_w - 1:-1]
    thresh = float(np.percentile(past, q_up))
    today = float(sig[-1])
    pct = float((past < today).mean() * 100)

    preds = []
    for row, p in zip(skill, prob):
        h = row["horizon"]
        lo, hi = row["auc_ci95"]
        actionable = row["significant_bh"]
        preds.append({
            **row,
            "prob_up": p,
            "direction": "up" if p >= 0.5 else "down",
            "edge_pp": (p - 0.5) * 100,
            "actionable": actionable,
            "interpretation": (
                "distinguishable from chance after multiple-testing correction"
                if actionable else
                f"AUC 95% CI [{lo:.3f}, {hi:.3f}] contains 0.50 - this probability "
                f"is not statistically distinguishable from a coin flip"
            ),
        })

    n_sig = sum(p["actionable"] for p in preds)
    return {
        "as_of": str(dates[-1]),
        "last_close": float(closes[-1]),
        "model": {
            "frozen_through": meta["oos_cutoff"],
            "n_seeds": meta["n_seeds"],
            "n_features": meta["n_features"],
            "calibration": "per-horizon Platt, fit on held-out data only"
            if platt else "raw sigmoid (model frozen before Platt was stored)",
        },
        "position": {
            "signal_z": today,
            "threshold_z": thresh,
            "percentile_of_trailing": pct,
            "window_days": roll_w,
            "quantile_rule": q_up,
            "stance": "LONG" if today >= thresh else "FLAT",
            "rationale": (
                f"today's ensemble signal ({today:+.2f}) sits at the {pct:.0f}th "
                f"percentile of the trailing {roll_w} days; the rule goes long above "
                f"the {q_up:.0f}th"
            ),
        },
        "horizons": preds,
        "verdict": {
            "n_actionable": int(n_sig),
            "n_horizons": len(preds),
            "mean_auc": float(np.mean([p["auc"] for p in preds])),
            "headline": (
                f"{n_sig} of {len(preds)} horizons carry statistically distinguishable skill"
                if n_sig else
                "No horizon carries statistically distinguishable skill. The "
                "probabilities below are the model's honest output, not an edge."
            ),
            "note": (
                "Confidence intervals use the effective sample size (test days / "
                "horizon), not the raw day count, because overlapping forward "
                "labels are not independent observations."
            ),
        },
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    if not (MODEL / "meta.json").exists():
        raise SystemExit("frozen model missing - run scripts/save_paper_model.py first")
    out = build(cfg)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    v, pos = out["verdict"], out["position"]
    print(f"predictions as of {out['as_of']} | stance {pos['stance']} "
          f"({pos['percentile_of_trailing']:.0f}th pct) | {v['n_actionable']}/{v['n_horizons']} "
          f"horizons actionable | mean AUC {v['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
