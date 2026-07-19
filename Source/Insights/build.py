"""Current forward predictions, stated with their statistical qualification.

Scores the most recent 60-day window with the FROZEN paper model and reports,
per horizon, the calibrated P(up) alongside the only thing that decides whether
that number means anything: that same model's out-of-sample AUC, its overlap-
corrected confidence interval, and whether it survives multiple-testing
correction across all 20 horizons.

Skill is measured on the frozen model's OWN out-of-sample period (everything
strictly after its training cutoff), not borrowed from horizons.json - that
artifact belongs to the backtest model, a different fit, so its error bars do
not describe the predictor that produced these probabilities.

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
from Source.Paper import frozen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "data" / "predictions.json"


def _skill_table(meta, df, idx, logits):
    """Per-horizon skill OF THE FROZEN MODEL, on its own out-of-sample period.

    Deliberately does NOT read horizons.json: that artifact is produced by the
    backtest model (a different fit, trained on train only), so pairing its AUC
    with this model's probabilities would attach error bars to a predictor that
    was never measured. Here the probability and its interval come from the same
    weights on the same days.

    AUC standard errors use the EFFECTIVE sample size (n / horizon): h-day
    labels sampled daily overlap h-fold, so treating them as independent would
    shrink every interval by ~sqrt(h) and manufacture skill.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    cutoff = meta["oos_cutoff"]
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    close = df["close"].to_numpy()
    n_rows = len(df)
    oos = dates[idx] > cutoff                       # strictly after the training cutoff

    rows = []
    for h in range(1, logits.shape[1] + 1):
        # a label exists only where the forward close has actually happened
        valid = oos & (idx + h < n_rows)
        t = idx[valid]
        if valid.sum() < 20:                        # too few realized labels to score
            rows.append({"horizon": h, "auc": float("nan"), "ic": float("nan"),
                         "auc_se": float("nan"), "auc_ci95": [float("nan")] * 2,
                         "n_labelled": int(valid.sum()), "eff_n": valid.sum() / h,
                         "p_value": float("nan")})
            continue
        y = (close[t + h] > close[t]).astype(int)
        score = logits[valid, h - 1]
        fwd = close[t + h] / close[t] - 1.0
        if len(np.unique(y)) < 2:                   # degenerate window, AUC undefined
            auc, se = float("nan"), float("nan")
        else:
            auc = float(roc_auc_score(y, score))
            se = _auc_se(auc, y, overlap=h)
        ic = float(spearmanr(score, fwd).statistic)
        rows.append({
            "horizon": h, "auc": auc, "ic": ic,
            "auc_se": se,
            "auc_ci95": [auc - 1.96 * se, auc + 1.96 * se],
            "n_labelled": int(valid.sum()),
            "eff_n": valid.sum() / h,
            "p_value": auc_pvalue(auc, y, overlap=h) if not np.isnan(auc) else float("nan"),
        })

    mt = multiple_testing([r["p_value"] for r in rows])
    bonf = mt.get("bonferroni_reject", [False] * len(rows))
    bh = mt.get("bh_reject", [False] * len(rows))
    for r, b, k in zip(rows, bonf, bh):
        r["significant_bonferroni"] = bool(b)
        r["significant_bh"] = bool(k)
    return rows, mt


def build(cfg) -> dict:
    meta = frozen.load_meta()
    df = frozen.feature_frame(cfg, meta)
    roll_w = int(cfg["backtest"].get("rolling_threshold_window", 250))

    # Score only what is actually used: the out-of-sample period (for skill) plus
    # a rolling-window run-up (for today's entry threshold). Scoring the full
    # history would discard ~80% of the forward passes every CI run.
    cutoff = meta["oos_cutoff"]
    dates_all = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    first_oos = int(np.argmax(dates_all > cutoff))
    idx, logits = frozen.score(cfg, meta, df, start=max(0, first_oos - roll_w))

    skill, mt = _skill_table(meta, df, idx, logits)
    dates = dates_all[idx]
    closes = df["close"].to_numpy()[idx]

    latest = logits[-1]
    platt = meta.get("platt")
    if platt:
        if len(platt) != len(latest):
            raise SystemExit(f"frozen model has {len(latest)} heads but {len(platt)} Platt "
                             "coefficients - re-run scripts/save_paper_model.py")
        prob = [float(1 / (1 + np.exp(-(p["a"] * z + p["b"]))))
                for p, z in zip(platt, latest)]
    else:                                             # pre-Platt frozen model
        prob = [float(1 / (1 + np.exp(-z))) for z in latest]

    mu, sd = np.array(meta["mu"]), np.array(meta["sd"])
    sig = ensemble_signal(logits, mu, sd)

    # the deployed rule: long only when today's signal clears the trailing quantile.
    # `sig` already starts roll_w days before the OOS period, so the trailing
    # window is fully covered without splicing in the stored signal history.
    q_up = cfg["backtest"]["quantile_upper"]
    past = sig[-roll_w - 1:-1]
    thresh = float(np.percentile(past, q_up))
    today = float(sig[-1])
    pct = float((past < today).mean() * 100)

    if len(skill) != len(prob):                       # never silently truncate horizons
        raise SystemExit(f"skill table has {len(skill)} horizons but {len(prob)} probabilities")
    preds = []
    for row, p in zip(skill, prob):
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
            "mean_auc": float(np.nanmean([p["auc"] for p in preds])),
            "oos_days_scored": int(max(p["n_labelled"] for p in preds)),
            "multiple_testing": {k: v for k, v in mt.items()
                                 if not k.endswith("_reject")},
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
    out = build(cfg)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    v, pos = out["verdict"], out["position"]
    print(f"predictions as of {out['as_of']} | stance {pos['stance']} "
          f"({pos['percentile_of_trailing']:.0f}th pct) | {v['n_actionable']}/{v['n_horizons']} "
          f"horizons actionable | mean AUC {v['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
