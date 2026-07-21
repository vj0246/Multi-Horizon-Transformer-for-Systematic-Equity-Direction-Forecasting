"""Slow layer - quarterly backbone refit behind a champion/challenger gate.

This is the only layer permitted to touch the ~70,000-parameter network, and it
refits from scratch on all history up to a cutoff rather than incrementally
nudging the incumbent. At ~3 new independent observations per quarter there is
nothing to nudge with; a clean refit on the full record is both simpler and
statistically honest.

Three guards, all of which exist because their absence is how retraining
schedules manufacture edges:

1. EMBARGO. Training stops `embargo_days` before the evaluation block starts.
   With h-day forward labels, a sample at index i peeks at i+h, so training rows
   within h days of the evaluation window would leak the very period used to
   judge the model. Embargo must be >= max horizon; enforced, not assumed.

2. CHAMPION/CHALLENGER. The retrained model does not take over by default. Both
   models are scored on the SAME held-out block, and the challenger is promoted
   only if it beats the incumbent by `min_improvement`. Automatic promotion means
   drifting to whichever model most recently got lucky.

3. TRIAL ACCOUNTING. Every candidate ever trained increments the lineage trial
   count, which is passed to `deflated_sharpe`. Four refits a year for five years
   is twenty chances to look good; a Sharpe that survives one trial but not
   twenty is not a finding.
"""
from __future__ import annotations

import numpy as np

from Source.Adaptive import versioning
from Source.Evaluation.suite import _auc_se, auc_pvalue, deflated_sharpe


def purged_indices(n: int, train_end: int, eval_start: int, eval_end: int,
                   embargo: int) -> tuple[np.ndarray, np.ndarray]:
    """Train indices purged+embargoed against an evaluation block.

    Training rows whose forward label would overlap the evaluation window are
    dropped, and a further `embargo` rows are removed before it.
    """
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    hard_stop = min(train_end, eval_start - embargo)
    train = np.arange(0, max(hard_stop, 0))
    ev = np.arange(max(eval_start, 0), min(eval_end, n))
    if len(train) and len(ev) and train[-1] >= ev[0] - embargo:
        raise AssertionError("purge failed: train overlaps the embargoed evaluation block")
    return train, ev


def evaluate_block(logits: np.ndarray, labels: np.ndarray, horizon: int) -> dict:
    """Score one model on an evaluation block, with overlap-corrected error bars."""
    from sklearn.metrics import roc_auc_score

    aucs, ses, ps = [], [], []
    for h in range(logits.shape[1]):
        y = labels[:, h].astype(int)
        if len(np.unique(y)) < 2:
            continue
        auc = float(roc_auc_score(y, logits[:, h]))
        aucs.append(auc)
        ses.append(_auc_se(auc, y, overlap=h + 1))
        ps.append(auc_pvalue(auc, y, overlap=h + 1))
    if not aucs:
        return {"mean_auc": float("nan"), "n_scored": 0}
    ph = min(horizon, logits.shape[1]) - 1
    return {
        "mean_auc": float(np.mean(aucs)),
        # The 20 horizon AUCs are computed from nested, overlapping windows and
        # are therefore strongly correlated. Averaging them does NOT shrink the
        # error by sqrt(20); assuming perfect correlation (mean of the per-horizon
        # SEs) is the conservative choice and the only defensible one here.
        "mean_auc_se": float(np.nanmean(ses)) if ses else float("nan"),
        "primary_auc": float(aucs[ph]) if ph < len(aucs) else float("nan"),
        "primary_auc_se": float(ses[ph]) if ph < len(ses) else float("nan"),
        "min_p_value": float(np.nanmin(ps)) if ps else float("nan"),
        "n_scored": int(len(labels)),
        "effective_n": float(len(labels) / horizon),
    }


def gate(champion_metrics: dict, challenger_metrics: dict, cfg: dict,
         n_trials: int, net_returns: np.ndarray | None = None) -> dict:
    """Decide promotion. Returns the verdict and the reason, never a bare bool.

    The gate FAILS CLOSED. A fixed improvement margin is not enough on its own:
    on a 126-day evaluation block the effective sample size at h=20 is ~6, where
    the standard error on AUC is ~0.25, so a 0.08 "improvement" is a third of one
    standard error - noise that a fixed 0.01 margin would happily promote. Every
    condition below must pass, and anything that cannot be evaluated counts as a
    failure rather than a pass.
    """
    rt = cfg.get("adaptive", {}).get("retrain", {})
    margin = float(rt.get("min_improvement", 0.01))
    need_dsr = bool(rt.get("promote_requires_dsr", True))
    min_eff_n = float(rt.get("min_effective_n", 30))
    z = float(rt.get("promote_z", 1.96))

    ch = challenger_metrics.get("mean_auc", float("nan"))
    cm = champion_metrics.get("mean_auc", float("nan")) if champion_metrics else float("-inf")
    eff_n = float(challenger_metrics.get("effective_n", 0.0))
    se_ch = float(challenger_metrics.get("mean_auc_se", float("nan")))
    se_cm = float(champion_metrics.get("mean_auc_se", float("nan"))) if champion_metrics else float("nan")
    # conservative: ignore the positive correlation between the two models scored
    # on the same block, which widens the interval rather than narrowing it
    se_diff = float(np.sqrt(np.nansum([se_ch ** 2, se_cm ** 2]))) if np.isfinite(se_ch) else float("nan")
    required = max(margin, z * se_diff) if np.isfinite(se_diff) else float("inf")

    verdict = {
        "challenger_mean_auc": ch,
        "champion_mean_auc": cm if np.isfinite(cm) else None,
        "observed_improvement": float(ch - cm) if np.isfinite(ch) and np.isfinite(cm) else None,
        "se_of_difference": se_diff if np.isfinite(se_diff) else None,
        "required_improvement": required if np.isfinite(required) else None,
        "fixed_margin": margin,
        "z": z,
        "effective_n": eff_n,
        "min_effective_n": min_eff_n,
        "n_trials": int(n_trials),
    }

    # 1. the block must be able to resolve a difference at all
    if eff_n < min_eff_n:
        verdict["promote"] = False
        verdict["reason"] = (
            f"evaluation block carries only {eff_n:.1f} independent observations "
            f"(minimum {min_eff_n:.0f}); no AUC difference measured here is "
            "distinguishable from noise, so no promotion is justified")
        return verdict

    # 2. the improvement must exceed sampling error, not just a fixed margin
    if not (np.isfinite(ch) and ch - cm >= required):
        verdict["promote"] = False
        verdict["reason"] = (
            f"improvement {ch - cm:+.4f} did not clear the required "
            f"{required:.4f} (= max of the {margin:.3f} margin and {z:g} x the "
            f"{se_diff:.4f} standard error of the difference)")
        return verdict

    # 3. deflated Sharpe, if required. Unevaluable -> refuse, never skip.
    if need_dsr:
        if net_returns is None or len(net_returns) <= 3:
            verdict["promote"] = False
            verdict["reason"] = (
                "promote_requires_dsr is set but no return series was supplied, so "
                "the trial-count deflation could not be evaluated; refusing rather "
                "than promoting on an unchecked criterion")
            return verdict
        from scipy import stats as sstats
        sharpe = float(np.mean(net_returns) / (np.std(net_returns) + 1e-12))
        dsr = deflated_sharpe(sharpe, n_obs=len(net_returns), n_trials=n_trials,
                              skew=float(sstats.skew(net_returns)),
                              kurt=float(sstats.kurtosis(net_returns, fisher=False)))
        verdict["deflated_sharpe"] = dsr
        if float(dsr.get("dsr", 0.0)) < 0.95:
            verdict["promote"] = False
            verdict["reason"] = (
                f"AUC improved but the deflated Sharpe is {dsr.get('dsr'):.3f} after "
                f"{n_trials} trials - not distinguishable from the best of "
                f"{n_trials} random attempts")
            return verdict

    verdict["promote"] = True
    verdict["reason"] = (
        f"challenger beat champion by {ch - cm:+.4f}, clearing the {required:.4f} "
        f"required improvement on {eff_n:.1f} independent observations, and survived "
        f"the {n_trials}-trial deflation")
    return verdict


def record(version: str, cutoff: str, verdict: dict, metrics: dict,
           parent: str | None) -> dict:
    """Register the candidate and apply the gate's decision to the registry."""
    versioning.register(version, cutoff, status="challenger", parent=parent,
                        metrics=metrics, notes=verdict.get("reason", ""))
    if verdict.get("promote"):
        return versioning.promote(version, verdict["reason"])
    return versioning.reject(version, verdict["reason"])
