"""Full evaluation suite: classification, error, financial, and statistical tests.

Covers the standard metric families used in the forecasting literature, plus the
two things most papers omit and that matter most here:

- MULTIPLE-TESTING CORRECTION. With 20 horizons, the best-looking AUC is a
  maximum over 20 correlated draws. At n~640 test samples the standard error of
  AUC is ~0.02, so an uncorrected "best horizon" of 0.57 is ~1 SE of noise away
  from chance. Benjamini-Hochberg and Bonferroni are applied to per-horizon
  p-values so a maximum is never reported as a discovery.
- DEFLATED SHARPE RATIO (Bailey & Lopez de Prado). A Sharpe selected as the best
  of N trials is biased upward. DSR discounts it by the number of trials, the
  track length, and the returns' skew/kurtosis, giving the probability the true
  Sharpe exceeds zero.

Error metrics note: MAPE is undefined for binary direction targets (it divides by
y=0), so it is reported as None for classification and only computed where a
continuous target exists. Reporting a fabricated MAPE would be worse than a null.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------ classification
def classification_metrics(y_true: np.ndarray, prob: np.ndarray, thresh: float = 0.5,
                           overlap: int = 1) -> dict:
    """`overlap` = label horizon in samples. An h-day label sampled daily overlaps
    its ~h neighbours, so the effective sample size is n/h, NOT n. Passing it is
    what keeps auc_se honest."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p > thresh).astype(int)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    base = max(y.mean(), 1 - y.mean())          # always-majority accuracy
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "baseline_accuracy": float(base),
        "accuracy_over_baseline": float(accuracy_score(y, pred) - base),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": auc,
        "auc_se": float(_auc_se(auc, y, overlap)),
        "auc_se_naive_iid": float(_auc_se(auc, y, 1)),
        "overlap_used": int(overlap),
        "effective_n": int(len(y) / max(overlap, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)),
    }


def _auc_se(auc: float, y: np.ndarray, overlap: int = 1) -> float:
    """Hanley-McNeil standard error of AUC, corrected for overlapping labels.

    Hanley-McNeil assumes i.i.d. observations. Direction labels over an h-day
    horizon sampled daily share ~h-1 of their h days with each neighbour, so the
    independent information is ~n/h observations. Feeding the raw n understates
    the SE by ~sqrt(h) (4.5x at h=20) and turns noise into "significance" - the
    single easiest way to fake an edge in this project.
    """
    if np.isnan(auc):
        return float("nan")
    ov = max(int(overlap), 1)
    n1, n0 = int(y.sum()) / ov, (len(y) - int(y.sum())) / ov     # effective counts
    if n1 < 2 or n0 < 2:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return float(np.sqrt(max(var, 0)))


def auc_pvalue(auc: float, y: np.ndarray, overlap: int = 1,
               two_sided: bool = True) -> float:
    """p-value for H0: AUC == 0.5, using the overlap-corrected SE.

    TWO-SIDED by default. The one-sided version (H1: AUC > 0.5) was the original
    default and is structurally blind to anti-skill: a model that ranks
    *backwards* is detecting real structure, and a one-sided test reports it as
    p ~ 1.0 - maximally insignificant - which reads as "no information" when the
    truth is "information, wrong sign".

    That blindness hid a genuine observation: at h=1 the frozen model scores AUC
    0.443, which is 2.6 SE BELOW chance. It is not a usable edge (see below), but
    a test that cannot see it is answering the wrong question.

    Two-sided is also the honest default because the sign of any apparent edge is
    only learned FROM the test set. Deciding after the fact that you would have
    inverted the signal is look-ahead bias, so both tails must count against the
    multiple-testing budget.

    Pass two_sided=False only where a directional prior genuinely exists and the
    opposite result would never be acted on.
    """
    se = _auc_se(auc, y, overlap)
    if np.isnan(auc) or np.isnan(se) or se == 0:
        return float("nan")
    z = (auc - 0.5) / se
    if two_sided:
        return float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(1 - stats.norm.cdf(z))


# ------------------------------------------------------------ error metrics
def error_metrics(y_true: np.ndarray, pred: np.ndarray, continuous: bool = False) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = p - y
    mse = float(np.mean(err ** 2))
    var = float(np.var(y))
    out = {
        "mse": mse,                       # == Brier score for probabilities
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - mse / var) if var > 0 else float("nan"),
    }
    # MAPE divides by the target; undefined for binary (y=0) targets.
    if continuous and np.all(np.abs(y) > 1e-9):
        out["mape"] = float(np.mean(np.abs(err / y)) * 100)
    else:
        out["mape"] = None
        out["mape_note"] = "undefined for binary targets (division by y=0)"
    return out


# ------------------------------------------------------------ financial
def financial_metrics(net_returns: np.ndarray, periods_per_year: float,
                      exposure: np.ndarray | None = None) -> dict:
    r = np.asarray(net_returns, dtype=float)
    if r.size == 0:
        return {}
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(dd.min())
    sd = r.std()
    downside = r[r < 0].std() if (r < 0).any() else 0.0
    ann = np.sqrt(periods_per_year)
    total = float(eq[-1] - 1)
    years = len(r) / periods_per_year
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else float("nan")
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "sharpe": float(r.mean() / sd * ann) if sd > 0 else 0.0,
        "sortino": float(r.mean() / downside * ann) if downside > 0 else float("nan"),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "max_drawdown": mdd,
        "total_return": total,
        "cagr": cagr,
        "volatility_annual": float(sd * ann),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "avg_exposure": float(np.mean(np.abs(exposure))) if exposure is not None else None,
        "n_periods": int(r.size),
    }


# ------------------------------------------------------------ statistical tests
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano test of equal predictive accuracy (squared-error loss).

    H0: the two models forecast equally well. Negative stat => model 1 better.
    Uses a Newey-West correction for the h-step overlap.
    """
    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    n = len(d)
    if n < 10:
        return {"stat": float("nan"), "p_value": float("nan"), "n": n}
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    acc = gamma0
    for lag in range(1, h):                       # Newey-West for overlapping forecasts
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        acc += 2 * (1 - lag / h) * cov
    var_d = acc / n
    if var_d <= 0:
        return {"stat": float("nan"), "p_value": float("nan"), "n": n}
    stat = dbar / np.sqrt(var_d)
    return {"stat": float(stat), "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))), "n": n}


def friedman_test(score_matrix: np.ndarray) -> dict:
    """Friedman test across models (columns) over blocks/horizons (rows)."""
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2 or m.shape[0] < 3:
        return {"stat": float("nan"), "p_value": float("nan")}
    stat, p = stats.friedmanchisquare(*[m[:, j] for j in range(m.shape[1])])
    return {"stat": float(stat), "p_value": float(p), "n_blocks": int(m.shape[0]),
            "n_models": int(m.shape[1])}


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    The expected maximum Sharpe from n_trials independent NOISE strategies is
    strictly positive; DSR asks whether an observed Sharpe beats that bar. Returns
    the probability the true Sharpe > 0 after deflation.
    """
    if n_obs < 3 or n_trials < 1:
        return {"dsr": float("nan"), "expected_max_sharpe_from_noise": float("nan")}
    e, gamma = np.e, 0.5772156649
    # expected maximum of n_trials draws from a standard normal
    z = ((1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
         + gamma * stats.norm.ppf(1 - 1 / (n_trials * e)))
    sr0 = z / np.sqrt(n_obs - 1)                 # noise bar, in per-period Sharpe units
    denom = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2)
    if denom <= 0:
        return {"dsr": float("nan"), "expected_max_sharpe_from_noise": float(sr0)}
    dsr = stats.norm.cdf((sharpe - sr0) * np.sqrt(n_obs - 1) / denom)
    return {"dsr": float(dsr), "expected_max_sharpe_from_noise": float(sr0),
            "n_trials": int(n_trials)}


def multiple_testing(p_values: list[float], alpha: float = 0.05) -> dict:
    """Bonferroni + Benjamini-Hochberg over a family of tests (e.g. 20 horizons).

    Returns per-hypothesis reject flags alongside the counts, aligned with the
    input list (NaN p-values are never rejected). Callers need the flags to
    label individual hypotheses; reconstructing them from the counts by
    re-sorting is subtly wrong when p-values tie at the cutoff.
    """
    raw = np.asarray(p_values, dtype=float)
    p = raw[~np.isnan(raw)]
    n = len(p)
    if n == 0:
        return {}
    order = np.argsort(p)
    bh_thresh = alpha * (np.arange(1, n + 1)) / n
    passed = p[order] <= bh_thresh
    n_bh = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    # BH rejects exactly the n_bh smallest p-values - take them by rank, not by
    # value, so ties at the boundary cannot inflate the rejection set.
    bh_reject = np.zeros(len(raw), dtype=bool)
    if n_bh:
        ranked = np.argsort(np.where(np.isnan(raw), np.inf, raw))
        bh_reject[ranked[:n_bh]] = True
    return {
        "n_tests": n,
        "alpha": alpha,
        "min_p": float(p.min()),
        "bonferroni_threshold": float(alpha / n),
        "n_significant_bonferroni": int((p <= alpha / n).sum()),
        "n_significant_bh": n_bh,
        "n_significant_uncorrected": int((p <= alpha).sum()),
        "bonferroni_reject": [bool(x) for x in (raw <= alpha / n) & ~np.isnan(raw)],
        "bh_reject": [bool(x) for x in bh_reject],
    }
