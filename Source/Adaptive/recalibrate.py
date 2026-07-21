"""Decision-layer recalibration - the MID layer (~40 parameters, monthly).

Why this layer exists in this shape: the backbone has ~70,000 parameters, and a
month of daily data carries ~1 independent observation at h=20. Gradient-updating
the network on that cadence fits noise. The DECISION layer is different - two
Platt coefficients per horizon - and can be honestly refit at monthly cadence
PROVIDED it is fit on a trailing window rather than on the month's new points.

  40 parameters on a 250-day trailing window  ->  ~12 independent obs at h=20
  40 parameters on the 21 new days            ->  ~1  independent obs   (refused)

The entry threshold is deliberately NOT refit here. It is already a rolling
past-only percentile recomputed every day with zero fitted parameters, which is
the correct fast-adaptation design at this data rate - and it is what survived
the level shift that broke the fixed validation threshold.

Leakage rule: every fit window ends strictly before the date being predicted.
`recalibrate_at` takes the index it is predicting FOR and only ever looks back.
"""
from __future__ import annotations

import numpy as np


def fit_platt_window(logits: np.ndarray, y: np.ndarray) -> list[dict]:
    """Per-horizon Platt coefficients on one window. Degenerate -> identity."""
    from sklearn.linear_model import LogisticRegression

    out = []
    for h in range(logits.shape[1]):
        col, yh = logits[:, h:h + 1], y[:, h]
        if len(np.unique(yh)) < 2:
            out.append({"a": 1.0, "b": 0.0, "degenerate": True})
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(col, yh)
        out.append({"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0]),
                    "degenerate": False})
    return out


def apply_platt(platt: list[dict], logits_row: np.ndarray) -> list[float]:
    return [float(1.0 / (1.0 + np.exp(-(p["a"] * z + p["b"]))))
            for p, z in zip(platt, logits_row)]


def recalibrate_at(logits: np.ndarray, labels: np.ndarray, t: int, cfg: dict,
                   horizon_max: int) -> dict | None:
    """Refit the decision layer using only data whose labels are known before `t`.

    A label for horizon h at index i is only observed at i+h, so the window must
    end at t - horizon_max, not at t. Skipping that embargo would let the model
    calibrate on the very move it is about to predict.
    """
    rc = cfg.get("adaptive", {}).get("recalibration", {})
    win = int(rc.get("window_days", 250))
    min_win = int(rc.get("min_window", 120))

    end = t - horizon_max                  # last index whose labels have resolved
    start = max(0, end - win)
    n = end - start
    if n < min_win:
        return None

    platt = fit_platt_window(logits[start:end], labels[start:end])
    return {
        "fit_start_index": int(start),
        "fit_end_index": int(end),
        "n_samples": int(n),
        "effective_n": float(n / horizon_max),
        "label_embargo_days": int(horizon_max),
        "platt": platt,
        "n_degenerate": sum(1 for p in platt if p["degenerate"]),
    }


def schedule(n_obs: int, cfg: dict, start_index: int) -> list[int]:
    """Indices at which a monthly recalibration would fire."""
    every = int(cfg.get("adaptive", {}).get("recalibration", {}).get("every_days", 21))
    return list(range(start_index, n_obs, max(every, 1)))
