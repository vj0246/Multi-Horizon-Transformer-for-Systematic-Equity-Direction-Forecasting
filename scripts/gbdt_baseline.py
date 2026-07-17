"""LightGBM baseline + purged/embargoed evaluation for the index track.

Why this exists
---------------
Two findings drove it:

1. The transformer trains on ~3,095 windows. With a 20-day label horizon and
   daily sampling, consecutive labels overlap ~20x, so the EFFECTIVE sample size
   is ~3095/20 ~= 155 independent observations (Lopez de Prado, Advances in
   Financial ML ch.4). A ~100k-parameter sequence model cannot be fit on that.
2. Gradient-boosted trees consistently outperform deep nets on tabular financial
   data with engineered features - which is exactly the shape of this feature set.

So this fits LightGBM on the per-date feature row (the same 19 stationary +
macro features the transformer sees at the last step of its window) and compares
it honestly against the transformer, per horizon.

Anti-leakage / anti-overfit measures:
- PURGE: training rows whose label window [t, t+h] overlaps the validation or
  test block are dropped (a 20-day label straddles the boundary otherwise).
- EMBARGO: an extra band of rows after each block is dropped.
- Sample uniqueness weights: each row is weighted 1/h to reflect that overlapping
  labels are not independent observations.
- Validation is used ONLY for early stopping; the test block is scored once.

Run:  python scripts/gbdt_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from Source.Pipeline.data_loader import load_ohlcv  # noqa: E402
from Source.Pipeline.dataset import build_features, resolve_feature_cols  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def purged_split(n: int, train_frac: float, val_frac: float, horizon: int, embargo: int):
    """Index arrays with label-overlap purging + an embargo band at each boundary."""
    tr_end = int(train_frac * n)
    va_end = int((train_frac + val_frac) * n)
    gap = horizon + embargo                       # label window + embargo
    train = np.arange(0, max(0, tr_end - gap))    # purge rows whose label leaks into val
    val = np.arange(tr_end, max(tr_end, va_end - gap))
    test = np.arange(va_end, n)
    return train, val, test


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    feat_cols = resolve_feature_cols(cfg)
    horizons = cfg["sequence"]["horizons"]
    embargo = int(cfg["backtest"].get("embargo_days", 5))

    df = build_features(load_ohlcv(ROOT / cfg["data"]["raw_csv"]), cfg)
    X_all = df[feat_cols].to_numpy(dtype="float32")

    print(f"rows={len(df)} features={len(feat_cols)} embargo={embargo}d")
    print(f"{'h':>3}{'VAL AUC':>10}{'TEST AUC':>10}{'n_train':>9}{'eff_n':>8}")

    val_aucs, test_aucs = [], []
    for h in range(1, horizons + 1):
        y_all = df[f"target_{h}"].to_numpy()
        ok = ~np.isnan(y_all)
        # rows must have a full label window ahead of them
        ok[len(df) - h:] = False
        idx = np.where(ok)[0]
        tr, va, te = purged_split(len(idx), cfg["split"]["train_frac"],
                                  cfg["split"]["val_frac"], h, embargo)
        tr, va, te = idx[tr], idx[va], idx[te]
        if len(va) < 50 or len(te) < 50:
            continue

        # uniqueness weight: a label spanning h days overlaps ~h neighbours
        w = np.full(len(tr), 1.0 / h)

        model = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.02, num_leaves=15,
            min_child_samples=60, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, verbosity=-1, random_state=42,
        )
        model.fit(
            X_all[tr], y_all[tr], sample_weight=w,
            eval_set=[(X_all[va], y_all[va])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pv = model.predict_proba(X_all[va])[:, 1]
        pt = model.predict_proba(X_all[te])[:, 1]
        a_v = roc_auc_score(y_all[va], pv)
        a_t = roc_auc_score(y_all[te], pt)
        val_aucs.append(a_v); test_aucs.append(a_t)
        print(f"{h:>3}{a_v:>10.4f}{a_t:>10.4f}{len(tr):>9}{len(tr)/h:>8.0f}")

    print(f"\nMEAN  VAL AUC {np.mean(val_aucs):.4f}   MEAN TEST AUC {np.mean(test_aucs):.4f}")
    print(f"test horizons above 0.5: {sum(1 for a in test_aucs if a > 0.5)}/{len(test_aucs)}")
    print("(transformer reference: mean TEST AUC 0.5033)")


if __name__ == "__main__":
    main()
