"""LightGBM on the cross-sectional panel - the tool the literature favours for
tabular cross-sectional equity ranking, tested against the panel Transformer.

The panel gives ~86k training rows across 85 names (vs the index track's ~3k), so
unlike the single-index model this is NOT data-starved. GBDT consumes the feature
vector at the last step of each window (the same 29 features the Transformer sees
there), which is the natural tabular form.

Evaluated exactly like the Transformer track so the comparison is like-for-like:
mean daily cross-sectional IC (Spearman of prediction vs realized 20d excess
return, per date), pooled rank IC, pooled AUC, and quintile attribution.
Validation drives early stopping only; the test block is scored once.

Run:  python scripts/gbdt_cross_section.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from Source.Pipeline.cross_section import build_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    ph = cfg["backtest"]["primary_horizon"]
    panel = build_panel(cfg)

    # tabular form: the feature vector at the last step of each window (t-1)
    Xtr, Xva, Xte = panel.X_train[:, -1, :], panel.X_val[:, -1, :], panel.X_test[:, -1, :]
    ytr, yva, yte = panel.y_train[:, ph - 1], panel.y_val[:, ph - 1], panel.y_test[:, ph - 1]
    print(f"panel {len(panel.tickers)} stocks | train {Xtr.shape} val {Xva.shape} test {Xte.shape}")

    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=31,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=10.0,
        verbosity=-1, random_state=42,
    )
    model.fit(Xtr, ytr, sample_weight=np.full(len(ytr), 1.0 / ph),
              eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])

    pv = model.predict_proba(Xva)[:, 1]
    pt = model.predict_proba(Xte)[:, 1]
    print(f"\npooled AUC h{ph}:  VAL {roc_auc_score(yva, pv):.4f}   TEST {roc_auc_score(yte, pt):.4f}")
    print("(panel Transformer reference: pooled TEST AUC 0.4876)")

    # cross-sectional IC on test, exactly like the Transformer track
    df = pd.DataFrame({
        "date": pd.to_datetime(panel.test_date),
        "signal": pt,
        "fwd20": panel.test_fwd20,
    }).dropna(subset=["fwd20"])
    df["excess20"] = df["fwd20"] - df.groupby("date")["fwd20"].transform("median")

    ics = [spearmanr(g["signal"], g["fwd20"]).correlation
           for _, g in df.groupby("date") if len(g) >= 15]
    ics = [x for x in ics if not np.isnan(x)]
    mean_ic = float(np.mean(ics))
    print(f"mean daily CS IC: {mean_ic:+.4f}  (IR {mean_ic/np.std(ics):+.3f}, "
          f"{100*np.mean([i>0 for i in ics]):.0f}% days positive)")
    print("(panel Transformer reference: mean daily IC -0.0290)")
    pooled = spearmanr(df["signal"], df["excess20"]).correlation
    print(f"pooled rank IC:   {pooled:+.4f}   (Transformer reference: -0.0260)")

    # quintile attribution
    q = df.groupby("date")["signal"].transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                                                labels=False, duplicates="drop"))
    print("\nquintile mean fwd20 (Q1..Q5):",
          [round(v * 100, 2) for v in df.groupby(q)["fwd20"].mean().tolist()])

    imp = sorted(zip(panel.feature_cols, model.feature_importances_),
                 key=lambda x: -x[1])[:10]
    print("\ntop features:", [f"{n}({v})" for n, v in imp])


if __name__ == "__main__":
    main()
