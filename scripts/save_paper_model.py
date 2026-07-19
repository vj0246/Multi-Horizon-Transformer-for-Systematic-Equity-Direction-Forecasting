"""Train and FREEZE the paper-trading model.

Paper trading needs a fixed model that scores genuinely-new days. This trains the
3-seed ensemble on ALL currently available data and saves each seed's weights,
the scaler, and the validation signal statistics + signal history needed for the
rolling-threshold rule. Frozen here; the daily paper step only ever loads and
infers - it never retrains, so every forward day is a true out-of-sample read.

Run on GPU (from repo root, inside the WSL venv):
    python scripts/save_paper_model.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISM", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

from Source.Backtest.run import ensemble_signal  # noqa: E402
from Source.Models.transformer import build_model, compile_model  # noqa: E402
from Source.Pipeline.data_loader import load_ohlcv  # noqa: E402
from Source.Pipeline.dataset import build_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Data" / "Processed_Data" / "paper_model"


def main():
    from Source.device import configure_devices
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    configure_devices(cfg)
    ds = build_dataset(load_ohlcv(ROOT / cfg["data"]["raw_csv"]), cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    n_features = len(ds.feature_cols)
    n_seeds = int(cfg["training"].get("n_seeds", 3))

    # train on train+val (hold nothing out - paper trading is forward-only)
    X = np.concatenate([ds.X_train, ds.X_val]); y = np.concatenate([ds.y_train, ds.y_val])
    vcut = int(0.9 * len(X))
    val_logits_accum = None
    for i in range(n_seeds):
        tf.keras.utils.set_random_seed(cfg["training"]["seed"] + i)
        m, _ = build_model(cfg, num_features=n_features)
        compile_model(m, cfg)
        es = tf.keras.callbacks.EarlyStopping(patience=cfg["training"]["early_stopping_patience"],
                                              restore_best_weights=True)
        m.fit(X[:vcut], y[:vcut], validation_data=(X[vcut:], y[vcut:]),
              epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
              callbacks=[es], verbose=0)
        # full model (architecture + weights + OPTIMIZER state) so training could
        # be resumed; possible because attention pooling is a registered Layer,
        # not a Lambda. Weights are also kept for lightweight loads.
        m.save(str(OUT / f"seed_{i}.keras"))
        m.save_weights(str(OUT / f"seed_{i}.weights.h5"))
        lv = m.predict(X[vcut:], verbose=0)
        val_logits_accum = lv if val_logits_accum is None else val_logits_accum + lv
    val_logits = val_logits_accum / n_seeds

    import joblib
    joblib.dump(ds.scaler, OUT / "scaler.pkl")
    mu, sd = val_logits.mean(0), val_logits.std(0)

    # Per-horizon Platt coefficients, fit ONLY on the held-out tail the seeds did
    # not fit on, so forward probabilities are calibrated without touching any
    # day the paper book will later trade.
    from sklearn.linear_model import LogisticRegression
    platt = []
    for h in range(val_logits.shape[1]):
        col, yh = val_logits[:, h:h + 1], y[vcut:][:, h]
        if len(np.unique(yh)) < 2:                # degenerate horizon -> plain sigmoid
            platt.append({"a": 1.0, "b": 0.0})
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(col, yh)
        platt.append({"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])})
    # signal history over train+val for the rolling threshold seed
    all_logits = None
    for i in range(n_seeds):
        m = tf.keras.models.load_model(str(OUT / f"seed_{i}.keras"))
        p = m.predict(X, verbose=0)
        all_logits = p if all_logits is None else all_logits + p
    sig_hist = ensemble_signal(all_logits / n_seeds, mu, sd)

    # The model trained on train+val, so the last training date is the validation
    # end. Every date STRICTLY AFTER this is out-of-sample - paper trading must
    # start here, and this is frozen so the curve's left edge never drifts.
    oos_cutoff = str(ds.df.iloc[ds.idx_val[-1]]["date"].date())
    (OUT / "meta.json").write_text(json.dumps({
        "n_seeds": n_seeds, "n_features": n_features,
        "feature_cols": ds.feature_cols,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "platt": platt,
        "signal_history": [round(float(x), 5) for x in sig_hist],
        "oos_cutoff": oos_cutoff,
        "trained_through": oos_cutoff,
    }, indent=2), encoding="utf-8")
    print(f"saved frozen paper model ({n_seeds} seeds, {n_features} feat) -> {OUT}")
    print(f"OOS cutoff (last training date) = {oos_cutoff}; paper trades strictly after it")


if __name__ == "__main__":
    main()
