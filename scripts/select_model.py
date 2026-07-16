"""Hyperparameter selection on the VALIDATION set only.

The index track trains on ~3k windows, so the notebook's inherited architecture
(d_model=64, 2 blocks) is heavily overparameterized. This searches a small grid
and reports mean validation AUC per config.

ANTI-CHEAT: this script never reads X_test / y_test. The winning config is chosen
on validation, written into config.yaml by hand, and only then is the test set
evaluated exactly once by Source/Backtest/run.py.

Run (GPU, from repo root):
    wsl -d Ubuntu bash -c "source scripts/wsl_gpu_env.sh; python scripts/select_model.py"
"""
from __future__ import annotations

import copy
import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISM", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

from Source.Backtest import metrics as M  # noqa: E402
from Source.Models.transformer import build_model, compile_model  # noqa: E402
from Source.Pipeline.data_loader import load_ohlcv  # noqa: E402
from Source.Pipeline.dataset import build_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GRID = [
    # d_model, layers, heads, ff, dropout, lr
    (16, 1, 2, 32, 0.3, 3e-4),
    (32, 1, 2, 64, 0.3, 3e-4),
    (32, 1, 4, 64, 0.1, 1e-4),
    (32, 2, 4, 64, 0.3, 3e-4),
    (64, 2, 4, 128, 0.1, 1e-4),   # current/incumbent
    (64, 2, 4, 128, 0.4, 3e-4),
]


def val_auc(cfg, ds, seed=42) -> float:
    tf.keras.utils.set_random_seed(seed)
    model, _ = build_model(cfg)
    compile_model(model, cfg)
    es = tf.keras.callbacks.EarlyStopping(patience=cfg["training"]["early_stopping_patience"],
                                          restore_best_weights=True)
    model.fit(ds.X_train, ds.y_train, validation_data=(ds.X_val, ds.y_val),
              epochs=cfg["training"]["epochs"], batch_size=cfg["training"]["batch_size"],
              callbacks=[es], verbose=0)
    probs = tf.sigmoid(model.predict(ds.X_val, verbose=0)).numpy()
    aucs = [r["auc"] for r in M.per_horizon_classification(probs, ds.y_val)
            if not np.isnan(r["auc"])]
    return float(np.mean(aucs))


def main():
    from Source.device import configure_devices
    base = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    configure_devices(base)
    ds = build_dataset(load_ohlcv(ROOT / base["data"]["raw_csv"]), base)
    print(f"train={len(ds.X_train)} val={len(ds.X_val)} features={len(ds.feature_cols)}")
    print(f"{'d_model':>8}{'layers':>7}{'heads':>6}{'ff':>5}{'drop':>6}{'lr':>8}{'VAL AUC':>10}")
    results = []
    for d_model, layers, heads, ff, drop, lr in GRID:
        cfg = copy.deepcopy(base)
        cfg["model"].update(d_model=d_model, num_layers=layers, num_heads=heads,
                            ff_dim=ff, dropout=drop)
        cfg["training"]["learning_rate"] = lr
        a = val_auc(cfg, ds)
        results.append(((d_model, layers, heads, ff, drop, lr), a))
        print(f"{d_model:>8}{layers:>7}{heads:>6}{ff:>5}{drop:>6}{lr:>8}{a:>10.4f}")
    best = max(results, key=lambda r: r[1])
    print(f"\nBEST on validation: d_model={best[0][0]} layers={best[0][1]} heads={best[0][2]} "
          f"ff={best[0][3]} dropout={best[0][4]} lr={best[0][5]} -> VAL AUC {best[1]:.4f}")


if __name__ == "__main__":
    main()
