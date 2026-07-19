"""Shared loading and scoring for the FROZEN paper model.

The paper book and the predictions artifact are published side by side on the
same page, so they must be produced by identical scoring logic - same feature
frame, same scaler application, same ensemble. Keeping one implementation here
is the only way that stays true as either caller changes.

The model is loaded for inference and never compiled: the saved optimizer state
is skipped deliberately rather than half-restored into a fresh optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Source.Pipeline.data_loader import load_ohlcv
from Source.Pipeline.dataset import build_features, resolve_feature_cols

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Data" / "Processed_Data" / "paper_model"


def load_meta() -> dict:
    if not (MODEL / "meta.json").exists():
        raise SystemExit("frozen model missing - run scripts/save_paper_model.py first")
    return json.loads((MODEL / "meta.json").read_text(encoding="utf-8"))


def feature_frame(cfg: dict, meta: dict):
    """Feature frame for the current CSV, guarded against feature drift."""
    if resolve_feature_cols(cfg) != meta["feature_cols"]:
        raise SystemExit("feature set changed since the model was frozen - re-run "
                         "scripts/save_paper_model.py")
    return build_features(load_ohlcv(ROOT / cfg["data"]["raw_csv"]), cfg)


def score(cfg: dict, meta: dict, df, start: int | None = None):
    """Ensemble logits for every window ending at index >= start.

    Returns (idx, logits) where idx[k] is the row the k-th window predicts FROM:
    the window spans df[idx-lookback:idx], matching make_windows, so the label
    for horizon h is close[idx + h] > close[idx].
    """
    import joblib

    from Source.Models.transformer import build_model
    lookback = cfg["sequence"]["lookback"]
    feat_cols = meta["feature_cols"]
    n_feat = len(feat_cols)
    scaler = joblib.load(MODEL / "scaler.pkl")

    feats = df[feat_cols].to_numpy(dtype="float32")
    idx = np.arange(max(lookback, start if start is not None else lookback), len(df))
    W = np.stack([feats[t - lookback:t] for t in idx]).astype("float32")
    W = scaler.transform(W.reshape(-1, n_feat)).reshape(W.shape).astype("float32")

    logits = None
    for i in range(meta["n_seeds"]):
        m, _ = build_model(cfg, num_features=n_feat)   # inference only, never compiled
        m.load_weights(str(MODEL / f"seed_{i}.weights.h5"))
        p = m.predict(W, verbose=0, batch_size=256)
        logits = p if logits is None else logits + p
    return idx, logits / meta["n_seeds"]
