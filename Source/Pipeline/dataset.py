"""Build model-ready tensors from clean OHLCV data.

Pipeline: features -> 20 binary targets -> 60-day sliding windows ->
temporal split -> StandardScaler (fit on train only). Preserves the exact
logic from the research notebook, no look-ahead leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pathlib import Path

from Source.Features.Returns import add_return_features
from Source.Features.Volatility import add_volatility_features

SENTIMENT_COL = "daily_sentiment"


def resolve_feature_cols(cfg: dict) -> list[str]:
    """Active feature columns. Appends sentiment / macro features when enabled."""
    cols = list(cfg["features"]["feature_cols"])
    if cfg["features"].get("use_sentiment", False):
        cols.append(SENTIMENT_COL)
    if cfg["features"].get("use_macro", False):
        cols += list(cfg["features"].get("macro_features", []))
    return cols


def _merge_sentiment(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Left-join daily FinBERT sentiment on date. Forward-fill gaps, 0.0 = neutral.

    Only invoked when features.use_sentiment is true. Raises if the CSV is missing
    so the failure is loud rather than silently training on all-zero sentiment.
    """
    path = Path(cfg["features"]["sentiment_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"use_sentiment=true but {path} not found. "
            f"Generate it: python -m Source.News.gdelt --days 730"
        )
    sent = pd.read_csv(path)
    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent.rename(columns={sent.columns[-1]: SENTIMENT_COL})[["date", SENTIMENT_COL]]
    df = df.merge(sent, on="date", how="left")
    df[SENTIMENT_COL] = df[SENTIMENT_COL].ffill().fillna(0.0)
    return df


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    scaler: StandardScaler
    df: pd.DataFrame          # feature frame after dropna (row t aligns to idx_*)
    feature_cols: list[str]
    horizons: int


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach all 16 engineered features and the 20 direction targets, then dropna."""
    fc = cfg["features"]
    df = add_return_features(df, fc["return_windows"], fc["momentum_period"], fc["ma_diff_window"])
    df = add_volatility_features(df, fc["vol_windows"], fc["volume_mean_window"])

    if cfg["features"].get("use_sentiment", False):
        df = _merge_sentiment(df, cfg)

    if cfg["features"].get("use_macro", False):
        from pathlib import Path as _P

        from Source.Features.Macro import add_breadth_features, add_macro_features
        df = add_macro_features(df)
        df = add_breadth_features(df, _P(cfg["universe"]["raw_dir"]))

    horizons = cfg["sequence"]["horizons"]
    for h in range(1, horizons + 1):
        df[f"target_{h}"] = (df["close"].shift(-h) > df["close"]).astype("float32")

    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep target NaNs at the tail out of training windows only (handled by loop bound),
    # but drop feature-warmup NaNs at the head.
    df = df.dropna(subset=resolve_feature_cols(cfg)).reset_index(drop=True)
    return df


def make_windows(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding 60-day windows. Stops `horizons` steps early so all targets are valid."""
    feat_cols = resolve_feature_cols(cfg)
    lookback = cfg["sequence"]["lookback"]
    horizons = cfg["sequence"]["horizons"]
    target_cols = [f"target_{h}" for h in range(1, horizons + 1)]

    feats = df[feat_cols].to_numpy(dtype="float32")
    targs = df[target_cols].to_numpy(dtype="float32")

    X_list, y_list, idx_list = [], [], []
    for t in range(lookback, len(df) - horizons):
        X_list.append(feats[t - lookback:t])
        y_list.append(targs[t])
        idx_list.append(t)

    X = np.asarray(X_list, dtype="float32")
    y = np.asarray(y_list, dtype="float32")
    idx = np.asarray(idx_list, dtype="int64")
    return X, y, idx


def temporal_split_and_scale(X, y, idx, cfg: dict) -> tuple:
    """70/15/15 chronological split; StandardScaler fit on train only."""
    train_frac = cfg["split"]["train_frac"]
    val_frac = cfg["split"]["val_frac"]
    train_end = int(train_frac * len(X))
    val_end = int((train_frac + val_frac) * len(X))

    X_train, y_train, idx_train = X[:train_end], y[:train_end], idx[:train_end]
    X_val, y_val, idx_val = X[train_end:val_end], y[train_end:val_end], idx[train_end:val_end]
    X_test, y_test, idx_test = X[val_end:], y[val_end:], idx[val_end:]

    scaler = StandardScaler()
    n_feat = X_train.shape[-1]
    scaler.fit(X_train.reshape(-1, n_feat))

    def tf(arr):
        return scaler.transform(arr.reshape(-1, n_feat)).reshape(arr.shape).astype("float32")

    return (
        tf(X_train), y_train, tf(X_val), y_val, tf(X_test), y_test,
        idx_train, idx_val, idx_test, scaler,
    )


def build_dataset(df_raw: pd.DataFrame, cfg: dict) -> Dataset:
    """End-to-end: raw OHLCV frame -> scaled train/val/test tensors + metadata."""
    df = build_features(df_raw, cfg)
    X, y, idx = make_windows(df, cfg)
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     idx_tr, idx_va, idx_te, scaler) = temporal_split_and_scale(X, y, idx, cfg)
    return Dataset(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te,
        idx_train=idx_tr, idx_val=idx_va, idx_test=idx_te, scaler=scaler,
        df=df, feature_cols=resolve_feature_cols(cfg), horizons=cfg["sequence"]["horizons"],
    )
