"""Panel dataset builder for the cross-sectional NSE universe.

Pools per-stock 60-day windows (same 11 stationary features, same 20 direction
targets as the index track) into one training set for a single shared-weight
Transformer. Two leakage guards specific to the panel setting:

1. The train/val/test split is by CALENDAR DATE, not sample count - the same
   market day must never sit in train for one stock and test for another.
2. The StandardScaler is fit on pooled TRAIN windows only.

Training windows are strided (universe.window_stride) to keep CPU runtime sane;
validation and test keep full daily resolution because the cross-sectional
backtest needs a signal for every name on every rebalance date.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Source.Pipeline.dataset import build_features, resolve_feature_cols

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Panel:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # parallel metadata for val/test rows (cross-sectional bookkeeping)
    val_ticker: np.ndarray      # str
    val_date: np.ndarray        # datetime64
    val_fwd20: np.ndarray       # 20-day forward log return (NaN near tail)
    test_ticker: np.ndarray
    test_date: np.ndarray
    test_fwd20: np.ndarray
    tickers: list[str]
    date_train_end: pd.Timestamp
    date_val_end: pd.Timestamp


def load_universe(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load per-ticker cleaned CSVs and attach features/targets. Filters short histories."""
    uni = cfg["universe"]
    raw_dir = ROOT / uni["raw_dir"]
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["date"])
        if len(df) < uni["min_history_days"]:
            print(f"  {path.stem}: only {len(df)} rows - dropped")
            continue
        df = df.sort_values("date").reset_index(drop=True)
        out[path.stem] = build_features(df, cfg)
    return out


def _windows_for_stock(
    df: pd.DataFrame, cfg: dict, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, end_date, fwd20) tuples for one stock, stepping `stride` days."""
    feat_cols = resolve_feature_cols(cfg)
    lookback = cfg["sequence"]["lookback"]
    horizons = cfg["sequence"]["horizons"]
    target_cols = [f"target_{h}" for h in range(1, horizons + 1)]

    feats = df[feat_cols].to_numpy(dtype="float32")
    targs = df[target_cols].to_numpy(dtype="float32")
    close = df["close"].to_numpy()
    dates = df["date"].to_numpy()

    X, y, d, fwd = [], [], [], []
    for t in range(lookback, len(df) - horizons, stride):
        X.append(feats[t - lookback:t])
        y.append(targs[t])
        d.append(dates[t])
        fwd.append(np.log(close[t + 20] / close[t]) if t + 20 < len(close) else np.nan)
    return (np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32"),
            np.asarray(d), np.asarray(fwd, dtype=float))


def build_panel(cfg: dict) -> Panel:
    stocks = load_universe(cfg)
    if len(stocks) < 10:
        raise SystemExit(f"Only {len(stocks)} tickers with sufficient history - "
                         f"run python -m Source.Ingestion.fetch_universe first.")
    stride = cfg["universe"]["window_stride"]
    split = cfg["split"]

    # Date-based split boundaries from the pooled distinct trading dates.
    all_dates = np.array(sorted(set(np.concatenate(
        [df["date"].to_numpy() for df in stocks.values()]))))
    date_train_end = pd.Timestamp(all_dates[int(split["train_frac"] * len(all_dates))])
    date_val_end = pd.Timestamp(
        all_dates[int((split["train_frac"] + split["val_frac"]) * len(all_dates))])

    Xtr, ytr = [], []
    Xva, yva, va_tick, va_date, va_fwd = [], [], [], [], []
    Xte, yte, te_tick, te_date, te_fwd = [], [], [], [], []

    for tick, df in stocks.items():
        # strided train windows
        X, y, d, _ = _windows_for_stock(df, cfg, stride)
        m = d < np.datetime64(date_train_end)
        Xtr.append(X[m]); ytr.append(y[m])
        # full-resolution val/test windows
        X, y, d, fwd = _windows_for_stock(df, cfg, 1)
        mv = (d >= np.datetime64(date_train_end)) & (d < np.datetime64(date_val_end))
        mt = d >= np.datetime64(date_val_end)
        Xva.append(X[mv]); yva.append(y[mv])
        va_tick.append(np.full(mv.sum(), tick)); va_date.append(d[mv]); va_fwd.append(fwd[mv])
        Xte.append(X[mt]); yte.append(y[mt])
        te_tick.append(np.full(mt.sum(), tick)); te_date.append(d[mt]); te_fwd.append(fwd[mt])

    X_train = np.concatenate(Xtr); y_train = np.concatenate(ytr)
    X_val = np.concatenate(Xva); y_val = np.concatenate(yva)
    X_test = np.concatenate(Xte); y_test = np.concatenate(yte)

    scaler = StandardScaler()
    nf = X_train.shape[-1]
    scaler.fit(X_train.reshape(-1, nf))

    def tf_(a):
        return scaler.transform(a.reshape(-1, nf)).reshape(a.shape).astype("float32")

    return Panel(
        X_train=tf_(X_train), y_train=y_train,
        X_val=tf_(X_val), y_val=y_val,
        X_test=tf_(X_test), y_test=y_test,
        val_ticker=np.concatenate(va_tick), val_date=np.concatenate(va_date),
        val_fwd20=np.concatenate(va_fwd),
        test_ticker=np.concatenate(te_tick), test_date=np.concatenate(te_date),
        test_fwd20=np.concatenate(te_fwd),
        tickers=list(stocks.keys()),
        date_train_end=date_train_end, date_val_end=date_val_end,
    )
