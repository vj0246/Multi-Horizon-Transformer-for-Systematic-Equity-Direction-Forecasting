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

# Sector map for sector-relative features (keyed by the CSV stem = ticker without
# the .NS suffix; "&" is written as "_" by fetch_universe). Coarse GICS-style
# buckets - enough to demean momentum within a peer group.
SECTORS = {
    "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy", "BPCL": "Energy", "GAIL": "Energy",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "HDFCBANK": "Bank", "ICICIBANK": "Bank", "SBIN": "Bank", "KOTAKBANK": "Bank", "AXISBANK": "Bank",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "MARUTI": "Auto", "M_M": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "TATAMOTORS": "Auto",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "TATASTEEL": "Metal", "JSWSTEEL": "Metal", "HINDALCO": "Metal",
    "GRASIM": "Cement", "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement",
    "SIEMENS": "CapGoods", "HAVELLS": "CapGoods", "LT": "CapGoods",
    "BHARTIARTL": "Telecom", "TITAN": "Consumer", "ASIANPAINT": "Consumer",
    # expanded universe
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "PFC": "Finance",
    "RECLTD": "Finance", "MUTHOOTFIN": "Finance", "PEL": "Finance",
    "BAJAJ-AUTO": "Auto", "MOTHERSON": "Auto", "BOSCHLTD": "Auto",
    "TVSMOTOR": "Auto", "ASHOKLEY": "Auto", "ESCORTS": "Auto",
    "ADANIPORTS": "Infra", "ADANIENT": "Infra", "CONCOR": "Infra",
    "POWERGRID": "Energy", "COALINDIA": "Energy", "IOC": "Energy", "TATAPOWER": "Energy",
    "VEDL": "Metal", "SAIL": "Metal", "JINDALSTEL": "Metal", "NMDC": "Metal",
    "BANKBARODA": "Bank", "PNB": "Bank", "CANBK": "Bank", "INDUSINDBK": "Bank",
    "BANKINDIA": "Bank", "IDFCFIRSTB": "Bank", "FEDERALBNK": "Bank",
    "DABUR": "FMCG", "MARICO": "FMCG", "GODREJCP": "FMCG", "COLPAL": "FMCG",
    "TATACONSUM": "FMCG", "UNITDSPR": "FMCG",
    "PIDILITIND": "Consumer", "BERGEPAINT": "Consumer", "ZEEL": "Consumer",
    "DLF": "Realty", "SHREECEM": "Cement",
    "LUPIN": "Pharma", "AUROPHARMA": "Pharma", "BIOCON": "Pharma",
    "TORNTPHARM": "Pharma", "APOLLOHOSP": "Pharma",
    "ABB": "CapGoods", "BEL": "CapGoods", "BHEL": "CapGoods",
    "IDEA": "Telecom",
}


def active_feature_cols(cfg: dict) -> list[str]:
    """Base stationary features plus cross-sectional features when enabled."""
    cols = resolve_feature_cols(cfg)
    if cfg["cross_section"].get("use_xs_features", False):
        cols = cols + list(cfg["cross_section"]["xs_features"])
    return cols


def _attach_cross_sectional_features(stocks: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    """Add features that describe each stock RELATIVE TO THE UNIVERSE on each date.

    Per-stock features (momentum, volatility, returns) describe a stock in
    isolation and carry no ranking information. These add: return/momentum
    demeaned by the universe, cross-sectional percentile ranks, and momentum
    demeaned by the stock's sector - exactly the relative signals a cross-
    sectional model needs. Computed only from same-date values (no look-ahead).
    """
    def wide(col: str) -> pd.DataFrame:
        return pd.DataFrame({t: df.set_index("date")[col] for t, df in stocks.items()})

    mom, ret, vol = wide("momentum_10"), wide("daily_ret"), wide("roll_vol_20")
    xs = {
        "xs_ret_vs_uni": ret.sub(ret.mean(axis=1), axis=0),
        "xs_mom_vs_uni": mom.sub(mom.mean(axis=1), axis=0),
        "xs_mom_rank": mom.rank(axis=1, pct=True) - 0.5,
        "xs_ret_rank": ret.rank(axis=1, pct=True) - 0.5,
        "xs_vol_rank": vol.rank(axis=1, pct=True) - 0.5,
    }
    sec = pd.Series({t: SECTORS.get(t, "OTHER") for t in stocks})
    sector_mean = mom.T.groupby(sec).transform("mean").T   # per date, per ticker = its sector's mean momentum
    xs["sector_rel_mom"] = mom - sector_mean

    for name, frame in xs.items():
        for t, df in stocks.items():
            df[name] = df["date"].map(frame[t]).astype("float32").fillna(0.0)
    return stocks


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
    feature_cols: list[str]
    scaler: StandardScaler
    date_train_end: pd.Timestamp
    date_val_end: pd.Timestamp
    # latest 60-day window per stock (forward, out-of-sample) computed from the
    # same load as the panel — no second universe load
    X_latest: np.ndarray
    latest_tickers: list[str]
    latest_asof: list[str]


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
    if cfg["cross_section"].get("objective", "classification") == "regression":
        out = _attach_excess_return_targets(out, cfg)
    elif cfg["cross_section"].get("relative_targets", False):
        out = _attach_relative_targets(out, cfg)
    if cfg["cross_section"].get("use_xs_features", False):
        out = _attach_cross_sectional_features(out, cfg)
    return out


def _attach_excess_return_targets(stocks: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    """Overwrite target_h with the CONTINUOUS cross-sectional excess log-return:
    the stock's h-day forward log return minus the universe median that date.

    This is the regression objective - the model is trained on the magnitude of
    relative out/underperformance, not just its sign.
    """
    horizons = cfg["sequence"]["horizons"]
    fwd_cols: dict[int, dict[str, pd.Series]] = {h: {} for h in range(1, horizons + 1)}
    for tick, df in stocks.items():
        c = df.set_index("date")["close"]
        for h in range(1, horizons + 1):
            fwd_cols[h][tick] = np.log(c.shift(-h) / c)
    for h in range(1, horizons + 1):
        wide = pd.DataFrame(fwd_cols[h])
        excess = wide.sub(wide.median(axis=1), axis=0)      # continuous excess return
        for tick, df in stocks.items():
            df[f"target_{h}"] = df["date"].map(excess[tick]).astype("float32")
    return stocks


def _attach_relative_targets(stocks: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    """Overwrite target_h with RELATIVE labels: did the stock beat the
    cross-sectional median h-day forward return on that date?

    Absolute direction labels cannot discriminate in a trending market (nearly
    every label is 1 in a bull window); relative labels are the canonical
    formulation when the trading decision is a rank.
    """
    horizons = cfg["sequence"]["horizons"]
    # h-day forward log return per stock on its own price series
    fwd_cols: dict[int, dict[str, pd.Series]] = {h: {} for h in range(1, horizons + 1)}
    for tick, df in stocks.items():
        c = df.set_index("date")["close"]
        for h in range(1, horizons + 1):
            fwd_cols[h][tick] = np.log(c.shift(-h) / c)
    for h in range(1, horizons + 1):
        wide = pd.DataFrame(fwd_cols[h])                    # index=date, cols=ticker
        excess = wide.sub(wide.median(axis=1), axis=0)
        tgt = (excess > 0).astype("float32").where(~excess.isna())
        for tick, df in stocks.items():
            df[f"target_{h}"] = df["date"].map(tgt[tick]).astype("float32")
    return stocks


def _windows_for_stock(
    df: pd.DataFrame, cfg: dict, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, end_date, fwd20) tuples for one stock, stepping `stride` days."""
    feat_cols = active_feature_cols(cfg)
    lookback = cfg["sequence"]["lookback"]
    horizons = cfg["sequence"]["horizons"]
    target_cols = [f"target_{h}" for h in range(1, horizons + 1)]

    feats = df[feat_cols].to_numpy(dtype="float32")
    targs = df[target_cols].to_numpy(dtype="float32")
    close = df["close"].to_numpy()
    dates = df["date"].to_numpy()

    X, y, d, fwd = [], [], [], []
    for t in range(lookback, len(df) - horizons, stride):
        if np.isnan(targs[t]).any():
            continue
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

    # Latest windows from the SAME loaded stocks (no second universe load).
    X_latest, latest_tickers, latest_asof = _latest_from_stocks(stocks, cfg, scaler)

    return Panel(
        X_train=tf_(X_train), y_train=y_train,
        X_val=tf_(X_val), y_val=y_val,
        X_test=tf_(X_test), y_test=y_test,
        val_ticker=np.concatenate(va_tick), val_date=np.concatenate(va_date),
        val_fwd20=np.concatenate(va_fwd),
        test_ticker=np.concatenate(te_tick), test_date=np.concatenate(te_date),
        test_fwd20=np.concatenate(te_fwd),
        tickers=list(stocks.keys()),
        feature_cols=active_feature_cols(cfg),
        scaler=scaler,
        date_train_end=date_train_end, date_val_end=date_val_end,
        X_latest=X_latest, latest_tickers=latest_tickers, latest_asof=latest_asof,
    )


def _latest_from_stocks(stocks: dict[str, pd.DataFrame], cfg: dict, scaler
                        ) -> tuple[np.ndarray, list[str], list[str]]:
    """Most recent 60-day scaled window per stock, from an already-loaded dict."""
    feat_cols = active_feature_cols(cfg)
    lookback = cfg["sequence"]["lookback"]
    X, tick, asof = [], [], []
    for t, df in stocks.items():
        feats = df[feat_cols].to_numpy(dtype="float32")
        if len(feats) < lookback:
            continue
        X.append(feats[-lookback:])
        tick.append(t)
        asof.append(pd.Timestamp(df["date"].iloc[-1]).strftime("%Y-%m-%d"))
    X = np.asarray(X, dtype="float32")
    nf = X.shape[-1]
    X = scaler.transform(X.reshape(-1, nf)).reshape(X.shape).astype("float32")
    return X, tick, asof


def latest_windows(cfg: dict, scaler) -> tuple[np.ndarray, list[str], list[str]]:
    """Standalone latest-window extractor (loads the universe itself).

    build_panel already exposes these via panel.X_latest/latest_tickers/
    latest_asof from its own load; use those in the pipeline. This wrapper is
    for ad-hoc/standalone inference and tests.
    """
    return _latest_from_stocks(load_universe(cfg), cfg, scaler)
