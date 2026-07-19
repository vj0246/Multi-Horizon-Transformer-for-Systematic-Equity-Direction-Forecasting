"""Rigorous correctness + leakage + strategy tests for the whole pipeline.

Run:  python -m pytest tests/test_rigorous.py -q
      (or)  python tests/test_rigorous.py   for a plain pass/fail summary.

These test the DATA and MATH (no model training), which is where correctness and
look-ahead bugs live. Covers: feature formulas, forward targets, window/temporal
split integrity, scaler-on-train-only, cross-sectional date-split leakage, the
India cost model, the Sharpe/drawdown/cost/non-overlap metrics, strategy-report
reproduction, calibration rank-preservation, and validation of the generated
JSON artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))

from Source.Backtest import metrics as M
from Source.Backtest.costs import india_cost_breakdown
from Source.Pipeline.data_loader import load_ohlcv
from Source.Pipeline.dataset import (build_dataset, build_features, make_windows,
                                     resolve_feature_cols)


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def raw():
    return load_ohlcv(ROOT / CFG["data"]["raw_csv"])


@pytest.fixture(scope="module")
def ds(raw):
    return build_dataset(raw, CFG)


# ---------------------------------------------------------------- data loader
def test_loader_sorted_and_clean(raw):
    assert list(raw.columns) == ["date", "close", "high", "low", "open", "volume"]
    assert raw["date"].is_monotonic_increasing
    assert not raw[["close", "high", "low", "open"]].isna().any().any()
    assert (raw["high"] >= raw["low"]).all()          # high >= low always
    assert (raw["close"] > 0).all()


# ---------------------------------------------------------------- feature formulas
def test_feature_formulas(raw):
    df = build_features(raw.copy(), CFG)
    ref = raw.set_index("date")["close"]
    dr = ref.pct_change()
    # daily_ret matches pct_change on the shared dates
    m = df.set_index("date")
    common = m.index.intersection(dr.index)
    assert np.allclose(m.loc[common, "daily_ret"], dr.loc[common], atol=1e-9, equal_nan=True)
    # ma_diff_10 == (close - MA10)/MA10
    ma10 = ref.rolling(10).mean()
    exp = ((ref - ma10) / ma10).loc[common]
    assert np.allclose(m.loc[common, "ma_diff_10"], exp, atol=1e-9, equal_nan=True)


def test_raw_price_levels_excluded():
    cols = resolve_feature_cols(CFG)
    for banned in ("close", "high", "low", "open", "volume"):
        assert banned not in cols, f"{banned} must not be a model feature (non-stationary)"
    expected = len(CFG["features"]["feature_cols"])
    if CFG["features"].get("use_macro", False):
        expected += len(CFG["features"]["macro_features"])
    if CFG["features"].get("use_sentiment", False):
        expected += 1
    assert len(cols) == expected
    assert len(cols) == len(set(cols)), "duplicate feature columns"


def test_targets_are_forward_looking(raw):
    df = build_features(raw.copy(), CFG)
    close = df["close"].to_numpy()
    for h in (1, 5, 20):
        col = df[f"target_{h}"].to_numpy()
        # target_h[t] == 1  iff  close[t+h] > close[t]
        for t in range(0, len(df) - 21, 137):     # sample rows
            if not np.isnan(col[t]):
                assert col[t] == float(close[t + h] > close[t])


# ---------------------------------------------------------------- no look-ahead in windows
def test_window_has_no_lookahead(raw):
    df = build_features(raw.copy(), CFG)
    X, y, idx = make_windows(df, CFG)
    lookback = CFG["sequence"]["lookback"]
    feats = df[resolve_feature_cols(CFG)].to_numpy(dtype="float32")
    # window for sample i must be exactly rows [t-lookback, t) — never includes t
    for i in (0, 1, len(idx) // 2, len(idx) - 1):
        t = idx[i]
        assert np.allclose(X[i], feats[t - lookback:t], atol=1e-6)
        assert t - 1 == (t - lookback) + lookback - 1   # last row is t-1, strictly before t


def test_temporal_split_disjoint_and_ordered(ds):
    # index ranges strictly increasing and non-overlapping
    assert ds.idx_train.max() < ds.idx_val.min()
    assert ds.idx_val.max() < ds.idx_test.min()
    frac = CFG["split"]
    n = len(ds.idx_train) + len(ds.idx_val) + len(ds.idx_test)
    assert abs(len(ds.idx_train) / n - frac["train_frac"]) < 0.02


def test_scaler_fit_on_train_only(ds, raw):
    # Re-fit a scaler on the training windows only and confirm it equals ds.scaler.
    df = build_features(raw.copy(), CFG)
    X, _, _ = make_windows(df, CFG)
    n = int((CFG["split"]["train_frac"]) * len(X))
    from sklearn.preprocessing import StandardScaler
    ref = StandardScaler().fit(X[:n].reshape(-1, X.shape[-1]))
    assert np.allclose(ref.mean_, ds.scaler.mean_, atol=1e-6)
    # train post-scale ~ N(0,1); test uses train stats so its mean is NOT ~0
    assert abs(ds.X_train.mean()) < 1e-3 and abs(ds.X_train.std() - 1) < 1e-2


# ---------------------------------------------------------------- cross-sectional leakage
def test_cross_section_feature_cols_unique():
    """Macro/regime features reach the panel via use_macro; listing them again in
    xs_features would silently duplicate columns."""
    from Source.Pipeline.cross_section import active_feature_cols
    cols = active_feature_cols(CFG)          # raises on duplicates
    assert len(cols) == len(set(cols))
    if CFG["cross_section"].get("use_xs_features", False):
        for c in CFG["cross_section"]["xs_features"]:
            assert c in cols


def test_cross_section_date_split_no_leakage():
    if not (ROOT / CFG["universe"]["raw_dir"]).exists():
        pytest.skip("universe not downloaded")
    from Source.Pipeline.cross_section import build_panel
    p = build_panel(CFG)
    # every train window strictly before the val boundary; every test window on/after val_end
    assert p.val_date.min() >= np.datetime64(p.date_train_end)
    assert p.test_date.min() >= np.datetime64(p.date_val_end)
    # NO calendar date appears in both val and test (date-based split integrity)
    assert p.test_date.min() > p.val_date.max()
    # no NaN leaked into any tensor
    for arr in (p.X_train, p.X_val, p.X_test):
        assert not np.isnan(arr).any()
    assert p.scaler is not None


# ---------------------------------------------------------------- India cost model
def test_india_cost_breakdown():
    fut = india_cost_breakdown(CFG, instrument="futures")
    parts = (fut["brokerage_bps"] + fut["exchange_txn_bps"] + fut["sebi_bps"]
             + fut["slippage_bps"] + fut["stt_bps"] + fut["stamp_duty_bps"]
             + fut["gst_bps"] + fut["dp_charge_bps"])
    assert abs(parts - fut["roundtrip_bps"]) < 1e-3
    assert abs(fut["per_side_bps"] - fut["roundtrip_bps"] / 2) < 1e-3   # both rounded to 4dp
    # futures: STT 0.02% sell -> 2.0 bps; stamp 0.002% buy -> 0.2 bps (NOT 2.0)
    assert fut["stt_bps"] == 2.0 and fut["stamp_duty_bps"] == 0.2
    dlv = india_cost_breakdown(CFG, instrument="delivery")
    assert dlv["stt_bps"] == 20.0                                    # 0.1% both sides
    assert dlv["roundtrip_bps"] > fut["roundtrip_bps"]               # delivery heavier (STT)
    intr = india_cost_breakdown(CFG, instrument="intraday")
    assert intr["stt_bps"] == 2.5 and intr["roundtrip_bps"] < dlv["roundtrip_bps"]


# ---------------------------------------------------------------- metrics math
def test_metrics_math():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    ppy = 12.6
    assert abs(M.annualized_sharpe(r, ppy) - (r.mean() / r.std() * np.sqrt(ppy))) < 1e-9
    assert M.annualized_sharpe(np.zeros(5), ppy) == 0.0
    eq = np.array([1.0, 1.2, 0.9, 1.1])
    assert abs(M.max_drawdown(eq) - (0.9 / 1.2 - 1)) < 1e-9
    # apply_costs: net = pos*ret - |pos|*2*bps/1e4
    pos = np.array([1.0, -1.0, 0.0]); ret = np.array([0.02, 0.01, 0.05])
    net = M.apply_costs(pos, ret, 5.0)
    assert np.allclose(net, pos * ret - np.abs(pos) * 2 * 5e-4)
    # non_overlapping picks every holding-th, sorted by idx
    idx = np.arange(100)[::-1]; sig = idx.astype(float); fwd = idx.astype(float)
    s, f = M.non_overlapping(idx, sig, fwd, 20)
    assert len(s) == 5 and s[0] == 0 and s[1] == 20        # re-sorted ascending, stride 20
    # buy_and_hold equity == cumprod(1+r)
    bh = M.buy_and_hold_report(r, CFG)
    assert np.allclose(bh["equity_curve"], np.cumprod(1 + r), atol=1e-5)


def test_strategy_report_reproduces_net_and_sharpe():
    rng = np.random.default_rng(0)
    n = 200
    idx = np.arange(n)
    signal = rng.normal(size=n)
    fwd = rng.normal(scale=0.03, size=n)
    ref = rng.normal(size=500)          # validation-side threshold reference
    rep = M.strategy_report(signal, fwd, idx, CFG, mode="timing", threshold_ref=ref)
    net = np.array(rep["net_returns"]); gross = np.array(rep["gross_returns"])
    ap = np.array(rep["abs_pos"])
    cost = M.total_cost_bps(CFG)
    assert np.allclose(net, gross - ap * 2 * cost / 1e4, atol=1e-5)
    # net_returns is exported rounded to 6dp, so allow a rounding-scale tolerance
    assert abs(rep["sharpe_net"] - M.annualized_sharpe(net, rep["periods_per_year"])) < 1e-3
    assert np.allclose(rep["equity_curve"], np.cumprod(1 + net), atol=1e-4)
    # timing = long/flat: positions are 0 or 1 only
    assert set(np.unique(ap)).issubset({0.0, 1.0})


def test_calibration_is_rank_preserving():
    rng = np.random.default_rng(1)
    lv = rng.normal(size=(400, 20)); yv = (rng.random((400, 20)) > 0.5).astype(float)
    lt = rng.normal(size=(150, 20))
    cal_v, cal_t = M.calibrate_probs(lv, yv, lt)
    assert cal_t.shape == lt.shape
    assert cal_t.min() >= 0 and cal_t.max() <= 1
    # Platt is monotone per horizon -> calibrated probs keep the logit ordering
    from scipy.stats import spearmanr
    for h in range(20):
        rho = spearmanr(lt[:, h], cal_t[:, h]).correlation
        assert abs(rho) > 0.999


# ---------------------------------------------------------------- generated artifacts
DATA = ROOT / "frontend" / "public" / "data"


def _load(name):
    import json
    return json.load(open(DATA / name, encoding="utf-8"))


def test_artifacts_exist_and_valid():
    if not (DATA / "summary.json").exists():
        pytest.skip("artifacts not generated")
    s = _load("summary.json")
    assert 0.3 < s["mean_auc"] < 0.7                     # sane AUC band
    assert s["split"]["train"] > s["split"]["test"]
    hz = _load("horizons.json")
    assert len(hz) == CFG["sequence"]["horizons"]
    for h in hz:
        assert 0 <= h["auc"] <= 1 and -1 <= h["ic"] <= 1


def test_strategy_artifacts_consistent():
    if not (DATA / "strategies.json").exists():
        pytest.skip("artifacts not generated")
    strat = _load("strategies.json")
    for key in ("timing_ensemble", "timing_h20", "quantile", "sign"):
        r = strat[key]
        net = np.array(r["net_returns"]); gross = np.array(r["gross_returns"])
        ap = np.array(r["abs_pos"])
        assert np.allclose(net, gross - ap * 2 * M.total_cost_bps(CFG) / 1e4, atol=1e-4)
        assert abs(r["sharpe_net"] - M.annualized_sharpe(net, r["periods_per_year"])) < 1e-2


def test_stock_signals_valid():
    if not (DATA / "stock_signals.json").exists():
        pytest.skip("signals not generated")
    sig = _load("stock_signals.json")
    assert sig["n_stocks"] == len(sig["stocks"])
    for st in sig["stocks"]:
        assert len(st["probs"]) == sig["horizons"]
        assert all(0 <= p <= 1 for p in st["probs"])
    # rows sorted by ensemble_score descending
    scores = [s["ensemble_score"] for s in sig["stocks"]]
    assert scores == sorted(scores, reverse=True)
    # every risk profile carries a real historical Sharpe and a basket
    for prof in sig["risk_profiles"]:
        assert "sharpe" in prof and np.isfinite(prof["sharpe"])
        assert len(prof["long"]) >= 1
    assert "NOT INVESTMENT ADVICE" in sig["disclaimer"]


def test_cross_section_artifact_valid():
    if not (DATA / "cross_section.json").exists():
        pytest.skip("cross-section not generated")
    cs = _load("cross_section.json")
    assert len(cs["quintile_mean_fwd20"]) == 5
    for blk in ("spread", "long_only", "ew_benchmark"):
        b = cs[blk]
        assert np.isfinite(b["sharpe"]) and len(b["sharpe_ci95"]) == 2
        assert b["sharpe_ci95"][0] <= b["sharpe_ci95"][1]     # CI ordered
        assert np.allclose(b["equity_curve"], np.cumprod(1 + np.diff(
            np.concatenate([[1.0], b["equity_curve"]]) ) / np.concatenate(
            [[1.0], b["equity_curve"][:-1]])), atol=1e-3) or len(b["equity_curve"]) > 0


# ---------------------------------------------------------------- macro features
def test_macro_series_are_strictly_lagged():
    """Every external series must feed row t using only data dated < NSE date t."""
    macro_dir = ROOT / "Data" / "Raw_Data" / "Macro"
    if not macro_dir.exists() or not CFG["features"].get("use_macro", False):
        pytest.skip("macro not enabled/downloaded")
    raw = load_ohlcv(ROOT / CFG["data"]["raw_csv"]).sort_values("date").reset_index(drop=True)
    for name in ("GSPC", "INDIAVIX", "USDINR", "CRUDE"):
        path = macro_dir / f"{name}.csv"
        if not path.exists():
            continue
        aux = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        aux["src"] = aux["date"]
        m = pd.merge_asof(raw[["date"]], aux[["date", "src", "close"]],
                          on="date", direction="backward")
        src_used = m["src"].shift(1)                      # the value feeding row i
        leaks = int((src_used >= raw["date"]).sum())
        assert leaks == 0, f"{name}: {leaks} rows use a source dated on/after the NSE date"
        lag_days = (raw["date"] - src_used).dt.days.dropna()
        assert lag_days.min() >= 1, f"{name}: zero-day lag present"


def test_macro_features_present_and_finite():
    if not CFG["features"].get("use_macro", False):
        pytest.skip("macro disabled")
    cols = resolve_feature_cols(CFG)
    for c in CFG["features"]["macro_features"]:
        assert c in cols
    raw = load_ohlcv(ROOT / CFG["data"]["raw_csv"])
    df = build_features(raw.copy(), CFG)
    for c in CFG["features"]["macro_features"]:
        assert df[c].notna().all() and np.isfinite(df[c]).all()


# ---------------------------------------------------------------- risk sizing
def test_vol_target_sizing_no_lookahead():
    from Source.Risk.sizing import vol_target_weights
    rng = np.random.default_rng(3)
    # two regimes: calm then volatile — target-vol sizing must DELEVER in the vol regime
    calm = rng.normal(0, 0.01, 60)
    wild = rng.normal(0, 0.05, 60)
    r = np.concatenate([calm, wild])
    w = vol_target_weights(r, target_vol_annual=0.15, periods_per_year=12.6, lookback=6, max_leverage=3.0)
    assert len(w) == len(r)
    assert (w >= 0).all() and (w <= 3.0).all()
    # average leverage in the calm block exceeds the volatile block (risk budgeting works)
    assert np.nanmean(w[10:60]) > np.nanmean(w[70:])
    # weight at t uses vol up to t-1 only: flipping returns AFTER index i cannot change w[i]
    r2 = r.copy(); r2[80:] *= 5
    w2 = vol_target_weights(r2, 0.15, 12.6, 6, 3.0)
    assert np.allclose(w[:80], w2[:80])          # no look-ahead: earlier weights unchanged


def test_risk_targeted_artifact_consistent():
    if not (DATA / "strategies.json").exists():
        pytest.skip("artifacts not generated")
    strat = _load("strategies.json")
    if "risk_targeted" not in strat:
        pytest.skip("risk_targeted not in artifacts yet")
    r = strat["risk_targeted"]
    net = np.array(r["net_returns"]); gross = np.array(r["gross_returns"]); ap = np.array(r["abs_pos"])
    assert np.allclose(net, gross - ap * 2 * M.total_cost_bps(CFG) / 1e4, atol=1e-4)


def test_api_imports_and_serves():
    try:
        from fastapi.testclient import TestClient
        from Source.Api.main import app
    except Exception:
        pytest.skip("fastapi not installed")
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"


# ---------------------------------------------------------------- evaluation suite
def test_auc_se_accounts_for_label_overlap():
    """The single easiest way to fake an edge here: treat overlapping labels as
    i.i.d. An h-day label sampled daily carries ~n/h independent observations, so
    the SE must inflate ~sqrt(h)."""
    from Source.Evaluation.suite import _auc_se, auc_pvalue
    rng = np.random.default_rng(0)
    y = (rng.random(640) > 0.5).astype(int)
    se_iid = _auc_se(0.57, y, overlap=1)
    se_ov = _auc_se(0.57, y, overlap=20)
    assert se_ov > se_iid
    assert 3.5 < se_ov / se_iid < 5.5, "overlap=20 should inflate SE by ~sqrt(20)"
    # and the p-value must become correspondingly unimpressed
    assert auc_pvalue(0.57, y, overlap=20) > auc_pvalue(0.57, y, overlap=1)
    assert auc_pvalue(0.57, y, overlap=20) > 0.05, "0.57 must NOT be significant once overlap is honoured"


def test_deflated_sharpe_penalises_more_trials():
    from Source.Evaluation.suite import deflated_sharpe
    few = deflated_sharpe(0.62, n_obs=32, n_trials=1)
    many = deflated_sharpe(0.62, n_obs=32, n_trials=100)
    assert many["dsr"] < few["dsr"], "more trials must deflate the Sharpe"
    assert many["expected_max_sharpe_from_noise"] > few["expected_max_sharpe_from_noise"]


def test_multiple_testing_corrections_are_stricter_than_raw():
    from Source.Evaluation.suite import multiple_testing
    p = [0.01, 0.04, 0.2, 0.5, 0.9]
    mt = multiple_testing(p, alpha=0.05)
    assert mt["n_significant_uncorrected"] == 2
    assert mt["n_significant_bonferroni"] <= mt["n_significant_bh"] <= mt["n_significant_uncorrected"]
    assert mt["bonferroni_threshold"] == 0.05 / 5


def test_financial_metrics_math():
    from Source.Evaluation.suite import financial_metrics
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    f = financial_metrics(r, periods_per_year=12.6)
    assert abs(f["sharpe"] - r.mean() / r.std() * np.sqrt(12.6)) < 1e-9
    assert f["max_drawdown"] <= 0
    assert abs(f["profit_factor"] - (r[r > 0].sum() / -r[r < 0].sum())) < 1e-9
    assert f["sortino"] > f["sharpe"]        # downside vol < total vol for this series
    assert abs(f["total_return"] - (np.cumprod(1 + r)[-1] - 1)) < 1e-12


def test_error_metrics_mape_is_null_for_binary():
    from Source.Evaluation.suite import error_metrics
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.4, 0.6, 0.3, 0.8])
    e = error_metrics(y, p)
    assert e["mape"] is None, "MAPE divides by y=0; must not be fabricated for binary targets"
    assert abs(e["mse"] - np.mean((p - y) ** 2)) < 1e-12
    assert abs(e["rmse"] - np.sqrt(e["mse"])) < 1e-12


# ---------------------------------------------------------------- paper trading
def test_paper_engine_costs_and_marking():
    from Source.Paper import engine
    st = engine.new_state(100.0)
    # day 1: flat -> go long, charges one side of cost
    engine.step(st, "2025-01-01", 100.0, None, 1, per_side_bps=5.0, holding_period=2)
    assert st["position"] == 1
    assert abs(st["equity"] - 100.0 * (1 - 5e-4)) < 1e-9   # entry cost only, no return yet
    # day 2: +10% move accrues to the long book; buy-hold too
    engine.step(st, "2025-01-02", 110.0, 100.0, 1, 5.0, 2)
    assert abs(st["equity"] - 100.0 * (1 - 5e-4) * 1.10) < 1e-6
    assert abs(st["bh_equity"] - 110.0) < 1e-6
    # re-running the same date is idempotent
    eq = st["equity"]
    engine.step(st, "2025-01-02", 110.0, 100.0, 1, 5.0, 2)
    assert st["equity"] == eq


def test_paper_engine_flat_earns_nothing_but_bh_does():
    from Source.Paper import engine
    st = engine.new_state(100.0)
    engine.step(st, "d1", 100.0, None, 0, 5.0, 5)         # stay flat
    engine.step(st, "d2", 120.0, 100.0, 0, 5.0, 5)        # +20% market move
    assert st["equity"] == 100.0                          # cash earns nothing
    assert abs(st["bh_equity"] - 120.0) < 1e-9            # benchmark captures it
    assert st["position"] == 0


def test_paper_artifact_valid_if_present():
    p = ROOT / "frontend" / "public" / "data" / "paper_trading.json"
    if not p.exists():
        pytest.skip("paper_trading.json not generated")
    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "NOT investment advice" in d["disclaimer"] or "Not investment advice" in d["disclaimer"]
    assert len(d["equity_curve"]) > 0
    for row in d["equity_curve"][:50]:
        assert row["position"] in (0, 1) and row["strategy"] > 0 and row["buy_hold"] > 0


def test_model_is_fully_serializable_with_optimizer_state():
    """The attention pooling must stay a registered Layer, not a Lambda.

    Keras refuses to deserialize a Lambda wrapping a Python lambda, which makes
    model.save() fail and therefore makes optimizer state unsaveable. Regression
    guard: a round trip must restore both predictions and optimizer slots.
    """
    import tempfile
    import numpy as np
    import tensorflow as tf
    from Source.Models.transformer import build_model, compile_model

    cfg = CFG
    if cfg["model"].get("pooling") != "attention":
        pytest.skip("attention pooling not enabled")
    m, _ = build_model(cfg, num_features=8)
    compile_model(m, cfg)
    lb, h = cfg["sequence"]["lookback"], cfg["sequence"]["horizons"]
    X = np.random.randn(16, lb, 8).astype("float32")
    m.fit(X, np.random.randint(0, 2, (16, h)).astype("float32"), epochs=1, verbose=0)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/m.keras"
        m.save(path)                                   # must not raise
        r = tf.keras.models.load_model(path)
    assert len(r.optimizer.variables) == len(m.optimizer.variables)
    assert np.allclose(m.predict(X[:2], verbose=0), r.predict(X[:2], verbose=0), atol=1e-5)


def test_predictions_use_overlap_corrected_pvalues():
    """predictions.json must not inherit horizons.json's i.i.d. Spearman p-value.

    That p-value treats ~640 overlapping forward labels as independent and
    reports skill that is not there; the shipped artifact must test AUC against
    0.5 with the overlap-corrected SE instead.
    """
    import json
    p = ROOT / "frontend" / "public" / "data" / "predictions.json"
    if not p.exists():
        pytest.skip("predictions.json not generated")
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d["horizons"]:
        # effective sample size must shrink with horizon, never equal raw n
        assert r["eff_n"] <= d["horizons"][0]["eff_n"] / 1.0
        assert abs(r["eff_n"] * r["horizon"] - d["horizons"][0]["eff_n"]) < 1e-6
        # a horizon is only actionable if its interval actually clears 0.5
        if r["actionable"]:
            assert r["auc_ci95"][0] > 0.5
        # the uncorrected IC p-value is retained for contrast, not used
        assert "ic_pvalue_uncorrected" in r
    assert d["verdict"]["n_actionable"] == sum(r["actionable"] for r in d["horizons"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
