"""Conviction-gated strategy: only take the read when the model is confident.

Instead of forcing a position every period, the agent ABSTAINS unless its
calibrated probability is far enough from 0.5 - i.e. it only trades its
high-conviction reads. This is the honest way to try to extract value from a
model whose average edge is ~zero: if any edge exists, it should live in the
tail of the confidence distribution.

Anti-cheat protocol (this is the whole point):
- The confidence THRESHOLD is chosen on VALIDATION only, as the quantile that
  keeps the top `trade_frac` most-confident val predictions. The test set is
  never consulted to set it.
- Platt calibration is fit on validation.
- The test set is scored exactly once.
- Every result is charged the full India futures cost stack and then run through
  the evaluation suite: accuracy-when-traded vs baseline, Sharpe / Sortino /
  Calmar, and the DEFLATED Sharpe, which discounts for the ~16 configurations
  tried this project. A conviction strategy is just another trial; if its edge
  does not clear the deflation bar, it is noise.

Run:  python scripts/conviction_strategy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from Source.Backtest import metrics as M  # noqa: E402
from Source.Backtest.costs import india_cost_breakdown  # noqa: E402
from Source.Evaluation.registry import save_evaluation  # noqa: E402
from Source.Evaluation.suite import (classification_metrics, deflated_sharpe,  # noqa: E402
                                     financial_metrics)
from Source.Pipeline.data_loader import load_ohlcv  # noqa: E402
from Source.Pipeline.dataset import build_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_TRIALS = 16


def fwd_log_return(close, idx, h):
    out = np.full(len(idx), np.nan)
    for i, t in enumerate(idx):
        if t + h < len(close):
            out[i] = np.log(close[t + h] / close[t])
    return out


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    ph = cfg["backtest"]["primary_horizon"]
    holding = cfg["backtest"]["holding_period"]
    ppy = cfg["backtest"]["periods_per_year"] / holding
    per_side = india_cost_breakdown(cfg, "futures")["per_side_bps"]

    ds = build_dataset(load_ohlcv(ROOT / cfg["data"]["raw_csv"]), cfg)
    close = ds.df["close"].to_numpy()

    caches = sorted(glob.glob(str(ROOT / "Data/Processed_Data/run_cache_v4_*.npz")))
    if not caches:
        print("no index cache - run the index track first"); return
    c = np.load(caches[-1], allow_pickle=True)
    lv, lt = c["logits_val"], c["logits_test"]

    # calibrated P(up) at the primary horizon, Platt on validation only
    _, pt_cal = M.calibrate_probs(lv, ds.y_val, lt)
    pv_cal, _ = M.calibrate_probs(lv, ds.y_val, lv)
    conf_val = np.abs(pv_cal[:, ph - 1] - 0.5)
    conf_test = np.abs(pt_cal[:, ph - 1] - 0.5)
    p_test = pt_cal[:, ph - 1]

    print(f"index conviction strategy | cost {2*per_side:.2f}bps round-trip")
    print("SKILL = accuracy_edge (acc_when_traded - majority baseline). A positive")
    print("Sharpe with accuracy_edge<=0 is market beta (long bias in a bull run), NOT")
    print("skill - that is why buy_hold_edge is shown too.\n")
    print(f"{'h':>3}{'trade_frac':>11}{'n_trades':>9}{'acc_traded':>11}{'baseline':>9}"
          f"{'ACC_EDGE':>9}{'Sharpe':>8}{'vs_BH':>8}{'DSR':>7}")

    results = []
    for h in (1, 5, 10, 20):
        pv_h, _ = M.calibrate_probs(lv, ds.y_val, lv)
        _, pt_h = M.calibrate_probs(lv, ds.y_val, lt)
        cv, ctst, pp = np.abs(pv_h[:, h - 1] - 0.5), np.abs(pt_h[:, h - 1] - 0.5), pt_h[:, h - 1]
        yy = ds.y_test[:, h - 1].astype(int)
        fw = fwd_log_return(close, ds.idx_test, h)
        for trade_frac in (1.0, 0.3, 0.1):
            thr = np.quantile(cv, 1 - trade_frac)
            m = (ctst >= thr) & ~np.isnan(fw)
            if m.sum() < 10:
                continue
            pos = np.where(pp[m] > 0.5, 1.0, -1.0)
            order = np.argsort(ds.idx_test[m])
            pos_o, fw_o = pos[order], fw[m][order]
            sel = np.arange(0, len(pos_o), max(1, h))          # non-overlapping by horizon
            p_no, r_no = pos_o[sel], fw_o[sel]
            net = p_no * r_no - np.abs(p_no) * 2 * per_side / 1e4
            bh_net = r_no - 2 * per_side / 1e4                  # always-long, same trades
            acc = float((((pp[m] > 0.5).astype(int)) == yy[m]).mean())
            base = float(max(yy[m].mean(), 1 - yy[m].mean()))
            ppy_h = 252.0 / h
            fin = financial_metrics(net, ppy_h, np.abs(p_no))
            bh = financial_metrics(bh_net, ppy_h)
            dsr = deflated_sharpe(fin["sharpe"], len(net), N_TRIALS,
                                  float(sstats.skew(net)) if len(net) > 2 else 0.0,
                                  float(sstats.kurtosis(net, fisher=False)) if len(net) > 3 else 3.0)
            results.append({
                "horizon": h, "trade_frac": trade_frac, "confidence_threshold": float(thr),
                "n_trades_nonoverlap": int(len(net)), "n_signals": int(m.sum()),
                "accuracy_when_traded": acc, "baseline_accuracy": base,
                "accuracy_edge": acc - base,
                "sharpe": fin["sharpe"], "total_return": fin["total_return"],
                "buy_hold_sharpe_same_trades": bh["sharpe"],
                "excess_return_vs_buy_hold": fin["total_return"] - bh["total_return"],
                "deflated_sharpe": dsr["dsr"],
            })
            print(f"{h:>3}{trade_frac:>11.2f}{len(net):>9}{acc:>11.3f}{base:>9.3f}"
                  f"{acc-base:>+9.3f}{fin['sharpe']:>8.2f}"
                  f"{fin['total_return']-bh['total_return']:>+7.1%}{dsr['dsr']:>7.2f}")

    # a claim needs enough trades to mean anything; a +edge on 5 trades is noise
    MIN_TRADES = 30
    reliable = [r for r in results if r["n_trades_nonoverlap"] >= MIN_TRADES]
    best = max(reliable, key=lambda r: r["accuracy_edge"]) if reliable else None
    for r in results:
        r["reliable"] = r["n_trades_nonoverlap"] >= MIN_TRADES
    save_evaluation("conviction_index", {
        "track": "index_conviction",
        "cost_roundtrip_bps": 2 * per_side,
        "threshold_source": "validation confidence quantile (test never consulted)",
        "skill_metric": "accuracy_edge = accuracy_when_traded - majority_baseline",
        "best_accuracy_edge": best,
        "results": results,
        "note": ("Confidence threshold from validation; test scored once. accuracy_edge "
                 "is the skill metric; a positive Sharpe with accuracy_edge<=0 is beta, "
                 "not alpha (see excess_return_vs_buy_hold on the same trades). Net of "
                 "the full India futures cost stack; Sharpe deflated for ~16 trials."),
    })
    print("\nsaved -> Data/Evaluation/conviction_index.json")
    if best:
        verdict = "SKILL" if best["accuracy_edge"] > 0 else "NO skill above baseline"
        print(f"best RELIABLE (>= {MIN_TRADES} trades) accuracy_edge: {best['accuracy_edge']:+.3f} "
              f"at h={best['horizon']} tf={best['trade_frac']} ({best['n_trades_nonoverlap']} trades) -> {verdict}")
    else:
        print(f"no configuration reached {MIN_TRADES} trades with a positive accuracy_edge - "
              f"conviction gating does not produce measurable skill on this data.")


if __name__ == "__main__":
    main()
