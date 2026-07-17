"""Evaluate every model with the full metric suite and write the registry.

Applies Source/Evaluation/suite.py to each model's TEST predictions and saves one
JSON per model to Data/Evaluation/, then prints the leaderboard.

The two headline questions it answers honestly:
  1. Does the best per-horizon AUC survive multiple-testing correction across the
     20 horizons? (a maximum over 20 correlated draws is not a discovery)
  2. Does the trading Sharpe survive deflation for the number of configurations
     tried this project? (a best-of-N Sharpe is biased upward)

Run:  python scripts/evaluate_models.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from Source.Evaluation.registry import leaderboard, save_evaluation  # noqa: E402
from Source.Evaluation.suite import (auc_pvalue, classification_metrics,  # noqa: E402
                                     deflated_sharpe, diebold_mariano,
                                     error_metrics, financial_metrics,
                                     friedman_test, multiple_testing)
from Source.Pipeline.data_loader import load_ohlcv  # noqa: E402
from Source.Pipeline.dataset import build_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Configurations genuinely tried on this project (used to deflate the Sharpe).
# Counted honestly: architecture grid (6) + objectives (2) + feature blocks
# (base/macro/positioning = 3) + model families (transformer/GBDT = 2) + threshold
# rules (3). Under-counting trials inflates the DSR, so this errs high.
N_TRIALS = 16


def evaluate_transformer_index(cfg) -> dict:
    caches = sorted(glob.glob(str(ROOT / "Data/Processed_Data/run_cache_v4_*.npz")))
    if not caches:
        return {}
    c = np.load(caches[-1], allow_pickle=True)
    lt = c["logits_test"]
    ds = build_dataset(load_ohlcv(ROOT / cfg["data"]["raw_csv"]), cfg)
    prob = 1 / (1 + np.exp(-lt))
    H = cfg["sequence"]["horizons"]
    ph = cfg["backtest"]["primary_horizon"]
    holding = cfg["backtest"]["holding_period"]
    ppy = cfg["backtest"]["periods_per_year"] / holding

    per_h, pvals = [], []
    for h in range(H):
        # overlap = h+1: an (h+1)-day label sampled daily overlaps its ~h neighbours
        m = classification_metrics(ds.y_test[:, h], prob[:, h], overlap=h + 1)
        p = auc_pvalue(m["auc"], ds.y_test[:, h].astype(int), overlap=h + 1)
        per_h.append({"horizon": h + 1, **m, "auc_p_value": p})
        pvals.append(p)

    aucs = [r["auc"] for r in per_h]
    best_i = int(np.nanargmax(aucs))

    # financial: the published primary strategy
    import json
    strat = json.loads((ROOT / "frontend/public/data/strategies.json").read_text())
    prim = strat.get("timing_rolling") or strat.get("timing_ensemble")
    net = np.asarray(prim["net_returns"], dtype=float)
    fin = financial_metrics(net, prim.get("periods_per_year", ppy),
                            np.asarray(prim.get("abs_pos", [])))
    bh = np.asarray(strat["buy_and_hold"]["net_returns"]) if "net_returns" in strat["buy_and_hold"] else None

    dsr = deflated_sharpe(fin["sharpe"], n_obs=len(net), n_trials=N_TRIALS,
                          skew=float(sstats.skew(net)), kurt=float(sstats.kurtosis(net, fisher=False)))

    # Diebold-Mariano vs an always-up forecast at the primary horizon
    y = ds.y_test[:, ph - 1]
    e_model = prob[:, ph - 1] - y
    e_naive = np.full_like(y, y.mean()) - y
    dm = diebold_mariano(e_model, e_naive, h=holding)

    return {
        "track": "index",
        "n_features": len(ds.feature_cols),
        "data_window": [str(ds.df.date.min().date()), str(ds.df.date.max().date())],
        "samples": {"train": int(len(ds.X_train)), "val": int(len(ds.X_val)),
                    "test": int(len(ds.X_test))},
        "effective_sample_size": int(len(ds.X_train) / holding),
        "mean_auc": float(np.nanmean(aucs)),
        "best_auc": float(aucs[best_i]),
        "best_auc_horizon": int(best_i + 1),
        "classification": per_h[ph - 1],
        "per_horizon": per_h,
        "error": error_metrics(ds.y_test[:, ph - 1], prob[:, ph - 1]),
        "financial": fin,
        "deflated_sharpe": dsr,
        "diebold_mariano_vs_naive": dm,
        "multiple_testing": multiple_testing(pvals),
        "costs": {"roundtrip_bps": json.loads(
            (ROOT / "frontend/public/data/summary.json").read_text()).get("roundtrip_cost_bps")},
        "benchmark": {"buy_and_hold_sharpe": strat["buy_and_hold"]["sharpe_net"],
                      "buy_and_hold_total_return": strat["buy_and_hold"]["total_return"]},
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    ev = evaluate_transformer_index(cfg)
    if ev:
        p = save_evaluation("transformer_index", ev)
        print(f"wrote {p.name}")
        mt, d = ev["multiple_testing"], ev["deflated_sharpe"]
        print(f"\nmean AUC {ev['mean_auc']:.4f} | best AUC {ev['best_auc']:.4f} "
              f"(h{ev['best_auc_horizon']}, SE ~{ev['classification']['auc_se']:.3f})")
        print(f"multiple testing over {mt['n_tests']} horizons: min p={mt['min_p']:.3f} | "
              f"significant: uncorrected {mt['n_significant_uncorrected']}, "
              f"BH {mt['n_significant_bh']}, Bonferroni {mt['n_significant_bonferroni']}")
        f = ev["financial"]
        print(f"Sharpe {f['sharpe']:+.2f} | Sortino {f['sortino']:+.2f} | "
              f"Calmar {f['calmar']:+.2f} | maxDD {f['max_drawdown']*100:.1f}% | "
              f"profit factor {f['profit_factor']:.2f}")
        print(f"Deflated Sharpe: P(true SR>0) = {d['dsr']:.3f} "
              f"(noise bar from {d['n_trials']} trials = {d['expected_max_sharpe_from_noise']:.3f})")
        print(f"Diebold-Mariano vs naive: stat {ev['diebold_mariano_vs_naive']['stat']:.2f}, "
              f"p={ev['diebold_mariano_vs_naive']['p_value']:.3f}")

    print("\n=== LEADERBOARD ===")
    print(leaderboard().to_string(index=False))


if __name__ == "__main__":
    main()
