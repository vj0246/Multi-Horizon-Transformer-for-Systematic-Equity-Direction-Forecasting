"""Intraday track: hourly bars, ~3-day holds, same encoder, same honesty rules.

This is the first configuration in the project where the experiment can actually
resolve the effect it is looking for. On daily bars at a 20-day horizon the test
set carries ~32 independent observations, so an AUC of ~0.686 would be needed for
significance while a realistic edge is 0.52-0.55: the experiment cannot conclude
anything either way. Hourly bars at a 20-BAR horizon give ~253 independent
observations (detectable AUC ~0.571) while costs stay at ~4% of the typical move.

Note the trap this avoids: hourly bars predicting the same 20-DAY horizon would
be WORSE, not better (~5 independent observations), because sampling more finely
does not create more independent 20-day periods. Frequency and horizon have to
move together.

Nothing else changes. Same Transformer, same temporal split, same scaler-on-train
only, same overlap-corrected error bars, same India costs.

Run:  python -m Source.Intraday.run
      python -m Source.Intraday.run --sentiment    (needs gdelt_sentiment.csv)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISM", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from Source.Backtest.costs import india_cost_breakdown  # noqa: E402
from Source.Evaluation.suite import _auc_se, auc_pvalue, multiple_testing  # noqa: E402
from Source.Intraday import features as F  # noqa: E402
from Source.Intraday import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "data" / "intraday.json"
SENT = ROOT / "Data" / "Processed_Data" / "gdelt_sentiment.csv"


def build_windows(d: pd.DataFrame, cols: list[str], lookback: int, horizons: int):
    feats = d[cols].to_numpy(dtype="float32")
    targs = d[[f"target_{h}" for h in range(1, horizons + 1)]].to_numpy(dtype="float32")
    idx = np.arange(lookback, len(d) - horizons)
    X = np.stack([feats[t - lookback:t] for t in idx]).astype("float32")
    y = targs[idx]
    return X, y, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizons", type=int, default=20, help="in BARS, not days")
    ap.add_argument("--lookback", type=int, default=60, help="in BARS")
    ap.add_argument("--sentiment", action="store_true", help="fuse GDELT tone")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    from Source.device import configure_devices
    configure_devices(cfg)

    bars_csv = ROOT / "Data" / "Raw_Data" / "Intraday" / f"NSEI_{args.interval}.csv"
    if not bars_csv.exists():
        raise SystemExit(f"{bars_csv} missing - run: python -m Source.Intraday.fetch "
                         f"--interval {args.interval}")
    raw = fetch.from_csv(bars_csv)
    d = F.add_intraday_features(raw)

    use_sent = args.sentiment
    if use_sent:
        if not SENT.exists():
            raise SystemExit(f"{SENT} missing - run: python -m Source.News.gdelt --days 730")
        from Source.News.gdelt import attach_sentiment
        tone = pd.read_csv(SENT, parse_dates=["datetime"])
        d = attach_sentiment(d, tone, time_col="datetime", lag_bars=1)

    cols = F.resolve_features(use_sent)
    d = F.make_targets(d, args.horizons)
    d = d.dropna(subset=cols).reset_index(drop=True)

    X, y, idx = build_windows(d, cols, args.lookback, args.horizons)
    n = len(X)
    if n < 500:
        raise SystemExit(f"only {n} windows - too few to split honestly")

    # temporal split, no shuffling - identical rule to the daily track
    tr_end = int(0.70 * n)
    va_end = int(0.85 * n)
    from sklearn.preprocessing import StandardScaler
    nf = len(cols)
    scaler = StandardScaler().fit(X[:tr_end].reshape(-1, nf))     # TRAIN ONLY

    def sc(a):
        return scaler.transform(a.reshape(-1, nf)).reshape(a.shape).astype("float32")

    Xtr, Xva, Xte = sc(X[:tr_end]), sc(X[tr_end:va_end]), sc(X[va_end:])
    ytr, yva, yte = y[:tr_end], y[tr_end:va_end], y[va_end:]

    import tensorflow as tf
    from Source.Models.transformer import build_model, compile_model
    icfg = dict(cfg)
    icfg["sequence"] = {"lookback": args.lookback, "horizons": args.horizons}

    n_seeds = int(cfg["training"].get("n_seeds", 3))
    acc = None
    for s in range(n_seeds):
        tf.keras.utils.set_random_seed(cfg["training"]["seed"] + s)
        m, _ = build_model(icfg, num_features=nf)
        compile_model(m, icfg)
        es = tf.keras.callbacks.EarlyStopping(
            patience=cfg["training"]["early_stopping_patience"], restore_best_weights=True)
        m.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=cfg["training"]["epochs"],
              batch_size=cfg["training"]["batch_size"], callbacks=[es], verbose=0)
        p = m.predict(Xte, verbose=0, batch_size=256)
        acc = p if acc is None else acc + p
    logits = acc / n_seeds

    # ---- per-horizon skill, overlap-corrected exactly as the daily track ----
    from sklearn.metrics import roc_auc_score
    rows = []
    for h in range(1, args.horizons + 1):
        yy = yte[:, h - 1].astype(int)
        if len(np.unique(yy)) < 2:
            continue
        auc = float(roc_auc_score(yy, logits[:, h - 1]))
        se = _auc_se(auc, yy, overlap=h)
        rows.append({"horizon_bars": h, "auc": auc, "auc_se": se,
                     "auc_ci95": [auc - 1.96 * se, auc + 1.96 * se],
                     "eff_n": len(yy) / h,
                     "p_value": auc_pvalue(auc, yy, overlap=h)})
    mt = multiple_testing([r["p_value"] for r in rows])
    bh = mt.get("bh_reject", [False] * len(rows))
    for r, flag in zip(rows, bh):
        r["significant_bh"] = bool(flag)

    # ---- economics: does any edge survive the round trip? ------------------
    rt = india_cost_breakdown(cfg, "futures")["roundtrip_bps"]
    bars_per_day = len(d) / max((d["datetime"].max() - d["datetime"].min()).days, 1)
    hold_days = args.horizons / bars_per_day
    close = d["close"].to_numpy()
    te_idx = idx[va_end:]
    fwd = close[te_idx + args.horizons] / close[te_idx] - 1.0
    typical_move_bps = float(np.std(fwd) * 1e4)
    breakeven_wr = 0.5 + rt / (2 * typical_move_bps)

    prim = rows[-1] if rows else {}
    n_sig = sum(r["significant_bh"] for r in rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": args.interval,
        "horizon_bars": args.horizons,
        "horizon_days_approx": round(hold_days, 2),
        "bars_per_day": round(bars_per_day, 2),
        "n_bars": int(len(d)),
        "n_windows": int(n),
        "split": {"train": tr_end, "val": va_end - tr_end, "test": n - va_end},
        "features": cols,
        "use_sentiment": bool(use_sent),
        "mean_auc": float(np.mean([r["auc"] for r in rows])) if rows else None,
        "n_significant_bh": int(n_sig),
        "horizons": rows,
        "economics": {
            "roundtrip_cost_bps": rt,
            "typical_move_bps": typical_move_bps,
            "cost_share_of_move": rt / typical_move_bps,
            "breakeven_win_rate": breakeven_wr,
            "note": (
                "An edge must clear the break-even win rate to be tradable, not "
                "merely clear 0.50. Statistical significance and profitability "
                "are separate bars and both must be passed."
            ),
        },
        "why_this_configuration": (
            f"{args.interval} bars held {args.horizons} bars (~{hold_days:.1f} days) "
            f"gives ~{prim.get('eff_n', 0):.0f} independent observations at the "
            "primary horizon, against ~32 on the daily track. Sampling finer while "
            "keeping a 20-DAY horizon would instead REDUCE independent "
            "observations to ~5 - frequency and horizon must move together."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"intraday {args.interval} | {len(d):,} bars | {n:,} windows "
          f"| test {n - va_end:,}")
    print(f"  mean AUC {report['mean_auc']:.4f} | {n_sig}/{len(rows)} horizons "
          f"significant after BH correction")
    print(f"  economics: typical move {typical_move_bps:.0f}bps vs {rt:.2f}bps cost "
          f"-> break-even win rate {breakeven_wr:.1%}")
    if rows:
        lo, hi = prim["auc_ci95"]
        print(f"  h={prim['horizon_bars']} bars: AUC {prim['auc']:.4f} "
              f"[{lo:.3f}, {hi:.3f}], eff n {prim['eff_n']:.0f}")


if __name__ == "__main__":
    main()
