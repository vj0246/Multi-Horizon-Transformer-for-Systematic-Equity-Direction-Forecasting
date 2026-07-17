"""Per-model evaluation registry.

Every evaluated model writes one JSON to Data/Evaluation/<name>.json holding its
full metric set (classification, error, financial, statistical) plus the context
needed to read it honestly: the data window, sample sizes, effective sample size,
cost assumptions, and how many trials it was selected from.

`leaderboard()` collates them. It ranks by DEFLATED Sharpe and reports the
multiple-testing verdict, not by raw best-AUC - ranking a family of runs by their
maximum is how noise gets promoted to a discovery.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "Data" / "Evaluation"


def save_evaluation(name: str, payload: dict) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": name,
               "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **payload}
    path = EVAL_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_all() -> list[dict]:
    if not EVAL_DIR.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(EVAL_DIR.glob("*.json"))]


def leaderboard() -> pd.DataFrame:
    rows = []
    for e in load_all():
        cls = e.get("classification", {}) or {}
        fin = e.get("financial", {}) or {}
        mt = e.get("multiple_testing", {}) or {}
        dsr = e.get("deflated_sharpe", {}) or {}
        rows.append({
            "model": e.get("model"),
            "mean_auc": e.get("mean_auc"),
            "best_auc": e.get("best_auc"),
            "auc_se": cls.get("auc_se"),
            "sig_after_bh": mt.get("n_significant_bh"),
            "n_tests": mt.get("n_tests"),
            "sharpe": fin.get("sharpe"),
            "dsr": dsr.get("dsr"),
            "total_return": fin.get("total_return"),
            "max_dd": fin.get("max_drawdown"),
            "eff_n": e.get("effective_sample_size"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "dsr" in df:
        df = df.sort_values("dsr", ascending=False, na_position="last")
    return df.reset_index(drop=True)
