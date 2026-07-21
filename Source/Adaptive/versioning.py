"""Model version registry - provenance for every prediction.

Today the paper book is honest because ONE model is frozen and its training
cutoff is stamped in its metadata, so no traded day can be in-sample. The moment
models are retrained on a schedule that guarantee needs rebuilding: a prediction
is only out-of-sample with respect to the model that actually produced it.

This registry stores, per version:
  train_cutoff  - last date the version was fitted on. A prediction for date D is
                  out-of-sample iff D > train_cutoff of the version that made it.
  trial_index   - how many candidate models have been trained in this lineage,
                  cumulative. This feeds deflated_sharpe(n_trials=...): every
                  retrain is another chance to get lucky, and a family of
                  retrains ranked by its best member is the classic way noise
                  gets promoted to a discovery.
  parent        - the version this one was a challenger to, so the lineage is
                  reconstructable.
  status        - champion | challenger | retired.

`assert_out_of_sample` is the guard the paper book calls. It raises rather than
warns: a silently in-sample prediction invalidates every number downstream of it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "Data" / "Adaptive" / "model_versions.json"


def _load() -> dict:
    if not REG.exists():
        return {"versions": [], "lineage": "index-timing"}
    return json.loads(REG.read_text(encoding="utf-8"))


def _save(reg: dict) -> None:
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def next_trial_index() -> int:
    """Cumulative count of models ever trained in this lineage."""
    return len(_load()["versions"]) + 1


def register(version: str, train_cutoff: str, *, status: str = "challenger",
             parent: str | None = None, metrics: dict | None = None,
             notes: str = "") -> dict:
    reg = _load()
    if any(v["version"] == version for v in reg["versions"]):
        raise ValueError(f"version {version} already registered - versions are immutable")
    entry = {
        "version": version,
        "train_cutoff": str(train_cutoff),
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "parent": parent,
        "trial_index": len(reg["versions"]) + 1,
        "metrics": metrics or {},
        "notes": notes,
    }
    reg["versions"].append(entry)
    _save(reg)
    return entry


def champion() -> dict | None:
    """The version currently serving predictions."""
    for v in reversed(_load()["versions"]):
        if v["status"] == "champion":
            return v
    return None


def promote(version: str, reason: str) -> dict:
    """Make `version` champion and retire the incumbent. Recorded, not silent."""
    reg = _load()
    found = None
    for v in reg["versions"]:
        if v["status"] == "champion":
            v["status"] = "retired"
            v["retired_reason"] = f"superseded by {version}"
        if v["version"] == version:
            found = v
    if found is None:
        raise KeyError(f"unknown version {version}")
    found["status"] = "champion"
    found["promoted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    found["promotion_reason"] = reason
    _save(reg)
    return found


def reject(version: str, reason: str) -> dict:
    """Record a challenger that failed its gate. Kept, not deleted - the count of
    rejected challengers IS the trial count that deflates the Sharpe."""
    reg = _load()
    for v in reg["versions"]:
        if v["version"] == version:
            v["status"] = "retired"
            v["retired_reason"] = reason
            _save(reg)
            return v
    raise KeyError(f"unknown version {version}")


def assert_out_of_sample(version_entry: dict, predict_date: str) -> None:
    """Raise unless `predict_date` is strictly after the version's training cutoff."""
    cutoff = version_entry["train_cutoff"]
    if str(predict_date) <= str(cutoff):
        raise ValueError(
            f"IN-SAMPLE PREDICTION: version {version_entry['version']} was trained "
            f"through {cutoff} but is being asked to predict {predict_date}. Every "
            "metric downstream of this would be invalid."
        )


def summary() -> dict:
    reg = _load()
    vs = reg["versions"]
    champ = champion()
    return {
        "lineage": reg.get("lineage"),
        "n_versions": len(vs),
        "n_trials": len(vs),
        "champion": champ["version"] if champ else None,
        "champion_cutoff": champ["train_cutoff"] if champ else None,
        "n_retired": sum(1 for v in vs if v["status"] == "retired"),
        "versions": vs,
        "note": (
            "n_trials counts every model ever trained in this lineage, including "
            "rejected challengers, and is passed to deflated_sharpe. Reporting the "
            "best of N retrains without deflating by N is how a retraining "
            "schedule manufactures an edge that was never there."
        ),
    }
