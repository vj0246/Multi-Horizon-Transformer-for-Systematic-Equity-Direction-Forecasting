"""Read-only FastAPI serving layer over the generated backtest artifacts.

The showcase site is fully static, but this exposes the same JSON over HTTP for
programmatic consumers. It never trains or mutates anything - it reads the files
in `output.artifacts_dir` produced by the backtest runs.

Run:
    uvicorn Source.Api.main:app --reload --port 8000
    # then GET http://localhost:8000/health, /summary, /signals, /cross-section
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DATA = ROOT / CFG["output"]["artifacts_dir"]

app = FastAPI(
    title="Multi-Horizon Transformer API",
    description="Read-only access to the Nifty 50 / NSE backtest artifacts. "
                "Research output only - not investment advice.",
    version="1.0.0",
)


def _read(name: str) -> dict | list:
    path = DATA / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts_dir": str(DATA), "available": sorted(p.name for p in DATA.glob("*.json"))}


@app.get("/summary")
def summary() -> dict:
    return _read("summary.json")


@app.get("/horizons")
def horizons() -> list:
    return _read("horizons.json")


@app.get("/strategies")
def strategies() -> dict:
    return _read("strategies.json")


@app.get("/cross-section")
def cross_section() -> dict:
    return _read("cross_section.json")


@app.get("/signals")
def signals() -> dict:
    """Latest per-stock, all-horizon signal. RESEARCH ONLY - not advice."""
    return _read("stock_signals.json")


@app.get("/signals/{ticker}")
def signal_for(ticker: str) -> dict:
    sig = _read("stock_signals.json")
    for row in sig.get("stocks", []):
        if row["ticker"].upper() == ticker.upper():
            return {"as_of": sig["as_of"], "disclaimer": sig["disclaimer"], **row}
    raise HTTPException(status_code=404, detail=f"{ticker} not in universe")
