# CLAUDE.md — Multi-Horizon Transformer (Nifty 50 Direction)

## Purpose
Deep-learning system predicting Nifty 50 (`^NSEI`) directional movement across 20 forward horizons (1–20 days) with a multi-output Transformer encoder. Research-grade. A static Next.js site showcases the real backtested results.

## Stack
- **Model/pipeline:** Python 3.10, TensorFlow 2.21 (Keras), scikit-learn, pandas, numpy, scipy, PyYAML.
- **Data:** yfinance OHLCV for `^NSEI`, 2007→2026 (~4,520 trading days).
- **Frontend:** Next.js 14 (App Router, static export), TypeScript, Tailwind, Recharts. No backend — reads precomputed JSON.
- **Deploy:** Vercel (static `out/`).

## Run / Build / Test
```bash
# 1. (optional) refresh data
python -m Source.Ingestion.Fetch_Market_Data

# 2. train + full backtest -> writes frontend/public/data/*.json
python -m Source.Backtest.run

# 2b. cross-sectional track (37 NSE large caps, real quantile L/S)
python -m Source.Ingestion.fetch_universe        # one-time data download
python -m Source.Backtest.run_cross_section      # -> cross_section.json

# 3. frontend
cd frontend && npm install && npm run dev      # local
npm run build                                   # static export -> frontend/out

# tests: data-integrity, leakage audits, cost/strategy math, artifact validation
python -m pytest tests/test_rigorous.py -q
```
Everything is driven by `config.yaml` (hyperparams, windows, split, costs).

### GPU training (WSL2 + CUDA)
Native-Windows TF ≥2.11 is CPU-only. Train on the RTX 2050 through WSL2:
```bash
# one-time: CUDA-enabled TF venv inside WSL Ubuntu
wsl -d Ubuntu bash -lc 'python3 -m venv ~/venvs/mht && ~/venvs/mht/bin/pip install "tensorflow[and-cuda]==2.21.0" numpy pandas scipy scikit-learn pyyaml yfinance'

# run training on GPU (repo is on the Windows drive, visible at /mnt/c)
wsl -d Ubuntu bash -lc 'source ~/venvs/mht/bin/activate && cd "/mnt/c/Users/vivaa/OneDrive/Desktop/Personal Projects/Multi-Horizon-Transformer-for-Systematic-Equity-Direction-Forecasting" && python -m Source.Backtest.run_cross_section'
```
`Source/device.py` logs the device and, with `training.require_gpu: true`, aborts rather than silently using CPU. `config.yaml` `training.require_gpu` enforces GPU-only runs.

## Directory Map
- `Documentation/` — full project docs (overview, getting started, data + leakage rules, architecture diagrams, per-file reference, metric definitions, results, instrument choice, research gaps). Update these when behaviour changes; they are the onboarding path for anyone new.
- `Source/Paper/` — paper trading. `engine.py` = long/flat book marked daily (costs, equity, trades); `run.py --refresh` = fetch fresh data, score OOS days with the FROZEN model (`Data/Processed_Data/paper_model/`, from `scripts/save_paper_model.py`), write `frontend/public/data/paper_trading.json`. Daily cron: `.github/workflows/paper_trading.yml` (commits update -> Vercel auto-deploys). Frozen model = trained through validation cutoff only, so all later days are true OOS.
- `Source/Insights/build.py` — current forward predictions (`predictions.json`): scores the latest window with the frozen paper model, reports per-horizon calibrated P(up) beside its overlap-corrected AUC 95% CI and multiple-testing verdict. Skill is measured on the FROZEN model's own OOS period (everything after `oos_cutoff`), never read from `horizons.json` — that artifact belongs to the backtest model (a different fit), so its error bars do not describe this predictor. Also never reuse its `p_value`: that is a Spearman IC p-value over raw overlapping labels (i.i.d. assumption violated); test AUC vs 0.5 with the effective-n SE instead.
- `Source/Paper/frozen.py` — single loader/scorer for the frozen model, shared by `Paper/run.py` and `Insights/build.py` so the paper book and the predictions table can never drift apart. Loads for inference and never compiles (the saved optimizer is skipped deliberately, not half-restored).
- `Source/Evaluation/` — metric suite (classification/error/financial/statistical, overlap-corrected AUC SE, deflated Sharpe, multiple-testing) + per-model registry (`Data/Evaluation/`). `scripts/evaluate_models.py`, `scripts/conviction_strategy.py`, `scripts/gbdt_baseline.py`.
- `Source/Risk/sizing.py` — vol-targeted position sizing (lagged trailing vol, no look-ahead); run.py emits a `risk_targeted` strategy variant. Config `risk:`.
- `Source/Api/main.py` — read-only FastAPI over the artifacts (`uvicorn Source.Api.main:app`). Never trains.
- `Source/Ingestion/fetch_fundamentals.py` — CURRENT fundamentals snapshot (yfinance). Display/context only; NOT model features (current fundamentals as history = leakage).
- `Source/Features/` — Returns.py, Volatility.py (feature engineering)
- `Source/Pipeline/` — data_loader.py (clean CSV), dataset.py (features→targets→windows→split→scale)
- `Source/Models/transformer.py` — TF Transformer (2 blocks, 4 heads, d_model=64, attention pooling) + attention model
- `Source/Backtest/metrics.py` — Sharpe/drawdown/IC/decile/costs
- `Source/Backtest/run.py` — index-track orchestrator, exports JSON artifacts
- `Source/Backtest/run_cross_section.py` — cross-sectional track (panel train + quantile spread)
- `Source/Pipeline/cross_section.py` — panel builder (date-based split, no cross-stock leakage); relative/absolute/regression targets + cross-sectional features (universe/sector-relative, per-date ranks)
- Cross-section objective: `cross_section.objective: regression` trains on continuous excess log-return (Huber loss); `classification` uses binary beat-median labels. Head stays linear Dense(20) - architecture unchanged. `compile_model(..., objective=)` switches the loss.
- `Source/Models/ensemble.py` — seed-ensemble (`training.n_seeds`): trains N models, averages predictions. Collapses GPU run-to-run nondeterminism + improves generalization. Both run scripts set `TF_DETERMINISTIC_OPS`/`TF_CUDNN_DETERMINISM` before importing TF.
- Universe is ~85 NSE names (price data only; no point-in-time fundamentals - would be look-ahead leakage). `Source/Ingestion/fetch_universe.py`.
- Pipeline is quiet (no per-epoch/headline prints; `verbose=0`); read results from the JSON artifacts, not stdout.
- `Source/Ingestion/fetch_universe.py` — NSE universe downloader (Data/Raw_Data/Universe/, gitignored)
- `Source/Ingestion/` — yfinance downloader
- `Source/News/build_sentiment.py` — NewsAPI + FinBERT sentiment, self-contained (parallel track, not fused; OFF by default)
- `Notebooks/Notebooks/Eda.ipynb` — original research notebook (source of truth for logic)
- `frontend/` — Next.js showcase site; `frontend/public/data/*.json` = generated artifacts
- `config.yaml` — all hyperparameters
- `Data/Raw_Data/^NSEI_daily.csv` — raw OHLCV

## Sentiment fusion (optional, off by default)
- `python -m Source.News.build_sentiment --query "..."` (needs `NEWSAPI_KEY`) writes `Data/Processed_Data/daily_sentiment.csv`.
- Set `features.use_sentiment: true` in config.yaml to append it as an extra feature. `resolve_feature_cols()` and the model derive feature count automatically.
- Kept OFF: NewsAPI only serves ~30 days, so historical sentiment can't be backfilled. Never train the historical backtest on fabricated/all-zero sentiment.

## Costs
- Per-side cost = `transaction_cost_bps` (fees) + `slippage_bps`, via `metrics.total_cost_bps(cfg)`. Charged round-trip on every trade.

## Env Vars (names only)
- `NEWSAPI_KEY` — only for the optional Source/News sentiment pipeline. Not needed for the model or site.

## Gotchas
- Raw yfinance CSV has 3 header rows (Price/Ticker/Date). `data_loader.load_ohlcv` strips them + drops the duplicate adj-close column.
- Attention pooling is `AttentionPooling1D`, a registered Keras Layer — NOT a Lambda. Keras cannot deserialize a Lambda wrapping a Python lambda, which breaks `model.save()` and makes optimizer state unsaveable. Its inner Dense is built in `build()`, otherwise weights silently fail to restore. Changing this invalidates existing `.weights.h5` files (re-run `scripts/save_paper_model.py`).
- Model outputs **raw logits** (`from_logits=True`); apply `tf.sigmoid` at inference. Logits are used directly as the alpha signal.
- Strategy metrics use **non-overlapping** 20-day returns (no overlap inflation) and charge transaction costs.
- No look-ahead: StandardScaler fit on train only; temporal split, no shuffling.
- Site is **static** — regenerate JSON via `run.py` then rebuild frontend. Do not hand-edit `frontend/public/data/*.json`.
- Results are honest research output; daily index direction is hard — expect near-coin-flip AUC and modest IC. Report as-is.

## Do Not
- Do not fabricate metrics. The site shows whatever `run.py` produces.
- Do not commit `Data/Raw_Data/*.csv`, model artifacts, or `frontend/node_modules`.
