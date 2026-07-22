# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose
Multi-Horizon Transformer (Nifty 50 direction). Deep-learning system predicting Nifty 50 (`^NSEI`) directional movement across 20 forward horizons (1–20 days) with a multi-output Transformer encoder. Research-grade. A static Next.js site showcases the real backtested results.

**The headline finding is negative and that is the deliverable:** mean OOS AUC ~0.51 against a 0.50 coin flip, 0/20 horizons significant after multiple-testing correction, and the paper book underperforms buy-and-hold. The contribution is the measurement apparatus (leakage audits, overlap-corrected error bars, deflated Sharpe, trial counting), not alpha. Never tune, retry, or reframe to make these numbers look better.

## Stack
- **Model/pipeline:** Python 3.10+, TensorFlow 2.21 (Keras), scikit-learn, pandas, numpy, scipy, PyYAML.
- **Data:** yfinance OHLCV for `^NSEI`, 2007→2026 (~4,600 raw rows; 4,343 after feature warmup drops).
- **Frontend:** Next.js 14 (App Router, static export), TypeScript, Tailwind, Recharts. No backend — reads precomputed JSON.
- **Deploy:** Vercel (static `out/`).

## Run / Build / Test
```bash
pip install -r requirements.txt   # 15 packages, derived from actual imports

# 1. (optional) refresh data
python -m Source.Ingestion.Fetch_Market_Data
python -m Source.Ingestion.fetch_macro           # VIX, USDINR, crude, S&P

# 2. train + full backtest -> writes frontend/public/data/*.json
python -m Source.Backtest.run                    # REUSE=1 rebuilds JSON from cache, no retrain

# 2b. cross-sectional track (~85 NSE large caps, real quantile L/S)
python -m Source.Ingestion.fetch_universe        # one-time data download
python -m Source.Backtest.run_cross_section      # -> cross_section.json

# 3. live layers (all read the FROZEN paper model; none of them train)
python -m Source.Paper.run --refresh             # -> paper_trading.json
python -m Source.Insights.build                  # -> predictions.json
python -m Source.Adaptive.run                    # -> adaptive.json (audit; --retrain to gate a challenger)

# 4. frontend
cd frontend && npm install && npm run dev        # local
npm run build                                    # static export -> frontend/out

# tests: leakage audits, cost/strategy/metric math, drift detectors,
# champion-challenger gate, artifact validation
python -m pytest tests/test_rigorous.py -q
python -m pytest tests/test_rigorous.py -q -k drift        # single test / subset
```
Everything is driven by `config.yaml` (hyperparams, windows, split, costs, adaptive cadences).

Artifact order matters: `Backtest.run` → `save_paper_model.py` → `Paper.run` → `Insights.build` → `Adaptive.run`. Later steps consume earlier outputs.

### GPU training (WSL2 + CUDA)
Native-Windows TF ≥2.11 is CPU-only. Train on the RTX 2050 through WSL2:
```bash
# one-time: CUDA-enabled TF venv inside WSL Ubuntu
wsl -d Ubuntu bash -lc 'python3 -m venv ~/venvs/mht && ~/venvs/mht/bin/pip install "tensorflow[and-cuda]==2.21.0" numpy pandas scipy scikit-learn pyyaml yfinance'

# run training on GPU (repo is on the Windows drive, visible at /mnt/c)
wsl -d Ubuntu bash -lc 'source ~/venvs/mht/bin/activate && cd "/mnt/c/Users/vivaa/OneDrive/Desktop/Personal Projects/Multi-Horizon-Transformer-for-Systematic-Equity-Direction-Forecasting" && python -m Source.Backtest.run_cross_section'
```
`Source/device.py` logs the device and, with `training.require_gpu: true` in config.yaml, aborts rather than silently falling back to CPU.

**WSL trap:** the pip GPU wheel does not put `nvidia-*-cu12` libs on `LD_LIBRARY_PATH`, so TF reports "Cannot dlopen some GPU libraries" and silently uses CPU. `scripts/wsl_gpu_env.sh` fixes it — source it before any training command. From Git Bash, call WSL with `MSYS_NO_PATHCONV=1` and an embedded `bash -c "source ...; cd ...; python -m ..."` (standalone /mnt paths get mangled).

## Architecture — the three things that require reading several files

**1. There are TWO model lineages. Never mix their numbers.**

| | Backtest model | Frozen paper model |
|---|---|---|
| Trained on | train split only | train + val |
| Built by | `Source/Backtest/run.py` | `scripts/save_paper_model.py` |
| Lives in | run cache (`Data/Processed_Data/run_cache_*.npz`) | `Data/Processed_Data/paper_model/` |
| Produces | `summary.json`, `horizons.json`, `strategies.json`, `calibration.json`, … | `paper_trading.json`, `predictions.json`, `adaptive.json` |
| Cutoff | the train/val boundary | `oos_cutoff` in its `meta.json`, frozen forever |

These are different fits with different weights. Attaching one's error bars to the other's predictions is a bug that already shipped once — `predictions.json` paired frozen-model probabilities with backtest-model AUCs, describing a predictor that was never measured. Anything reporting skill for the frozen model must compute it from that model's own logits over its own OOS period.

**2. Effective sample size is the binding constraint on every claim.**

An h-day forward label sampled daily overlaps its neighbour h-fold, so independent observations ≈ `n / h` — about **33 at h=20** on the test set. Consequences that recur throughout the codebase:
- AUC standard errors use effective n, never raw n (`suite._auc_se(..., overlap=h)`). Using raw n shrinks intervals ~4.5x and manufactures significance — this is the single easiest way to fake an edge here.
- Strategy returns are non-overlapping 20-day blocks, never overlapping daily.
- `Source/Adaptive` sizes each retraining layer by parameter count against the independent observations its cadence delivers.
- A realistic edge (AUC 0.52–0.55) is *below what this sample can resolve*; ~0.70 would be needed for single-horizon significance. Treat any large AUC as a leak to hunt, not a win.

**3. Dataflow is one-way: Python writes JSON, the site reads it.**

`Source/**` → `frontend/public/data/*.json` → typed static imports in `frontend/lib/data.ts` → `app/page.tsx`. There is no backend and no fetch at runtime (`Source/Api/main.py` is an optional read-only convenience, not part of the site). To change a number on the site, regenerate the artifact and rebuild — never hand-edit the JSON.

## Directory Map
- `Documentation/` — full project docs (overview, getting started, data + leakage rules, architecture diagrams, per-file reference, metric definitions, results, instrument choice, research gaps). Update these when behaviour changes; they are the onboarding path for anyone new.
- `Source/Paper/` — paper trading. `engine.py` = long/flat book marked daily (costs, equity, trades); `run.py --refresh` = fetch fresh data, score OOS days with the FROZEN model (`Data/Processed_Data/paper_model/`, from `scripts/save_paper_model.py`), write `frontend/public/data/paper_trading.json`. Daily cron: `.github/workflows/paper_trading.yml` (commits update -> Vercel auto-deploys). Frozen model = trained through validation cutoff only, so all later days are true OOS.
- `Source/Insights/build.py` — current forward predictions (`predictions.json`): scores the latest window with the frozen paper model, reports per-horizon calibrated P(up) beside its overlap-corrected AUC 95% CI and multiple-testing verdict. Skill is measured on the FROZEN model's own OOS period (everything after `oos_cutoff`), never read from `horizons.json` — that artifact belongs to the backtest model (a different fit), so its error bars do not describe this predictor. Also never reuse its `p_value`: that is a Spearman IC p-value over raw overlapping labels (i.i.d. assumption violated); test AUC vs 0.5 with the effective-n SE instead.
- `Source/Paper/frozen.py` — single loader/scorer for the frozen model, shared by `Paper/run.py` and `Insights/build.py` so the paper book and the predictions table can never drift apart. Loads for inference and never compiles (the saved optimizer is skipped deliberately, not half-restored).
- `Source/Adaptive/` — layered retraining, sized by PARAMETER COUNT vs independent observations (a week carries ~0.25 indep obs at h=20, so weekly gradient updates to the ~70k-param backbone would fit noise). `drift.py` = ADWIN + Page-Hinkley, monitor-only, 0 fitted params; `recalibrate.py` = ~40 Platt params on a trailing window whose end is embargoed by `horizons`; `retrain.py` = quarterly purged refit behind a champion/challenger gate; `versioning.py` = per-version `train_cutoff` + cumulative `trial_index` feeding deflated Sharpe. `run.py [--retrain]` -> `adaptive.json`. The gate FAILS CLOSED: it refuses when the eval block has < `min_effective_n` independent obs, requires the gain to exceed `promote_z` x SE (not just a fixed margin), and refuses if DSR is required but unevaluable.
- `Source/Intraday/` — hourly track. `fetch.py` = source-agnostic bar ingestion (`SOURCES` dict; yfinance verified at 5,057 hourly bars/730d, `--source csv` is the drop-in for a broker or NSE feed); `features.py` = intraday-only inputs (overnight gap, session position, VWAP deviation, volume vs SAME time-of-day not a flat mean - intraday volume is U-shaped); `run.py` = train+evaluate with the same encoder, split and error bars. **Frequency and horizon must move together**: hourly bars at a 20-DAY horizon give ~5 independent obs (worse than daily's 32); hourly at 20-BAR gives ~37. Result so far: mean AUC 0.5031, 0/20 significant - same finding as daily, because 730d of history (Yahoo's 1h cap) is the real constraint, not bar size.
- `Source/News/gdelt.py` — historical news tone, free, NO API key, history to 2017. This is what makes sentiment fusable at all; NewsAPI's ~30 days never could be. Two traps handled: GDELT rate-limits hard (429 -> backoff), and an UNPARENTHESISED OR query returns HTTP 200 with a text error body, so a malformed query looks like an empty success. `attach_sentiment` merges asof-backward THEN shifts, so a bar never sees news published inside its own interval.
- `Source/Journal/` — trade journal + P&L attribution (`journal.json`). `attribution.py` decomposes each closed trade into win / signal_error / cost_drag / **noise**, where noise = |move| below one holding-period sigma; sub-noise losses are NOT mistakes and must never be "learned" from. Hit rate is reported with an EXACT binomial p-value (normal approx is wrong at n~13). `bandit.py` = Thompson sampling over FIXED validated rules - RL sized to the data (policy learning needs ~1e6 decisions; this book has made ~13). Always read `separation.verdict` before the arm means: argmax names a winner whether or not one exists.
- `Source/Advisor/` — OPTIONAL LLM commentary, OFF by default; with no API key the journal writes deterministic template text, so CI needs no secret. `client.py` is provider-abstracted (groq | anthropic) with input screening, schema validation + one retry, backoff on 429/5xx, explicit max_tokens, usage logging; `prompts.py` holds versioned prompts. **The guardrail is structural, not textual:** `Journal/run.py` filters the payload so the model only ever sees realised history and can never emit a trade call. Never pass it a forward prediction.
- `Source/Evaluation/` — metric suite (classification/error/financial/statistical, overlap-corrected AUC SE, deflated Sharpe, multiple-testing) + per-model registry (`Data/Evaluation/`). `scripts/evaluate_models.py`, `scripts/conviction_strategy.py`, `scripts/gbdt_baseline.py`.
- `Source/Risk/sizing.py` — vol-targeted position sizing (lagged trailing vol, no look-ahead); run.py emits a `risk_targeted` strategy variant. Config `risk:`.
- `Source/Api/main.py` — read-only FastAPI over the artifacts (`uvicorn Source.Api.main:app`). Never trains.
- `Source/Ingestion/session.py` — shared market session. (a) **NSE trading calendar** with the published holiday list: nothing else in the repo knew about holidays, so `market_open` was wrong on Diwali and any trading-day count was inflated (October 2026 is 18 days, not the 22 a weekday count gives). `calendar_covers()` reports when a range falls outside the known years rather than silently degrading to weekends-only. (b) **`download()`** — yfinance on a curl_cffi Chrome-impersonating session with retries on a FRESH client per attempt, because Yahoo answers a throttled request with an EMPTY body and yfinance caches that emptiness on the client, so reusing one re-reads the same nothing. It RAISES rather than returning empty: a silent empty frame becomes a silent gap in a training set. **Deliberately not ported from the source module: its sample-data fallback.** Serving embedded sample prices when the network fails is good UX in a demo app and a correctness disaster here - a backtest would train on fabricated prices and report the metrics as real.
- `Source/Ingestion/nse.py` — authoritative NSE India fundamentals (P/E, sector P/E, market cap, 52w range, annualised vol, and `delivery_pct` which yfinance does NOT expose - the share of volume actually settled vs squared off intraday, a real conviction/microstructure signal). Serves only TODAY's values, so like fetch_fundamentals these are look-ahead leakage if applied backwards. The point of the module is `append_archive`: it APPENDS a dated snapshot per run, building a genuine point-in-time panel going forward and finally removing the "no free point-in-time NSE fundamentals" blocker. `archive_span()['usable_as_features']` stays False until >=250 distinct dates - do not use these as model features before that. Needs `curl_cffi` (chrome impersonation); NSE blocks datacenter IPs so it fails on CI and works from an Indian connection. Idempotent per day; never rewrites a previous day's row.
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
- `Source/Ingestion/fetch_universe.py` — NSE universe downloader (~85 names -> `Data/Raw_Data/Universe/`, gitignored). Price data ONLY; no point-in-time fundamentals, which would be look-ahead leakage. Writes a simple single-header CSV, unlike the multi-header `^NSEI` export.
- Pipeline is quiet (no per-epoch/headline prints; `verbose=0`); read results from the JSON artifacts, not stdout.
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
- `metrics.total_cost_bps(cfg)` is the single entry point. With `backtest.cost_model: india` (the active setting) it dispatches to `Source/Backtest/costs.py::india_cost_breakdown`, the itemized stack: brokerage, exchange txn, SEBI, STT (sell-side for futures/intraday), stamp duty (buy-side), 18% GST, slippage, DP charges on delivery sells. The flat `transaction_cost_bps + slippage_bps` path is legacy fallback only.
- Round-trip by instrument: **futures 9.58bps** (what the index track uses), intraday 10.47, options 25.53, delivery 28.22. Charged round-trip on every trade.

## Env Vars (names only)
- `GROQ_API_KEY` — optional LLM commentary (`Source/Advisor/`, provider `groq` by default). Read from the env or `.env.local`. Without it the journal writes deterministic template text, so nothing breaks.
- `ANTHROPIC_API_KEY` — same layer, only if `advisor.provider: anthropic`.
- `NEWSAPI_KEY` — only for the optional Source/News sentiment pipeline. Not needed for the model or site.
- Never read, print or echo the values. `.env*` is gitignored by three separate rules; keep it that way.

## Gotchas
- Raw yfinance CSV has 3 header rows (Price/Ticker/Date). `data_loader.load_ohlcv` strips them + drops the duplicate adj-close column.
- Attention pooling is `AttentionPooling1D`, a registered Keras Layer — NOT a Lambda. Keras cannot deserialize a Lambda wrapping a Python lambda, which breaks `model.save()` and makes optimizer state unsaveable. Its inner Dense is built in `build()`, otherwise weights silently fail to restore. Changing this invalidates existing `.weights.h5` files (re-run `scripts/save_paper_model.py`).
- Model outputs **raw logits** (`from_logits=True`); apply `tf.sigmoid` at inference. Logits are used directly as the alpha signal.
- Strategy metrics use **non-overlapping** 20-day returns (no overlap inflation) and charge transaction costs.
- No look-ahead: StandardScaler fit on train only; temporal split, no shuffling.
- Site is **static** — regenerate JSON via `run.py` then rebuild frontend. Do not hand-edit `frontend/public/data/*.json`.
- Results are honest research output; daily index direction is hard — expect near-coin-flip AUC and modest IC. Report as-is.

## Do Not
- Do not fabricate metrics. The site shows whatever the pipeline produces.
- Do not hardcode a metric into frontend prose. The daily cron regenerates the artifacts, so a hardcoded figure silently drifts out of sync with the table beside it — derive it from the artifact instead.
- Do not retry/loop a model until a metric clears a threshold. That manufactures a false positive rather than finding an edge, and every attempt must be counted as a trial in `deflated_sharpe(n_trials=...)`.
- Do not commit `Data/Raw_Data/*.csv`, model artifacts, or `frontend/node_modules`.
