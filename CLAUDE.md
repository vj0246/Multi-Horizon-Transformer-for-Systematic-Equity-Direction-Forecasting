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
python Source/Ingestion/Fetch_Market_Data.py

# 2. train + full backtest -> writes frontend/public/data/*.json
python -m Source.Backtest.run

# 3. frontend
cd frontend && npm install && npm run dev      # local
npm run build                                   # static export -> frontend/out
```
Everything is driven by `config.yaml` (hyperparams, windows, split, costs).

## Directory Map
- `Source/Features/` — Returns.py, Volatility.py (feature engineering)
- `Source/Pipeline/` — data_loader.py (clean CSV), dataset.py (features→targets→windows→split→scale)
- `Source/Models/transformer.py` — TF Transformer (2 blocks, 4 heads, d_model=64, attention pooling) + attention model
- `Source/Backtest/metrics.py` — Sharpe/drawdown/IC/decile/costs
- `Source/Backtest/run.py` — orchestrator, exports JSON artifacts
- `Source/Ingestion/` — yfinance downloader
- `Source/News/` — NewsAPI + FinBERT sentiment (parallel track, not fused)
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
- Model outputs **raw logits** (`from_logits=True`); apply `tf.sigmoid` at inference. Logits are used directly as the alpha signal.
- Strategy metrics use **non-overlapping** 20-day returns (no overlap inflation) and charge transaction costs.
- No look-ahead: StandardScaler fit on train only; temporal split, no shuffling.
- Site is **static** — regenerate JSON via `run.py` then rebuild frontend. Do not hand-edit `frontend/public/data/*.json`.
- Results are honest research output; daily index direction is hard — expect near-coin-flip AUC and modest IC. Report as-is.

## Do Not
- Do not fabricate metrics. The site shows whatever `run.py` produces.
- Do not commit `Data/Raw_Data/*.csv`, model artifacts, or `frontend/node_modules`.
