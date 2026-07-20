# 5. File Reference

Every file in the repository, what it does, and how to run it.

## Repository layout

```
.
├── config.yaml                 # ALL hyperparameters. Nothing is hardcoded elsewhere.
├── Source/                     # Library code
│   ├── device.py               # GPU/CPU selection, fail-fast on CPU
│   ├── Ingestion/              # Data downloaders
│   ├── Features/               # Feature engineering
│   ├── Pipeline/               # Loading, windowing, splitting, scaling
│   ├── Models/                 # Transformer + seed ensemble
│   ├── Backtest/               # Costs, metrics, orchestrators
│   ├── Evaluation/             # Metric suite + per-model registry
│   ├── Paper/                  # Live forward paper trading
│   ├── Insights/               # Current predictions artifact
│   ├── Risk/                   # Volatility-targeted sizing
│   ├── News/                   # Optional sentiment track
│   └── Api/                    # Read-only FastAPI over artifacts
├── scripts/                    # Runnable entry points
├── tests/test_rigorous.py      # 32 tests
├── Data/                       # Raw + processed data
├── frontend/                   # Next.js static showcase site
└── Documentation/              # You are here
```

---

## `Source/Ingestion/` — data acquisition

| File | Lines | Purpose |
|------|-------|---------|
| `Fetch_Market_Data.py` | 33 | Downloads `^NSEI` daily OHLCV from yfinance, to the ticker/path set in `config.yaml`. |
| `fetch_macro.py` | 52 | Downloads India VIX, USDINR, crude, S&P 500 → `Data/Raw_Data/Macro/`. |
| `fetch_universe.py` | 42 | Downloads 85 NSE large caps → `Data/Raw_Data/Universe/` (gitignored). Writes a **simple single-header CSV**, unlike the `^NSEI` export. |
| `fetch_fundamentals.py` | 55 | **Current** fundamentals snapshot. Display only — never a model feature (would be look-ahead leakage). |

```bash
python -m Source.Ingestion.Fetch_Market_Data
python -m Source.Ingestion.fetch_macro
python -m Source.Ingestion.fetch_universe
```

## `Source/Features/` — feature engineering

| File | Lines | Produces |
|------|-------|----------|
| `Returns.py` | 47 | `daily_ret`, `roll_mean_ret_{5,10,20}`, `momentum_10`, `ma_diff_10` |
| `Volatility.py` | 53 | `roll_vol_{5,10,20}`, `log_volume`, `vol_roll_mean_5` |
| `Macro.py` | 96 | The 8 macro features. **Enforces the strict lag** via `_asof_lagged`. |

Library modules — imported, not run directly.

## `Source/Pipeline/` — data to tensors

| File | Lines | Key functions |
|------|-------|---------------|
| `data_loader.py` | 53 | `load_ohlcv` — strips the 3 header rows and duplicate adj-close |
| `dataset.py` | 153 | `build_features`, `make_windows`, `temporal_split_and_scale`, `build_dataset`, `resolve_feature_cols` |
| `cross_section.py` | 337 | `load_universe`, panel builder, date-based split, cross-sectional features, `latest_windows` |

`resolve_feature_cols(cfg)` derives the active feature list from config and
**raises if a column appears twice** (macro listed under both `use_macro` and
`xs_features` was a real bug — duplicate features silently double-weight a
signal). Covered by a test.

## `Source/Models/` — the network

| File | Lines | Contents |
|------|-------|----------|
| `transformer.py` | 127 | `AttentionPooling1D`, `positional_encoding`, `_encoder_block`, `build_model`, `compile_model` |
| `ensemble.py` | 49 | `train_ensemble` — trains `n_seeds` models, averages predictions |

`build_model(cfg, num_features=None)` returns `(model, attention_model)` sharing
weights; the second exposes block-2 attention scores for interpretability.
`num_features` overrides input width for the wider cross-sectional panel.

`compile_model(model, cfg, objective=)` switches loss: BCE-on-logits for
`classification`, Huber for `regression`. Architecture is identical either way.

## `Source/Backtest/` — costs, metrics, orchestration

| File | Lines | Contents |
|------|-------|----------|
| `costs.py` | 105 | India cost model. `INSTRUMENTS` dict, `india_cost_breakdown`, `total_cost_bps` |
| `metrics.py` | 307 | Sharpe, drawdown, IC, decile attribution, `calibrate_probs`, `per_horizon_*` |
| `run.py` | 453 | Index-track orchestrator. Trains, backtests, writes ~15 JSON artifacts |
| `run_cross_section.py` | 296 | Cross-sectional orchestrator. Panel train + quantile spread |

```bash
python -m Source.Backtest.run                 # REUSE=1 to skip retraining
python -m Source.Backtest.run_cross_section
```

### Cost model

Per-side cost = fees + slippage, charged **round-trip on every trade**.

| Instrument | Round-trip (bps) |
|------------|------------------|
| Futures | 9.58 |
| Intraday equity | 10.47 |
| Options | 25.53 |
| Delivery equity | 28.22 |

Components: brokerage, exchange transaction charge, SEBI turnover fee, STT
(sell-side only for futures/intraday), stamp duty (buy-side only), GST at 18% on
brokerage plus exchange plus SEBI, slippage, and DP charges on delivery sells.
The index track uses **futures**, the cheapest and the realistic instrument for
trading an index.

> A 10x error once lived here: futures stamp duty was 2.0 bps instead of 0.2 bps
> (0.002%). It is now covered by a test asserting the component sum.

## `Source/Evaluation/` — the measurement apparatus

| File | Lines | Contents |
|------|-------|----------|
| `suite.py` | 233 | `classification_metrics`, `_auc_se`, `auc_pvalue`, `error_metrics`, `financial_metrics`, `diebold_mariano`, `friedman_test`, `deflated_sharpe`, `multiple_testing` (returns per-hypothesis reject flags) |
| `registry.py` | 63 | Per-model JSON registry in `Data/Evaluation/`, `leaderboard()` ranked by deflated Sharpe |

See [Evaluation](06-evaluation.md) for every formula.

## `Source/Paper/` — live forward paper trading

| File | Lines | Contents |
|------|-------|----------|
| `engine.py` | 95 | Long/flat book marked daily. `new_state`, `step`, `summary`. Pure dict state, idempotent per date |
| `frozen.py` | 65 | **Shared** loader/scorer for the frozen model. Used by both `run.py` and `Insights/build.py` so they can never drift apart |
| `run.py` | 154 | Scores post-cutoff days with the frozen model, steps the book, writes `paper_trading.json` |

```bash
python -m Source.Paper.run              # score with current data
python -m Source.Paper.run --refresh    # re-download data first (what the cron does)
```

The frozen model was trained only through the validation cutoff, and that cutoff
is **stored in the model metadata**. Paper trading therefore can never trade an
in-sample day, and the curve's left edge cannot drift as data grows.

Daily cron: `.github/workflows/paper_trading.yml` (weekdays 11:30 UTC) refetches
data, steps the book, rebuilds predictions, and commits. The push triggers the
Vercel deploy. The commit step runs with `if: always()` so a failure in the
predictions step cannot discard a computed paper book.

## `Source/Insights/build.py` — current predictions

216 lines. Scores the latest window and writes `predictions.json`: per-horizon
calibrated P(up) beside that horizon's out-of-sample AUC, overlap-corrected 95%
CI, effective n, and multiple-testing verdict.

```bash
python -m Source.Insights.build
```

**Two rules encoded here:**

1. Skill is measured on the **frozen model's own** OOS period, never read from
   `horizons.json` — that artifact belongs to the backtest model, a different
   fit, so its error bars would describe a predictor that was never measured.
2. Never reuse `horizons.json`'s `p_value` for significance. That is a Spearman
   IC p-value over raw overlapping labels. Test AUC against 0.5 with the
   effective-n standard error instead.

## `Source/Risk/sizing.py`

52 lines. Volatility-targeted position sizing using **lagged** trailing
volatility (no look-ahead). `run.py` emits a `risk_targeted` strategy variant.
Config: `risk.target_vol_annual: 0.15`, `lookback: 6`, `max_leverage: 2.0`.

## `Source/Api/main.py`

80 lines. Read-only FastAPI over the generated artifacts. **Never trains.**

```bash
uvicorn Source.Api.main:app --reload
```

## `Source/News/` — optional sentiment track

| File | Lines | Purpose |
|------|-------|---------|
| `build_sentiment.py` | 90 | Self-contained: NewsAPI fetch → FinBERT scoring → daily aggregate in `daily_sentiment.csv` |

Off by default — see [Data](03-data.md#sentiment-optional-off).

## `scripts/` — entry points

| Script | Lines | Purpose |
|--------|-------|---------|
| `select_model.py` | 87 | Hyperparameter grid on **validation only** |
| `save_paper_model.py` | 110 | Trains and FREEZES the 3-seed paper model, stores scaler, signal stats, Platt coefficients, and the OOS cutoff |
| `evaluate_models.py` | 134 | Runs the full metric suite over every model, writes the registry |
| `conviction_strategy.py` | 145 | Trades only when the model is confident — high-conviction gating |
| `gbdt_baseline.py` | 108 | LightGBM baseline, purged/embargoed CV |
| `gbdt_cross_section.py` | 90 | LightGBM on the panel — the tool the literature favours |
| `wsl_gpu_env.sh` | — | Fixes `LD_LIBRARY_PATH` for CUDA in WSL |
| `train_gpu.sh` | — | Wrapper: source env, run a module on GPU |

## `tests/test_rigorous.py`

533 lines, 32 tests. Groups:

| Group | Covers |
|-------|--------|
| Feature correctness | Formula-by-formula verification |
| Leakage audits | Forward targets, no-lookahead windows, macro strict lag, disjoint splits, scaler-on-train-only |
| Cross-section | Date-split no-leakage, no NaN, no duplicate features |
| Cost math | India component sum, round-trip application |
| Metric math | Sharpe, drawdown, non-overlap, net = gross − cost |
| Statistical | Overlap-corrected SE, deflated Sharpe, multiple-testing reject flags including ties |
| Calibration | Platt rank-preservation |
| Paper engine | Cost/marking correctness, flat-earns-nothing |
| Serialization | Full save/load round-trip **with optimizer state** |
| Artifacts | Schema and range validity of shipped JSON |

```bash
python -m pytest tests/test_rigorous.py -q
```

## `frontend/`

| Path | Purpose |
|------|---------|
| `app/page.tsx` | The entire single-page site |
| `components/charts.tsx` | All Recharts visualisations + `PredictionTable` |
| `components/ui.tsx` | `Section`, `Panel`, `Stat` primitives |
| `lib/data.ts` | Typed static imports of every artifact |
| `public/data/*.json` | **Generated artifacts — never hand-edit** |

Static export (`output: export`), no backend. Regenerate JSON with the Python
pipeline, then rebuild.

## Generated artifacts

| File | Written by | Contents |
|------|-----------|----------|
| `summary.json` | `Backtest/run.py` | Headline metrics, split sizes, costs |
| `horizons.json` | `Backtest/run.py` | Per-horizon AUC, accuracy, IC (backtest model) |
| `strategies.json` | `Backtest/run.py` | Every strategy variant with equity curves |
| `calibration.json` | `Backtest/run.py` | Reliability bins, pre and post Platt |
| `decile.json`, `threshold_sweep.json`, `yearly.json`, `walkforward.json`, `attention.json`, `training.json`, `price.json`, `features.json` | `Backtest/run.py` | Supporting charts |
| `cross_section*.json` | `run_cross_section.py` | Four published configs, none cherry-picked |
| `stock_signals.json` | `run_cross_section.py` | Per-stock forward signal + risk profiles |
| `paper_trading.json` | `Paper/run.py` | Live paper book |
| `predictions.json` | `Insights/build.py` | Current forward predictions with error bars |

Continue to [Evaluation](06-evaluation.md).
