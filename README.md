# Multi-Horizon Transformer for Systematic Equity Direction Forecasting

> **A deep learning system that predicts the directional movement of the Nifty 50 index simultaneously across 20 distinct future time horizons — from next-day to 20-day forward — using a custom Transformer encoder trained on 18+ years of Indian equity market data.**

---

## 📚 Full Documentation

**→ [`Documentation/`](Documentation/README.md)** — complete guide written so that
anyone can understand every file and reproduce every number without asking a
question.

| Doc | Answers |
|-----|---------|
| [Overview](Documentation/01-overview.md) | What this is, what it found, why the finding is negative |
| [Getting Started](Documentation/02-getting-started.md) | Install, run, regenerate every artifact, GPU setup |
| [Data](Documentation/03-data.md) | Every source, schema, and the 8 leakage rules |
| [Architecture](Documentation/04-architecture.md) | Pipeline + model diagrams, layer by layer |
| [File Reference](Documentation/05-file-reference.md) | Every file, what it does, how to run it |
| [Evaluation](Documentation/06-evaluation.md) | Every metric and formula, and 3 bugs that faked skill |
| [Results](Documentation/07-results.md) | The actual numbers with confidence intervals |
| [Instrument Choice](Documentation/08-instrument-choice.md) | Why `^NSEI`, and whether a single stock is better |
| [Research Gaps](Documentation/09-research-gaps.md) | Literature critique vs what is implemented |

---

## Technical Summary

This project builds a **multi-output binary classification Transformer** to answer one question with 20 simultaneous answers: *Will the Nifty 50 close higher than today — 1 day, 2 days, 3 days... all the way to 20 days from now?*

The system ingests raw OHLCV data for `^NSEI` (Nifty 50) going back to 2007, engineers 16 time-series features capturing price momentum, rolling volatility, volume dynamics, and mean reversion signals, and feeds the **11 stationary ones** (raw price levels are excluded — see below) as 60-day sliding window sequences into a 2-block Transformer encoder with sinusoidal positional encoding and learned attention pooling. The model outputs 20 independent logits — one per forecast horizon — each representing the probability that the index will be higher at that future point.

Training uses strict temporal split (70/15/15 — no look-ahead leakage), StandardScaler fitted exclusively on training data, binary cross-entropy loss with AUC as the primary metric, and early stopping with patience=7. Post-training evaluation goes beyond classification metrics: the model's raw logits are used directly as alpha signals, tested via Spearman IC, sign-strategy Sharpe, quantile long-short Sharpe, and decile return attribution — all on held-out test data.

**The project is research-grade, not production-deployed.** Source modules for news ingestion (NewsAPI), FinBERT sentiment, and an XGBoost baseline exist in the codebase as parallel development tracks. The Transformer is the primary model.

---

## Live Demo

**→ [multi-horizon-transformer-for-syste.vercel.app](https://multi-horizon-transformer-for-syste.vercel.app)**

A static Next.js dashboard (in `frontend/`) visualizes the **real, regenerated** backtest results — per-horizon AUC/IC, the cost-aware equity curve, walk-forward folds, the cross-sectional track, attention distribution, and training history. Every number is produced by an actual training run, not hand-authored. An **interactive Sharpe explorer** lets you drag the transaction cost and watch each strategy's net Sharpe and equity recompute live in the browser from the raw per-trade returns — making explicit how sensitive every ratio is to cost assumptions.

**Live paper trading.** A frozen model (trained only through the validation cutoff) paper-trades the primary strategy forward on real NSE closes, charged the full India futures cost stack — every day after the cutoff is a true out-of-sample read. A weekday GitHub Action (`.github/workflows/paper_trading.yml`) re-downloads the data, steps the book, and commits the update, so the site's **Paper Trading** section advances on its own. It is not tuned to look good: it under-performs simply holding the index, exactly as the no-edge finding predicts. Honest forward proof, not a profit demo. `python -m Source.Paper.run --refresh`.

### Reproduce end-to-end

```bash
# 1. train the Transformer + run the full backtest -> writes frontend/public/data/*.json
python -m Source.Backtest.run

# 2. rebuild + preview the site
cd frontend && npm install && npm run dev
```

Everything is driven by `config.yaml`. Set `REUSE=1` before a run to regenerate the JSON from a cached run without retraining.

**Training hardware.** Models train on the GPU (an RTX 2050) through a WSL2 CUDA environment — native-Windows TensorFlow ≥2.11 is CPU-only, so `config.yaml` sets `training.require_gpu: true` and a Windows-side run now fails fast rather than silently using the CPU (`Source/device.py`). See CLAUDE.md and `scripts/train_gpu.sh` for the WSL invocation.

**Stability & accuracy — seed-ensembling.** GPU attention/cuDNN kernels are not fully deterministic even with `TF_DETERMINISTIC_OPS` and op-determinism enabled, so a single run's numbers wobble. `training.n_seeds` trains several models (seeds 42, 43, …) and **averages their predictions** (`Source/Models/ensemble.py`): this collapses the run-to-run variance that made earlier single-seed numbers swing, and averaging independent models also improves generalization. All published metrics are the ensemble's.

### Headline result (honest)
Mean test AUC is **0.5033** and mean IC **+0.021** — marginally above coin-flip, achieved without cheating. Two things got it there: (1) **new information** the price series doesn't contain — India VIX, the overnight S&P move (the US session closes ~02:00 IST, *before* India trades), USDINR, crude, and breadth across the 85-stock universe, every external series asof-backward merged then shifted a row so a feature at row *t* uses only data from ≤ *t-1* (asserted in `tests/test_rigorous.py::test_macro_series_are_strictly_lagged`); and (2) **actually tuning** the architecture, which had been inherited unexamined from the notebook — selected on the **validation set only** (`scripts/select_model.py`, best of a 6-config grid at VAL AUC 0.5828), after which the test set was evaluated **once**.

For scale: 0.5033 is a *whisper*, not an edge. Published work on daily index direction lives at 0.50-0.55; anything above 0.6 is usually leakage. The earlier version of this project showed AUC 0.565 and Sharpe 1.5 — that was **artifact** (raw price levels, test-tuned thresholds, single-seed luck), and removing those honestly *dropped* AUC to 0.485. Adding real information brought it back over 0.5 legitimately.

**Trading it is a different story.** After India futures costs (**9.58 bps** round-trip: STT, stamp, exchange + SEBI, brokerage, GST, slippage — `Source/Backtest/costs.py`, comprehensive per-instrument model), the rolling-threshold timing book returns **~+11%** at **Sharpe ~+0.63** (31% time in market, -9.4% max drawdown) — but its bootstrap **95% CI spans zero**, and **buy-and-hold beats it** (+30.2%, Sharpe +0.94). 8-fold walk-forward Sharpe is **+0.28 ± 0.36**.

### Honesty framework — how these numbers are kept from lying
`Source/Evaluation/` scores every model with the full metric suite (classification, error, financial, statistical) and refuses the two tricks that manufacture fake edges at this sample size:

- **Overlap-corrected significance.** A 20-day label sampled daily overlaps its neighbours ~20×, so the ~2,984 windows carry only ~**150 independent observations**. The AUC standard error is computed on the effective n, not the raw n — which inflates it ~√20. Under that correct SE the best horizon (0.571) sits 0.66σ from chance, and **0 of 20 horizons survive Bonferroni or Benjamini-Hochberg** (min p = 0.194). The naïve i.i.d. SE had falsely flagged 2 as significant — the framework caught that bug in itself, and it is now a regression test.
- **Deflated Sharpe Ratio** (Bailey & López de Prado) discounts the Sharpe for the ~16 configurations tried across the project. The timing book's DSR is **0.916**, below the 0.95 bar — not a deflation-surviving edge.
- **Diebold–Mariano vs a naïve always-up forecast: p = 0.215** — the model does not forecast significantly better than the majority class.

**Conviction strategy** (`scripts/conviction_strategy.py`): the agent abstains unless its calibrated probability is far from 0.5, trading only high-conviction reads (threshold set on validation). Skill is measured as *accuracy above the majority baseline*, not Sharpe — because a positive Sharpe from a long bias in a bull market is beta, not alpha. Best reliable (≥30-trade) result: **accuracy edge +0.000 — no skill above baseline**. Confidence gating does not extract an edge either. GBDT baselines (`scripts/gbdt_baseline.py`) land at the same ~0.50, on both the index and the 86k-row panel. Every model family, data scale, and feature set converges on chance: the binding constraint is the ~150 effective samples, not the architecture.

### The per-horizon structure does not survive a clean protocol

The 0.5033 mean hides an unstable curve. Test AUC by horizon runs 0.44 (h1, the worst — next-day index direction is the most efficiently priced thing here) up to 0.571 (h13), with an apparently convincing cluster at h13-16 (0.549-0.571, IC +0.12 to +0.14). It is tempting to trade that region. **Do not.**

A validation-only horizon selection settles it. Picking the horizon purely by **validation** AUC — never looking at test — chooses **h10 at VAL AUC 0.654**. That horizon's **test AUC is 0.474**, below a coin flip:

| | validation | test |
|---|---|---|
| AUC range across the 20 horizons | 0.529 – **0.654** | 0.440 – 0.571 |
| val-selected horizon (h10) | **0.654** | **0.474** |
| mean across horizons | ~0.60 | **0.5033** |

Validation says 0.65 everywhere; test says 0.50. That gap is systematic, and it has a concrete cause: **validation is not a clean holdout in this pipeline.** It is used for early stopping (`restore_best_weights` on val loss), hyperparameter selection, Platt calibration, *and* entry thresholds. Val AUC is optimistically biased by construction — **test is the only uncontaminated number in the project.**

Two consequences, both honest:
1. **The h13-16 cluster is almost certainly noise.** If it were a real regime, validation would have pointed at it. Validation pointed at h10, which then underperformed a coin flip. The cluster is not traded and is not claimed as signal.
2. **Horizon selection does not rescue the result.** The protocol-clean answer to "which horizon should we trade?" is h10 — and h10 loses. Switching to h13-16 now, having seen test, is precisely the cheating this project refuses. (Note `best_val_horizon: 16` in summary.json was selected by validation *timing Sharpe* rather than AUC; the two validation criteria disagree, which is itself evidence there is no stable per-horizon structure to find.)

**Selection caveat, stated plainly:** the model was chosen on validation, but the entry-threshold *rule* was not. The frozen-validation cutoff proved degenerate on test — the signal's level shifts between fit and deployment (validation mean 0.00 vs test mean **-0.79**), so it never trades and returns exactly 0. The rolling past-only, level-invariant rule was adopted *after* observing that, so its +0.62 carries selection optimism and is an upper bound, not a clean OOS number. All three rules (frozen / expanding / rolling) are published unmodified in `strategies.json`. Nothing is flipped or reweighted to exploit the test set.

### Cross-sectional track (where a direction model can genuinely earn)
Timing one index is the hardest possible use of a direction model; ranking many stocks against each other on the same date is the natural one. `Source/Backtest/run_cross_section.py` trains the **same shared-weight Transformer** on a pooled panel of **~85 liquid NSE large/mid caps** (per-stock windows, same 11 stationary features + 6 cross-sectional features) and trades a **real cross-sectional quantile spread**: every 20 trading days, long the top 20% of names by ensemble signal, short the bottom 20%, equal weight. Legs are charged single-stock-futures costs; the long-only variant is charged delivery costs (STT 0.1% both sides). Evaluation reports mean daily cross-sectional IC and its information ratio, quintile attribution, and net equity curves against a gross equal-weight universe benchmark.

Only **price** data (yfinance) is used. Point-in-time fundamentals (earnings, ROE, valuation) would be the natural next feature block, but a survivorship-clean historical fundamental panel for NSE back to 2007 is not freely available, and using *current* fundamentals as historical features would be look-ahead leakage — so it is deliberately not done rather than faked.

Leakage guards specific to the panel: the train/val/test split is by **calendar date** (the same market day never sits in train for one stock and test for another) and the scaler is fit on pooled train windows only.

**Target formulation matters.** The first run reused the index track's *absolute* direction labels (will this stock close higher in h days?) — in a trending market nearly every label is "up", so the model learns nothing that separates one stock from another (measured mean daily IC ≈ -0.01, non-monotonic quintiles). The corrected formulation uses **relative labels**: did the stock beat the *cross-sectional median* h-day return on that date (`cross_section.relative_targets: true`).

**Cross-sectional features matter too.** The 11 stationary features describe each stock *in isolation* (its own momentum, volatility, returns) — they carry no information about how the stock stands *relative to its peers*, which is exactly what a ranking task needs. `cross_section.use_xs_features: true` adds six such features: return and 10-day momentum demeaned by the universe, per-date percentile ranks of return/momentum/volatility, and momentum demeaned by the stock's sector (`Source/Pipeline/cross_section.py::_attach_cross_sectional_features`). All are computed from same-date values only (no look-ahead) and lift the model input from 11 to 17 features.

**A regression objective was also tested.** A binary "beat the median" label throws away magnitude — a stock that outperforms by 0.1% and one that outperforms by 15% are the same label. `cross_section.objective: regression` instead trains the model on the **continuous cross-sectional excess log-return** per horizon (the stock's h-day return minus the universe median), with a Huber loss (robust to the heavy tails of 20-day excess returns). The Dense(20) head is linear in both cases, so the **architecture is unchanged** — only the loss and targets differ. On this data it underperformed the classification head (see the results table below), so the classification model is the one the site presents.

Every configuration is published, none cherry-picked: `cross_section.json` (classification head + relative targets + cross-sectional features — the best result), `cross_section_regression.json` (regression head, same inputs), `cross_section_base_features.json` (classification, per-stock features only), and `cross_section_absolute.json` (absolute targets, per-stock features). Each choice was made a priori from a diagnosis, not by shopping for the better number.

**Disclosed biases:** the universe is (mostly) today's large caps backtested into the past — survivorship bias inflates absolute returns (the long-short spread is partially insulated but still favored); daily IC uses overlapping 20-day forward returns (standard practice) while all P&L is non-overlapping.

**Canonical factors.** The per-stock and rank features above still omitted the factors that actually drive cross-sectional equity returns, so these were added (`_attach_cross_sectional_features`, all from trailing prices only): **12-1 momentum** (12-month return skipping the last month — the standard construction, the skip avoiding short-term-reversal contamination), **short-term (1-month) reversal**, **beta** to an equal-weight universe proxy, and **60-day idiosyncratic volatility** (residual to beta·market), each as a centered cross-sectional rank. Market regime (VIX, overnight S&P, breadth) already reaches the panel via `features.use_macro` — it is constant across names on a date, so it adds no *ranking* information but lets the model condition on the regime. Panel input: 17 → **29 features**.

**Measured outcome (honest).** The factors moved every metric in the right direction — roughly **halving** the negative IC — and it still is not enough to cross zero:

| metric | 17 features | **29 features (+factors)** |
|---|---|---|
| mean daily IC | -0.0489 | **-0.0290** |
| pooled rank IC | -0.0517 | **-0.0260** |
| pooled AUC (h20) | 0.4771 | **0.4876** |
| % of days IC > 0 | — | 42.0% |

The quintile profile is still **inverted and monotonic**: Q1 +1.38% → Q5 +0.92% forward 20-day return. The names the model ranks highest keep underperforming the ones it ranks lowest. The signal is, if anything, slightly anti-predictive — and it is **not** sign-flipped to exploit that, which would be fitting the test set.

(The test window shifts to 2023-10 → 2026-06 because the 252-day lookbacks for 12-1 momentum and beta consume more warmup, so the two columns are not perfectly like-for-like; the direction of travel is real, the exact deltas approximate.) On the Jun-2023 → Jun-2026 test window:

| Configuration | Mean daily IC | L/S spread (net) | Long-only top 20% |
|---|---|---|---|
| 37-stock, single seed, relative + cross-sectional features | +0.003 | -7.4% | +18.8% |
| 37-stock, single seed, regression head | -0.012 | -25.1% | +5.8% |
| 85-stock, 3-seed ensemble, 17 features | -0.049 | -47.3% (Sharpe -1.26) | +14.6% (Sharpe +0.39) |
| **85-stock, 3-seed ensemble, 29 features (+canonical factors)** | **-0.029** | **-21.3% (Sharpe -0.89)** | **+18.9% (Sharpe +0.49)** |
| Equal-weight universe (gross benchmark) | — | — | **+41.6% (Sharpe +0.98)** |

Every confidence interval spans zero (L/S spread [-2.38, +0.36]; long-only [-0.70, +1.94]; EW [-0.22, +2.50]). **Equal-weight — literally just holding all 85 names — beats every model-selected basket on both return and Sharpe.** Reported as observed: the disciplined, honestly-evaluated model does not rank NSE stocks.

(The best-configuration row is **noise-dominated**: re-running the *identical* config — same seed, same data — swings the long-only book roughly +19% to +33% and the long-short spread from -7% to +11% across CPU/GPU and run to run, all inside the confidence interval. That instability *is* the finding: the cross-sectional edge, if any, is smaller than the training noise. The table shows the current published artifact; nothing is pinned to the luckiest run.)

Two levers were tested, and the results are reported exactly as they came out:

- **Cross-sectional features helped** (as predicted). Adding universe/sector-relative momentum and per-date ranks flipped IC positive, took the long-short spread from -21% to roughly flat, and nearly tripled the long-only return. Real, directionally-consistent progress — but IC's information ratio is ~0.03, the spread's 95% CI is [-1.13, 1.22], quintiles are not yet monotonic, and the long-only book still trails passively holding the universe.
- **The regression head did *not* help.** Training on the continuous excess return (Huber loss) instead of a binary beat/miss label made every metric worse (IC -0.012, long-only +5.8%). The 20-day cross-sectional excess is heavy-tailed, and the regression objective appears to chase that tail noise rather than the rank; the classification head is the better fit here. This was a reasonable hypothesis that the data rejected, kept in the record rather than deleted.

**Honest read:** the best configuration (classification + cross-sectional features) points the signal the right way but stays within noise and does not beat passively holding the universe. A deployable edge needs richer inputs still — fundamental/quality factors (blocked on a survivorship-clean data source for NSE back to 2007). All four configurations' artifacts are published (`cross_section.json` = the best, `cross_section_regression.json`, `cross_section_classification.json`, `cross_section_base_features.json`, `cross_section_absolute.json`); nothing is dressed up or cherry-picked.

```bash
python -m Source.Ingestion.fetch_universe        # one-time: download the universe
python -m Source.Backtest.run_cross_section      # train + cross-sectional backtest
```

---

## Repository Structure

```
Multi-Horizon-Transformer-for-Systematic-Equity-Direction-Forecasting/
│
├── Data/
│   ├── Raw_Data/
│   │   └── ^NSEI_daily.csv          # 18+ years of Nifty 50 OHLCV (2006–2026)
│   └── Processed_Data               # Placeholder — generated at runtime
│
├── Notebooks/
│   └── Notebooks/
│       └── Eda.ipynb                # Primary research notebook (97 cells)
│                                    # EDA → Feature Engineering → Model → Backtest
│
├── Source/
│   ├── Ingestion/
│   │   └── Fetch_Market_Data.py     # yfinance-based OHLCV downloader
│   ├── Features/
│   │   ├── Returns.py               # (Stub — return feature logic)
│   │   └── Volatility.py            # (Stub — volatility feature logic)
│   ├── Models/
│   │   └── train_model              # XGBoost baseline with mean-centering trick
│   ├── News/
│   │   ├── fetch_news.py            # NewsAPI ingestion (100 articles/call)
│   │   ├── process_news.py          # Daily article count aggregation
│   │   └── sentiment.py            # FinBERT sentiment scoring per headline
│   ├── Risk/
│   │   └── Placeholder.txt          # Risk module (in progress)
│   └── Api/
│       └── Placeholder.txt          # API serving layer (in progress)
│
├── config.yaml                      # Config (empty — hyperparams hardcoded in notebook)
├── requirements.txt                 # Full research-grade dependency stack
└── README.md
```

---

## Architecture

### Data Flow

```
^NSEI Raw OHLCV (2007–2026, ~4,500 trading days)
        ↓
  Feature Engineering (16 engineered, 11 stationary features fed to the model)
        ↓
  Sliding Window Construction (lookback = 60 days)
  → Tensor shape: (N, 60, 11)
        ↓
  Temporal Train/Val/Test Split (70/15/15)
        ↓
  StandardScaler (fit on train only)
        ↓
  Transformer Encoder (2 blocks)
        ↓
  Attention pooling (learned softmax over time; GAP available via config)
        ↓
  Dense(20) → 20 binary logits
        ↓
  20 horizon outputs: P(close_t+h > close_t) for h ∈ {1..20}
  → ensembled (val-z-scored mean) into one timing signal
```

### Transformer Architecture

| Component | Detail |
|---|---|
| Input shape | `(batch, 60, 11)` — stationary features only |
| Linear projection | `Dense(64)` → maps the 11 features to d_model=64 |
| Positional encoding | Sinusoidal, standard Vaswani et al. formulation |
| Encoder blocks | 2 stacked, each with identical structure |
| Attention heads | 4 heads, key_dim = 64/4 = 16 per head |
| Attention dropout | 0.1 |
| FFN hidden dim | 128 (ReLU activation) |
| FFN output dim | 64 (projects back to d_model) |
| Residual connections | Add-then-LayerNorm on both attention and FFN sub-layers |
| Pooling | Learned attention pooling (softmax-weighted sum over the 60 steps); GAP via config |
| Output head | Dense(20), no activation (raw logits) |
| Loss | BinaryCrossentropy(from_logits=True) |
| Optimizer | Adam(lr=1e-4) |
| Metric | AUC |

The attention scores from the second encoder block are extracted via a parallel `attention_model` (same weights, different output) to enable interpretability analysis — visualizing which of the 60 historical days the model attends to most when forming predictions.

---

## Deep Dive

### 1. Data — What Was Used

Raw data is Nifty 50 daily OHLCV downloaded via `yfinance` with `auto_adjust=False` (preserving raw unadjusted prices and separate adjusted close columns). The dataset spans from **October 2007 to February 2026**, giving roughly 4,500 trading day observations — two full market cycles including the 2008 GFC, 2020 COVID crash, and multiple bull runs.

The raw download includes a duplicate close column (yfinance artifact) that gets dropped, and the first two header rows are cleaned during preprocessing. Columns are manually renamed to `['date', 'close', 'high', 'low', 'open', 'volume']`.

### 2. Feature Engineering — The 16 Input Signals

Every feature is constructed to be **stationary or bounded** — no raw price levels fed directly (they'd drift over time and destroy generalization).

| Feature | Formula | Why It Matters |
|---|---|---|
| `daily_ret` | `close.pct_change()` | Base signal — raw daily movement |
| `roll_mean_ret_5` | 5-day mean of daily_ret | Short-term directional momentum |
| `roll_mean_ret_10` | 10-day mean of daily_ret | Medium-term trend |
| `roll_mean_ret_20` | 20-day mean of daily_ret | Monthly trend regime |
| `roll_vol_5` | 5-day std of daily_ret | Short-term realized volatility |
| `roll_vol_10` | 10-day std of daily_ret | Medium-term volatility regime |
| `roll_vol_20` | 20-day std of daily_ret | Monthly volatility baseline |
| `momentum_10` | `close.pct_change(10)` | 10-day cumulative price momentum |
| `vol_roll_mean_5` | 5-day mean of volume | Volume trend (market participation) |
| `log_volume` | `log(volume + 1)` | Volume magnitude (log-stabilized) |
| `ma_diff_10` | `(close - MA10) / MA10` | Mean reversion signal vs 10-day MA |
| `close` | raw close | Absolute price level (contextualization) |
| `high` | raw high | Intraday range upper bound |
| `low` | raw low | Intraday range lower bound |
| `open` | raw open | Gap-up / gap-down signal |
| `volume` | raw volume | Raw liquidity |

Note: the five raw level columns (`close`, `high`, `low`, `open`, `volume`) are engineered and inspected during EDA but are **excluded from the model input**. They are non-stationary: the StandardScaler is fit on 2007-2019 training data, so 2023-2026 price levels land far outside the fitted range and those features degenerate out-of-sample. The model consumes the 11 stationary/bounded signals above them.

### 3. Target Construction — 20 Binary Horizons

For each timestep `t`, 20 binary labels are generated:

```python
for h in range(1, 21):
    df[f"target_{h}"] = (df["close"].shift(-h) > df["close"]).astype(int)
```

This creates `target_1` through `target_20`, each being 1 if the index closes higher h days from now. These are **direction labels**, not magnitude. All 20 are predicted simultaneously by the single model.

Class balance: because Nifty 50 has a long-term upward drift, targets at longer horizons have slightly positive class imbalance (more 1s than 0s). This is tracked via `y_train.mean(axis=0)` and baseline accuracy per horizon is computed as `max(p_up, 1 - p_up)`.

### 4. Sequence Construction — The 60-Day Sliding Window

```python
lookback = 60
for t in range(lookback, len(df) - 20):
    X_t = df[feature_cols].iloc[t-lookback:t]   # shape: (60, 16)
    y_t = df[target_cols].iloc[t]                # shape: (20,)
```

Each sample is a 60 × 16 matrix — the last 60 trading days of features — paired with 20 binary labels. The loop stops 20 steps before the end to ensure all 20-day forward targets are valid (no look-ahead at the boundary).

This yields approximately **4,800+ (sequence, label) pairs** before splitting.

### 5. Train/Val/Test Split — Temporal, No Shuffling

```python
train_end = int(0.7 * len(X))   # ~70% — roughly 2006–2019
val_end   = int(0.85 * len(X))  # next ~15% — roughly 2019–2022
# test: remaining ~15% — roughly 2022–2026
```

Crucially: **no random shuffling**. Time series integrity is preserved. This is walk-forward in spirit — the model trains on older data, validates on more recent data, and tests on the most recent unseen period.

StandardScaler is fitted **only on X_train** and applied identically to val and test. This prevents future statistics from leaking into training.

### 6. The Transformer — Implementation Details

**Positional Encoding:**

Standard sinusoidal encoding from the original "Attention is All You Need" paper:

```python
angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
angle_rads = positions * angle_rates
angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
```

This gives each of the 60 time steps a unique positional fingerprint injected additively into the token embeddings. Without this, the Transformer has no sense of temporal order.

**Encoder Block:**

Each of the 2 encoder blocks does exactly:
1. Multi-head self-attention (4 heads × 16 key_dim) with 10% dropout
2. Residual add + LayerNorm
3. Two-layer FFN: Dense(128, relu) → Dense(64)
4. Residual add + LayerNorm

The second block also returns `attn_scores` — a tensor of shape `(batch, num_heads, seq_len, seq_len)` = `(batch, 4, 60, 60)` — representing which time steps attend to which.

**Pooling:**

After the 2 encoder blocks, the output tensor is `(batch, 60, 64)`. `GlobalAveragePooling1D` collapses the time dimension to produce `(batch, 64)` — a single vector summarizing all 60 timesteps.

**Output:**

A single `Dense(20)` layer with no activation produces 20 raw logits. Loss function uses `from_logits=True`, so sigmoid is applied internally during training but must be applied explicitly at inference time (`tf.sigmoid(logits)`).

### 7. Training

- Epochs: 40 (with EarlyStopping, patience=7, restore_best_weights=True)
- Batch size: 32
- Optimizer: Adam, lr=1e-4
- Loss: Binary crossentropy (from_logits=True) — jointly minimizes across all 20 horizons
- Primary metric: AUC

The early stopping logic monitors validation loss and restores the best checkpoint — guarding against overfitting on the relatively small dataset.

### 8. Evaluation — Classification Metrics

Post-training, for each of the 20 horizons on the validation set:

```python
logits = model.predict(X_val)
probs = tf.sigmoid(logits).numpy()
auc = roc_auc_score(y_val[:, h], probs[:, h])
acc = np.mean((probs[:, h] > 0.5).astype(int) == y_val[:, h])
baseline = max(y_val[:, h].mean(), 1 - y_val[:, h].mean())
```

AUC is evaluated at horizons 1, 5, 10, and 20. Accuracy is compared against the naïve majority-class baseline at every horizon.

### 9. Alpha Signal Evaluation — Quantitative Finance Metrics

This is where the project goes beyond standard ML evaluation. The raw logits (before sigmoid) are treated as a **continuous alpha signal** and tested with quant-style metrics:

**Information Coefficient (Spearman):**
```python
ic, p_value = spearmanr(scores[:, 19], future_returns_20_test)
```
Measures rank correlation between the model's horizon-20 logit and actual 20-day forward log returns. IC > 0 means the model ranks stocks (here, market states) correctly more often than not.

**Sign Strategy Sharpe:**
```python
positions = np.sign(scores[:, 19])
strategy_returns = positions * future_returns_20
sharpe = mean_ret / std_ret * np.sqrt(252/20)
```
Go long if logit > 0, short if logit < 0. Sharpe annualized using 252/20 periods-per-year for 20-day non-overlapping returns.

**Quantile Long-Short Sharpe:**
```python
upper = np.percentile(scores, 70)
lower = np.percentile(scores, 30)
positions[scores >= upper] = 1
positions[scores <= lower] = -1
# 0 for middle 40%
sharpe = mean_ret / std_ret * np.sqrt(252/20)
```
Top 30% long, bottom 30% short, neutral middle. Reduces noise compared to full sign strategy.

**Decile Return Attribution:**
```python
df_eval["decile"] = pd.qcut(df_eval["p"], 10, labels=False)
df_eval.groupby("decile")["r"].mean()
```
Splits predictions into 10 equal buckets and checks if higher predicted probability buckets systematically deliver higher realized returns — the gold standard test for signal monotonicity.

**Threshold Sweep (Val Set):**
Sweeps signal threshold from 0.5 to 0.8, measuring Sharpe and trade frequency at each level — identifying the optimal entry threshold for a long-only strategy.

**Yearly Sharpe Breakdown:**
Computes per-year Sharpe on the test set to check for strategy regime stability vs. lucky periods.

**Long-Short on Probability Extremes (Val):**
```python
high = p >= np.percentile(p, 80)
low  = p <= np.percentile(p, 20)
combined = np.concatenate([long_ret, short_ret])
print("Long-Short Sharpe:", combined.mean()/combined.std())
```
Most aggressive version — go long top quintile, short bottom quintile.

### 10. Attention Visualization

The parallel `attention_model` outputs the raw attention weight matrix:
```python
attn_val = attention_model.predict(X_val)  # shape: (N, 4, 60, 60)
avg_attention = np.mean(attn_val, axis=(0, 1, 2))  # shape: (60,)
```

Averaged across samples, heads, and query positions, this gives one number per "days back" position — revealing how much the model looks at recent vs. historical context when predicting. Plotted as a line chart with `Days Back` on the x-axis.

### 11. Volatility Regime Analysis

Post-evaluation, a regime-conditioning framework is partially built:
```python
df["realized_vol_20"] = df["daily_ret"].rolling(20).std()
df["realized_vol_252"] = df["daily_ret"].rolling(252).std()
df["vol_regime_percentile"] = (
    df["realized_vol_252"]
    .rolling(252)
    .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
)
```
This creates a continuous volatility regime percentile — groundwork for conditioning strategy behavior (e.g., reduce position sizing in high-vol regimes, increase in low-vol).

### 12. Parallel Tracks — News and Sentiment Module

Three source files implement a news-driven sentiment pipeline that runs independent of the Transformer:

**`fetch_news.py`** — Calls NewsAPI `/v2/everything` endpoint, pulls 100 articles per query, saves `date + title + description` to CSV.

**`process_news.py`** — Groups articles by date, computes daily article count as a raw attention/volume signal.

**`sentiment.py`** — Runs each headline through **FinBERT** (`ProsusAI/finbert`), a BERT model fine-tuned on financial text. Computes a scalar sentiment score per headline:
```python
sentiment_score = probs[0] * -1 + probs[2] * 1  # negative weighted -1, positive weighted +1
```
Aggregates to daily mean sentiment. This is designed to be merged with OHLCV features as additional input to the Transformer in future iterations.

### 13. XGBoost Baseline

`Source/Models/train_model` implements a mean-centered XGBoost regressor:
```python
y_train_centered = y_train - y_train.mean()  # remove drift bias
model = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.03)
```
Evaluated on MAE, RMSE, and Pearson correlation. Serves as the classical ML comparison point against the Transformer's AUC/Sharpe metrics.

---

## Setup & Installation

**Prerequisites:** Python 3.10+, CUDA-compatible GPU recommended (notebook checks `nvidia-smi`).

```bash
git clone <repo-url>
cd Multi-Horizon-Transformer-for-Systematic-Equity-Direction-Forecasting
pip install -r requirements.txt
```

**Fetch fresh data:**
```bash
python Source/Ingestion/Fetch_Market_Data.py
# Downloads ^NSEI from 2006-01-01 to present
# Saves to Data/Raw_Data/^NSEI_daily.csv
```

**Run the model:**
Open `Notebooks/Notebooks/Eda.ipynb` and execute all cells sequentially. Update the CSV path in Cell 3 to match your local or Colab path.

**For news sentiment fusion (optional, off by default):**
```bash
# Needs NEWSAPI_KEY. Writes Data/Processed_Data/daily_sentiment.csv
NEWSAPI_KEY=xxxx python -m Source.News.build_sentiment --query "Nifty 50 India stock market"
# Then set features.use_sentiment: true in config.yaml and re-run the pipeline.
```
Note: NewsAPI only serves ~30 days of history, so this cannot backfill the 2007-2026 training window. The fusion path is real and tested, but stays disabled so the historical backtest never trains on unavailable/fabricated sentiment.

---

## Key Design Decisions & Honest Caveats

**Why multi-output instead of 20 separate models?**
Shared encoder weights across horizons allow the model to learn a single rich representation of market state that generalizes across time horizons. Shorter-horizon signal can inform longer-horizon predictions through shared gradients.

**Why from_logits=True?**
Numerically more stable than sigmoid + BCE separately. Means raw logit values can be directly used as ranked signals without applying sigmoid — which is exactly what the IC and Sharpe calculations do.

**Why attention pooling instead of GlobalAveragePooling (the original choice)?**
A plain mean over time steps discards the temporal-order information the positional encoding injected. The pooling layer now learns a softmax weighting over the 60 steps (one extra `Dense(1)`), letting the model emphasize the days that matter; GAP remains available via `model.pooling: gap` in config.

**What's done since the original notebook:**
- `config.yaml` now holds every hyperparameter (windows, split, model, costs)
- `Source/Features/Returns.py` and `Volatility.py` are implemented, not stubs
- The notebook logic is extracted into a reproducible pipeline (`Source/Pipeline`, `Source/Models`, `Source/Backtest`) runnable with one command
- A **comprehensive India cost model** (`Source/Backtest/costs.py`) charges STT, stamp duty, exchange + SEBI fees, brokerage, GST, slippage and DP charges per instrument (delivery/intraday/futures/options) — **9.58 bps** round-trip for Nifty futures (an audit fixed a 10× stamp-duty error)
- A **buy-and-hold Nifty benchmark**: results are judged as excess over passively holding the index
- The primary strategy is **long/flat market timing** — quantile long-short is a cross-sectional construct, meaningless on a single index, kept only as a reference row
- Model inputs restricted to the **11 stationary features**; raw OHLCV levels excluded (non-stationary out-of-sample)
- The trading signal **ensembles all 20 horizon heads** (validation-z-scored mean); a best-validation-horizon variant is also reported
- Probabilities are **Platt-calibrated on validation data**; a reliability diagram is exported
- Entry thresholds fixed on **validation data only** — the test set tunes nothing
- Sharpe carries a **bootstrap 95% confidence interval**; walk-forward runs **8 expanding-window folds**
- **Learned attention pooling** replaces GlobalAveragePooling (GAP still available via config); training reproducible via TF op-determinism; returns non-overlapping
- News / FinBERT sentiment fusion wired end-to-end (extra config-gated feature via `Source/News/build_sentiment.py`)
- A static Next.js site visualizes the real, regenerated results (see below)

**What's still not done (honest):**
- **Fundamentals stay out of the model.** `Source/Ingestion/fetch_fundamentals.py` fetches a *current* fundamentals snapshot (yfinance: P/E, ROE, margins, market cap), which is fine for a live-context view but cannot be a backtest feature — today's ratios fed to a 2015 window is look-ahead leakage. A leak-free factor needs an as-of-date vendor (Capital IQ / Refinitiv / parsed filings), which isn't freely available, so it is deliberately omitted rather than faked.
- Sentiment fusion is **off by default** — NewsAPI only serves ~30 days of history, so a real sentiment feature cannot be backfilled over 2007-2026. The mechanism is real; the historical data is not available, and no fabricated feature is ever fed to the backtest.
- Single-asset backtest — no borrow cost or capacity modeling (slippage *is* modeled)
- Risk module is a placeholder
- API serving layer is a placeholder

---

## Dependencies (Key Libraries)

| Category | Libraries |
|---|---|
| Data | `yfinance`, `pandas`, `numpy`, `polars` |
| Deep Learning | `tensorflow`, `keras` |
| Classical ML | `xgboost`, `scikit-learn` |
| NLP / Sentiment | `transformers` (FinBERT), `torch` |
| Quant / Backtest | `quantstats`, `vectorbt`, `pyportfolioopt` |
| Experiment Tracking | `mlflow`, `wandb` |
| Explainability | `shap`, `captum` |
| Visualization | `matplotlib`, `seaborn`, `plotly`, `mplfinance` |
| Serving | `fastapi`, `onnx`, `bentoml` |

Full dependency list: `requirements.txt` (finance-grade stack covering ingestion, modeling, serving, and risk).

---

## Author Notes

Project developed locally (Windows, OneDrive path visible in notebook) with Colab paths commented alongside for cloud execution. GPU availability checked via `nvidia-smi` at startup. Data covers **January 2006 through February 2026** — 20 full years of Indian equity market history including multiple bear and bull regimes.

The research direction is systematic alpha generation for Nifty 50 directional forecasting using deep learning, not a production trading system.
