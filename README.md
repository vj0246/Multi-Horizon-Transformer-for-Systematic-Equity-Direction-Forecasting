# Multi-Horizon Transformer for Systematic Equity Direction Forecasting

> **A deep learning system that predicts the directional movement of the Nifty 50 index simultaneously across 20 distinct future time horizons — from next-day to 20-day forward — using a custom Transformer encoder trained on 18+ years of Indian equity market data.**

---

## Technical Summary

This project builds a **multi-output binary classification Transformer** to answer one question with 20 simultaneous answers: *Will the Nifty 50 close higher than today — 1 day, 2 days, 3 days... all the way to 20 days from now?*

The system ingests raw OHLCV data for `^NSEI` (Nifty 50) going back to 2007, engineers 16 time-series features capturing price momentum, rolling volatility, volume dynamics, and mean reversion signals, and feeds the **11 stationary ones** (raw price levels are excluded — see below) as 60-day sliding window sequences into a 2-block Transformer encoder with sinusoidal positional encoding and learned attention pooling. The model outputs 20 independent logits — one per forecast horizon — each representing the probability that the index will be higher at that future point.

Training uses strict temporal split (70/15/15 — no look-ahead leakage), StandardScaler fitted exclusively on training data, binary cross-entropy loss with AUC as the primary metric, and early stopping with patience=7. Post-training evaluation goes beyond classification metrics: the model's raw logits are used directly as alpha signals, tested via Spearman IC, sign-strategy Sharpe, quantile long-short Sharpe, and decile return attribution — all on held-out test data.

**The project is research-grade, not production-deployed.** Source modules for news ingestion (NewsAPI), FinBERT sentiment, and an XGBoost baseline exist in the codebase as parallel development tracks. The Transformer is the primary model.

---

## Live Demo

**→ [multi-horizon-transformer-for-syste.vercel.app](https://multi-horizon-transformer-for-syste.vercel.app)**

A static Next.js dashboard (in `frontend/`) visualizes the **real, regenerated** backtest results — per-horizon AUC/IC, the cost-aware equity curve, walk-forward folds, attention distribution, and training history. Every number is produced by an actual training run, not hand-authored.

### Reproduce end-to-end

```bash
# 1. train the Transformer + run the full backtest -> writes frontend/public/data/*.json
python -m Source.Backtest.run

# 2. rebuild + preview the site
cd frontend && npm install && npm run dev
```

Everything is driven by `config.yaml`. Training is made reproducible via TensorFlow op-determinism, so reruns give stable numbers. Set `REUSE=1` before `run.py` to regenerate the JSON from a cached run without retraining.

### Headline result (honest)
After realistic India **futures** costs (~11.2 bps round-trip: STT, stamp duty, exchange + SEBI fees, brokerage, GST, slippage — see `Source/Backtest/costs.py`), the model has **no exploitable edge**. On the 2023-2026 test window the long/flat ensemble timing strategy returns ≈ **-3%** versus **+40%** for buy-and-hold; net Sharpe is **-0.62 with a bootstrap 95% CI of [-1.10, 0.00]**, mean test AUC is **0.48** (below coin-flip), and 8-fold walk-forward Sharpe is **+0.25 ± 0.79** — statistically indistinguishable from zero. Notably, restricting inputs to stationary features and fixing all thresholds on validation data *lowered* the headline numbers versus earlier, sloppier evaluations: the apparent edge was evaluation artifact, not alpha. Daily Nifty direction is close to efficient; the site reports these numbers as-is, benchmarked against passively holding the index.

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
- An **itemized India cost model** (`Source/Backtest/costs.py`) charges STT, stamp duty, exchange + SEBI fees, brokerage, GST, and slippage — ~11.2 bps round-trip for Nifty futures
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
