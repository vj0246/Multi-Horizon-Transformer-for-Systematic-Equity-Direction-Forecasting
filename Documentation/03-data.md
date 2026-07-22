# 3. Data

## Sources

| Source | Contents | Location | Tracked in git |
|--------|----------|----------|----------------|
| yfinance `^NSEI` | Daily OHLCV, 2007→2026, ~4,619 days | `Data/Raw_Data/^NSEI_daily.csv` | yes |
| yfinance macro | India VIX, USDINR, crude, S&P 500 | `Data/Raw_Data/Macro/*.csv` | yes |
| yfinance universe | 85 NSE large caps, daily OHLCV | `Data/Raw_Data/Universe/*.csv` | no (gitignored) |
| NewsAPI + FinBERT | Daily sentiment score | `Data/Processed_Data/daily_sentiment.csv` | no, and **off by default** |

All sources are free. No paid data feed is used anywhere.

## Raw CSV gotcha

The raw yfinance export for `^NSEI` has **three header rows**
(`Price` / `Ticker` / `Date`) and a duplicated adjusted-close column.
`Source/Pipeline/data_loader.load_ohlcv` strips both. Loading it with a plain
`pd.read_csv` produces silent garbage.

The universe CSVs are written by `fetch_universe.py` in a **different, simpler
format** (`date,close,high,low,open,volume`, single header) and must be read with
`pd.read_csv(path, parse_dates=["date"])`, not `load_ohlcv`.

## Features

19 features per day: 11 price/volume plus 8 macro. **Raw OHLCV price levels are
deliberately excluded** — they are non-stationary, so a model trained on
2007–2020 price levels sees out-of-range inputs in 2024 and degrades.

### Price and volume (11) — `Source/Features/`

| Feature | Definition |
|---------|------------|
| `daily_ret` | 1-day log return |
| `roll_mean_ret_5/10/20` | Rolling mean return |
| `roll_vol_5/10/20` | Rolling standard deviation of returns |
| `momentum_10` | 10-day price momentum |
| `log_volume` | log(volume) |
| `vol_roll_mean_5` | 5-day mean volume |
| `ma_diff_10` | Price minus 10-day moving average |

### Macro (8) — `Source/Features/Macro.py`

| Feature | Definition | Why |
|---------|------------|-----|
| `vix_chg`, `vix_z60` | India VIX change, 60-day z-score | Volatility regime |
| `spx_ret`, `spx_ret_5` | Overnight S&P return | Global risk sentiment |
| `usdinr_ret` | Rupee move | Foreign flow proxy |
| `crude_ret` | Crude oil move | India is oil-importing |
| `breadth_above_ma20` | Fraction of universe above 20-day MA | Market internals |
| `xs_dispersion` | Cross-sectional return dispersion | Regime indicator |

## Leakage rules

These are the rules that make the results trustworthy. Every one has a test.

### 1. Macro series are lagged, always

Auxiliary series publish at different times and can be revised. Every macro
feature is joined **backwards** and then shifted one day:

```python
def _asof_lagged(df, aux, col):
    merged = pd.merge_asof(df[["date"]].sort_values("date"),
                           aux.sort_values("date"),
                           on="date", direction="backward")
    return merged[col].shift(1).to_numpy()
```

`direction="backward"` means never look forward. `.shift(1)` means today's model
input uses at most yesterday's macro reading. Without the shift, the India VIX
close would leak same-day information the model could not have had.

### 2. Targets are strictly forward

```python
df[f"target_{h}"] = (df["close"].shift(-h) > df["close"]).astype("float32")
```

Window `t` spans `df[t-60:t]` — it **excludes** row `t` — and predicts from row
`t`. The label for horizon `h` is `close[t+h] > close[t]`.

### 3. Windows stop early

`make_windows` iterates `range(lookback, len(df) - horizons)` so every window has
all 20 targets defined. No partially-labelled rows enter training.

### 4. Temporal split, no shuffling

70% train / 15% validation / 15% test, chronological. Never `train_test_split`.

### 5. Scaler fit on training data only

`StandardScaler` is fit on the training tensor and *applied* to validation and
test. Fitting on the full dataset would leak test-period mean and variance.

### 6. Everything selected on validation

Architecture and learning rate come from `scripts/select_model.py`, which reads
validation only. Platt calibration is fit on validation. The test set was scored
once.

**One honest exception, disclosed on the site and in `summary.json`:** the
entry-*threshold rule* was not selected cleanly. The frozen validation cutoff
turned out degenerate on test (the signal level shifts, so it never traded and
returned exactly 0). The rolling-window rule was adopted after observing that.
Its Sharpe therefore carries selection optimism and is an upper bound, not a
clean out-of-sample estimate. All three rules are published unmodified.

### 7. Cross-sectional splits are by date

The panel is split on **calendar date**, not row. Splitting by row would put
2024-03-05 for RELIANCE in train and the same date for TCS in test — the model
would learn that day's market move and "predict" it for other stocks.

### 8. No point-in-time fundamentals

`nse.py` and `screener.py` cover fundamentals now: NSE serves today's values (archived forward), Screener serves real history behind a disclosure lag.
Applying today's P/E to 2015 rows would be look-ahead leakage of the worst kind.
Free survivorship-clean NSE fundamentals back to 2007 do not exist, so
fundamental factors remain unimplemented rather than implemented wrongly.

## Known biases, stated plainly

| Bias | Status |
|------|--------|
| **Survivorship** | Present. The 85-name universe is today's large caps; names that delisted are absent. Cross-sectional returns are biased upward. Not correctable with free data. |
| **Look-ahead** | Addressed by rules 1–7 above, with tests. |
| **Selection on the threshold rule** | Present and disclosed (see rule 6). |
| **Multiple testing** | Corrected — Bonferroni and Benjamini-Hochberg across 20 horizons. |
| **Overlapping labels** | Corrected — effective-n standard errors, non-overlapping strategy returns. |

## Sentiment (optional, off)

```bash
python -m Source.News.gdelt --days 730     # free, no API key
# then set features.use_sentiment: true in config.yaml
```

**Kept off deliberately.** NewsAPI's free tier serves only ~30 days of history,
so sentiment cannot be backfilled to 2007. Training the historical backtest on
zero-filled or fabricated sentiment would invent a feature that did not exist —
worse than omitting it.

Continue to [Architecture](04-architecture.md).
