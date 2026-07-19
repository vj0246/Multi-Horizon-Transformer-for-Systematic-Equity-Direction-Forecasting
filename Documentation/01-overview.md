# 1. Overview

## What the system does

One Transformer encoder reads a 60-day window of engineered features and emits
**20 logits at once** — one per forward horizon, 1 through 20 trading days. Each
logit answers: *will the close in `h` days be higher than today's close?*

```
60 days x 19 features  ->  Transformer encoder  ->  20 logits
                                                    (h=1 ... h=20)
```

Predicting all horizons from a shared trunk is a deliberate choice: short- and
long-range structure inform each other through shared gradients, instead of
training 20 unrelated models.

## Two tracks

| Track | Question | Entry point | Artifact |
|-------|----------|-------------|----------|
| **Index** | Will the Nifty 50 rise over the next h days? | `Source/Backtest/run.py` | `summary.json`, `horizons.json`, `strategies.json` |
| **Cross-section** | Which of 85 NSE stocks will beat the universe median? | `Source/Backtest/run_cross_section.py` | `cross_section.json`, `stock_signals.json` |

The cross-sectional track exists because ranking stocks *against each other* is
where direction models historically earn: it neutralises market beta, so the
model must find relative information rather than ride the index drift.

## The headline finding

**No exploitable edge in either track.**

| Measure | Value | Baseline | Verdict |
|---------|-------|----------|---------|
| Mean OOS AUC (index) | 0.5123 | 0.50 | Indistinguishable |
| Horizons significant after correction | 0 of 20 | — | None |
| Cross-sectional pooled IC | −0.026 | 0.00 | Slightly backwards |
| Cross-sectional quintiles | inverted | monotonic | Ranks wrong way |
| Long/short spread return | −20.4% | — | Loses |
| Equal-weight benchmark | +41.6% | — | Beats every model basket |
| Paper trading, 674 OOS days | +19.2% | +24.5% buy-and-hold | Underperforms |

Every lever was tried and none moved the result:

- **Better model?** LightGBM ≈ Transformer ≈ 0.50. Architecture is not the constraint.
- **More data?** 28x more (86k panel rows vs 3k). No improvement.
- **More features?** Macro (India VIX, USDINR, crude, overnight S&P, breadth). No improvement.
- **Tuned architecture?** Validation-only grid search over dropout and learning rate. No improvement.

## Why: the binding constraint

The limit is **effective sample size**, not model capacity.

A 20-day forward label sampled daily overlaps its neighbour by 19 of 20 days.
Consecutive labels are therefore almost the same observation. The independent
information in the test set is roughly:

```
effective n = number of test days / horizon
            = 654 / 20
            ≈ 33 independent observations
```

You cannot establish a small edge from 33 observations. To be significant at
h=20, AUC would need to reach roughly **0.70** on its own, and about **0.80** to
survive Bonferroni correction across all 20 horizons. An honest edge in daily
index direction is perhaps 0.52 to 0.55. **The experiment cannot resolve the
effect size it is looking for**, and no amount of modelling fixes that.

The only real fix is more independent observations, which means higher-frequency
data (intraday bars) or a wider cross-section of genuinely independent assets.
Free intraday NSE history going back to 2007 does not exist, and paid feeds were
out of scope.

## What is genuinely solid here

The measurement apparatus, not the alpha:

- Strict temporal splits, scaler fit on training data only, no shuffling
- `merge_asof(direction="backward")` plus `.shift(1)` on every macro series
- Overlap-corrected AUC standard errors (effective n, not raw n)
- Deflated Sharpe ratio accounting for the number of strategies tried
- Bonferroni and Benjamini-Hochberg correction across the 20 horizons
- Real India transaction costs, per instrument, charged on every trade
- A frozen model whose cutoff is stored in its metadata, so forward paper
  trading can never drift into in-sample days
- 32 tests covering leakage, cost math, metric math, and artifact validity

## What this is not

- Not investment advice
- Not a trading system — it has no demonstrated edge
- Not a claim that Nifty direction is unpredictable in principle, only that it
  is not predictable *from daily OHLCV and these macro series, at this sample size*

Continue to [Getting Started](02-getting-started.md).
