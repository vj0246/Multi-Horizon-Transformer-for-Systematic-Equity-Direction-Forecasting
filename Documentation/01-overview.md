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

## Four tracks

| Track | Question | Entry point | Artifact |
|-------|----------|-------------|----------|
| **Index, daily** | Will the Nifty 50 rise over the next h days? | `Source/Backtest/run.py` | `summary.json`, `horizons.json` |
| **Cross-section** | Which of 85 NSE stocks beats the universe median? | `Source/Backtest/run_cross_section.py` | `cross_section.json` |
| **Intraday, hourly** | Same question on hourly bars at a ~3-day horizon | `Source/Intraday/run.py` | `intraday.json` |
| **Paper trading** | What does the frozen model do forward, on real prices? | `Source/Paper/run.py` | `paper_trading.json` |

The cross-sectional track exists because ranking stocks *against each other*
neutralises market beta, forcing the model to find relative information rather
than ride the index drift. The intraday track exists to attack the sample-size
constraint below.

## The headline finding

**No exploitable edge in either track.**

| Measure | Value | Baseline | Verdict |
|---------|-------|----------|---------|
| Mean AUC, index daily | 0.5033 | 0.50 | Indistinguishable |
| Mean AUC, frozen model on its own 675 OOS days | 0.5132 | 0.50 | 0 of 20 actionable |
| Mean AUC, intraday hourly | 0.5031 | 0.50 | 0 of 20 significant |
| Cross-sectional pooled IC | −0.026 | 0.00 | Slightly backwards |
| Cross-sectional quintiles | inverted | monotonic | Ranks wrong way |
| Long/short spread return | −20.4% | — | Loses |
| Equal-weight benchmark | +41.6% | — | Beats every model basket |
| Paper trading, 676 OOS days | +18.5% | +23.8% buy-and-hold | Underperforms by 5.2pp |

**Four independent experiments, four null results.** That consistency is itself
informative — it is what a genuine absence of signal looks like, rather than one
unlucky configuration. Full numbers: [Full Evaluation](14-full-evaluation.md).

Every lever was tried and none moved the result:

- **Better model?** LightGBM ≈ Transformer ≈ 0.50. Architecture is not the constraint.
- **More data?** 28x more (86k panel rows vs 3k). No improvement.
- **More features?** Macro (India VIX, USDINR, crude, overnight S&P, breadth). No improvement.
- **Higher frequency?** Hourly bars, 5,057 of them. Same 0.503.
- **Tuned architecture?** Validation-only grid over dropout and learning rate. No improvement.

## Why: the binding constraint

The limit is **effective sample size**, not model capacity.

A 20-day forward label sampled daily overlaps its neighbour by 19 of 20 days.
Consecutive labels are therefore almost the same observation. The independent
information in the test set is roughly:

```
effective n = number of test observations / horizon
            = 654 / 20
            ≈ 33 independent observations
```

You cannot establish a small edge from 33 observations. To be significant at
h=20, AUC would need to reach roughly **0.70** on its own, and about **0.80** to
survive Bonferroni correction across all 20 horizons. An honest edge in daily
index direction is perhaps 0.52 to 0.55. **The experiment cannot resolve the
effect size it is looking for**, and no amount of modelling fixes that.

The intraday track tested the obvious fix and it did not work: hourly bars give
eff n **37** against daily's 32, because the binding constraint is not bar size
but that free hourly history stops at 730 days. Worse, frequency and horizon must
move *together* — hourly bars at the same 20-**day** horizon give eff n **5**,
which is worse than daily. And at 5-minute bars transaction costs consume 64% of
the typical move, so the required win rate becomes 82%.

The honest fix is more independent observations with costs still small relative
to the move. That means years of intraday history, which free sources do not
provide, or forward collection starting now.

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
- 58 tests covering leakage, point-in-time reporting lags, cost math, metric math, drift detectors, the retraining gate, and artifact validity

## What this is not

- Not investment advice
- Not a trading system — it has no demonstrated edge
- Not a claim that Nifty direction is unpredictable in principle, only that it
  is not predictable *from daily OHLCV and these macro series, at this sample size*

Continue to [Getting Started](02-getting-started.md).
