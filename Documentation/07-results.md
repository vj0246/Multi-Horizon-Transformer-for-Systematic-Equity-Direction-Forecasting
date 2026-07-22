# 7. Results

Every number here is read from the committed artifacts. Nothing is rounded
favourably or omitted.

## Index track

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Trading days | 4,343 | 2008-06 → 2026-02 |
| Split | 2,984 / 639 / 640 | train / validation / test |
| Features | 19 | 11 price+volume, 8 macro |
| **Mean AUC (backtest model, test)** | **0.5033** | Coin flip is 0.50 |
| **Mean AUC (frozen model, its own 675 OOS days)** | **0.5132** | Also indistinguishable |
| Mean IC (Spearman) | +0.021 | Near zero |
| **Horizons significant after correction** | **0 of 20** | None |

### Strategy performance (test, net of 9.58 bps round-trip futures cost)

| Metric | Strategy | Buy & hold |
|--------|----------|-----------|
| Net Sharpe | 0.630 | 0.940 |
| 95% CI on Sharpe | **[−0.576, 1.939]** | — |
| Total return | +11.2% | +30.2% |
| Max drawdown | −9.4% | — |

The confidence interval **spans zero**. The strategy loses to simply holding the
index, and its Sharpe is not distinguishable from zero.

Per the honest disclosure in `summary.json`, this Sharpe also carries selection
optimism: the entry-*threshold rule* was chosen after observing that the frozen
validation cutoff was degenerate on test. Read 0.630 as an **upper bound**, not a
clean out-of-sample estimate.

### Walk-forward (8 expanding folds)

| Metric | Value |
|--------|-------|
| Mean Sharpe | 0.277 |
| Std across folds | 0.364 |

The standard deviation exceeds the mean. No stable edge across regimes.

## Cross-sectional track (85 NSE names)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pooled rank IC | **−0.026** | Slightly **backwards** |
| Pooled AUC h=20 | 0.4876 | Below coin flip |
| Quintile mean 20d forward return | 1.38 / 1.27 / 1.22 / 0.93 / 0.92 % | **Inverted** — supposed best quintile is worst-ranked |
| Long/short spread | **−20.4%** | Loses money |
| Equal-weight benchmark | **+41.6%** | Beats every model basket |

The model ranks stocks slightly the wrong way. Note the temptation and the trap:
flipping the signal because test IC is negative would be **look-ahead bias** —
you only know the sign from the test set. It is not done.

Four configurations are published, none cherry-picked: `cross_section.json`
(classification + cross-sectional features, best), plus `_base_features`,
`_absolute`, and `_regression` archives.

> **Noise warning.** The cross-sectional track is noise-dominated. The same
> config and seed swings long-only from +19% to +33% run-to-run, because GPU
> `MultiHeadAttention` is not fully deterministic. Do not read precision into
> these point estimates or pin results to the luckiest run.

## Paper trading (live, forward)

676 out-of-sample days from 2023-10-19, scored by a model frozen before that date.

| Metric | Value |
|--------|-------|
| Paper return | **+18.5%** |
| Buy & hold | **+23.8%** |
| Excess | **−5.2%** |
| Sharpe | 0.69 |
| Max drawdown | −11.4% |
| Round trips | 13 |
| Time in market | 40.1% |
| Current position | LONG |

The Sharpe of 0.69 is **the index's own bull-run beta, not alpha** — the strategy
is in the market 40% of the time during a rising market. Excess return is
negative. This is exactly what the no-edge finding predicts, demonstrated forward
on real prices with real costs.

## Baselines

| Model | Result |
|-------|--------|
| Transformer (index) | AUC 0.5033 |
| LightGBM (index) | ≈ 0.50 |
| Transformer (cross-section) | IC −0.026 |
| LightGBM (cross-section) | ≈ 0 |
| Diebold-Mariano vs naive | **p = 0.215** — not distinguishable |
| Deflated Sharpe | 0.916 |

**Architecture is not the bottleneck.** A gradient-boosted tree and a Transformer
reach the same place, which is what you expect when the signal-to-noise ratio,
not model capacity, is binding.

## Everything that was tried and did not work

| Lever | Change | Outcome |
|-------|--------|---------|
| Better model | LightGBM, purged CV | ≈ 0.50 |
| More data | 86k panel rows (28x) | No improvement |
| More features | 8 macro series | No improvement |
| Cross-sectional features | Universe-demeaned, ranks, sector-relative | Moved metrics right, still noise |
| Architecture tuning | Validation-only grid | No improvement |
| Regression objective | Huber on excess return | **Worse** (IC −0.012) |
| Positioning data | — | **Worse** |
| Seed ensembling | 3 seeds | Reduced variance, not bias |
| Conviction gating | Trade only when confident | No significant edge |
| Volatility targeting | 15% annual target | Changed risk, not skill |

## How to read all of this

The correct summary is: **daily Nifty 50 direction is not predictable from daily
OHLCV and these macro series at this sample size.**

That is a real, useful finding. It is not a claim that markets are efficient, nor
that the approach is worthless at higher frequency or with better data. It is a
statement about what this dataset can support.

The binding constraint is [effective sample size](01-overview.md#why-the-binding-constraint):
~33 independent observations at h=20. Detecting a realistic 0.52–0.55 AUC edge
would need roughly 0.70 to clear significance. **The experiment cannot resolve
the effect size it is looking for.**

Continue to [Instrument Choice](08-instrument-choice.md).
