# 6. Evaluation

Every metric computed, its formula, and why it is there.
Implementation: `Source/Evaluation/suite.py`.

## Classification metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| Accuracy | (TP+TN)/N | Misleading alone — beaten by always predicting "up" when 53% of days are up |
| Precision | TP/(TP+FP) | Of predicted ups, how many rose |
| Recall | TP/(TP+FN) | Of actual ups, how many were caught |
| F1 | 2PR/(P+R) | Harmonic mean |
| **ROC-AUC** | P(score(up) > score(down)) | **The primary metric.** Threshold-free, balance-insensitive |
| Confusion matrix | TP/FP/TN/FN | Reported in full |

AUC is primary because it measures *ranking* ability independent of any decision
threshold or class imbalance. 0.50 = coin flip.

## Error metrics

| Metric | Formula |
|--------|---------|
| MSE | mean((y − ŷ)²) |
| RMSE | √MSE |
| MAE | mean(\|y − ŷ\|) |
| MAPE | mean(\|(y − ŷ)/y\|) × 100 |
| R² | 1 − SS_res/SS_tot |

`error_metrics(..., continuous=False)` returns `None` for MAPE on binary targets
— MAPE divides by `y`, which is 0 for half the observations. Reporting `inf` or
silently dropping those rows would both be wrong.

## Financial metrics

| Metric | Formula | Why |
|--------|---------|-----|
| **Sharpe** | mean(r)/std(r) × √periods | Risk-adjusted return |
| **Sortino** | mean(r)/std(r⁻) × √periods | Penalises only *downside* deviation |
| **Calmar** | annual return / \|max drawdown\| | Return per unit of worst loss |
| Max drawdown | min(equity/cummax(equity) − 1) | Worst peak-to-trough |
| Profit factor | Σ gains / \|Σ losses\| | >1 is profitable gross |
| Hit rate | fraction of periods with r > 0 | |
| Exposure | mean(\|position\|) | Time in market |

**Strategy returns are non-overlapping.** A 20-day holding period is evaluated on
20-day blocks that do not share days. Using overlapping daily returns would
inflate the Sharpe by roughly √20 ≈ 4.5x. All returns are **net of India costs**.

## Statistical tests

### Overlap-corrected AUC standard error

The single most important correction in this project.

Hanley-McNeil assumes i.i.d. observations. Direction labels over an h-day horizon
sampled daily share h−1 of their h days with each neighbour, so independent
information is roughly n/h observations:

```python
def _auc_se(auc, y, overlap=1):
    ov = max(int(overlap), 1)
    n1, n0 = int(y.sum()) / ov, (len(y) - int(y.sum())) / ov   # EFFECTIVE counts
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc*(1-auc) + (n1-1)*(q1-auc**2) + (n0-1)*(q2-auc**2)) / (n1*n0)
    return sqrt(max(var, 0))
```

Feeding raw `n` understates the SE by ~√h (4.5x at h=20) and turns noise into
"significance". This is the easiest way to fake an edge in this project.

### Deflated Sharpe Ratio

Bailey & López de Prado. Adjusts an observed Sharpe for **the number of
strategies tried**, plus skewness and kurtosis of returns. Trying 20 strategies
and reporting the best without deflation is how backtests lie.

### Multiple testing

Testing 20 horizons at α=0.05 means ~1 false positive expected by chance alone.

- **Bonferroni:** reject if p ≤ α/n. Conservative, controls family-wise error.
- **Benjamini-Hochberg:** controls false discovery rate. Less conservative.

`multiple_testing` returns per-hypothesis `bonferroni_reject` / `bh_reject` flags
alongside the counts. Callers must use the flags — reconstructing them by taking
the k-th smallest p-value as a cutoff over-rejects when p-values tie.

### Diebold-Mariano

Tests whether two forecasts differ significantly in accuracy. Used against a
naive baseline. Result: p = 0.215 — **not distinguishable from naive**.

### Friedman

Non-parametric test across multiple models over multiple datasets, for whether
any model consistently ranks higher.

## The interpretation rule

A probability is only meaningful if its AUC interval clears 0.50. From the
shipped `predictions.json`:

| h | P(up) | AUC | 95% CI | eff n | Verdict |
|---|-------|-----|--------|-------|---------|
| 7 | 68.0% | 0.471 | [0.354, 0.589] | 95 | indistinct |
| 10 | 69.7% | 0.483 | [0.342, 0.624] | 66 | indistinct |
| 13 | 59.8% | 0.584 | [0.426, 0.741] | 51 | indistinct |
| 20 | 62.7% | 0.553 | [0.349, 0.758] | 33 | indistinct |

A 69.7% P(up) at h=10 looks like a strong call. Its AUC interval spans 0.50, so
there is **no evidence** the model ranks better than chance at that horizon. The
site is laid out to make this visible in the same row rather than buried.

## Three bugs that manufactured skill

Each produced apparent edge that was not real. Each is now a regression test.
These are the most transferable lessons here.

### 1. The i.i.d. AUC standard error

`_auc_se` originally used raw `n`. It reported **2 horizons as Bonferroni-
significant**. After correcting for overlap: **0 of 20**. The framework's own
statistics were the source of the apparent edge.

### 2. The Spearman IC p-value

`horizons.json` carries a `p_value` that is a **Spearman IC p-value over raw
overlapping labels** — the same i.i.d. error in a different metric. Feeding it
into multiple-testing reported **4 of 20 horizons actionable**. Testing AUC
against 0.50 with the effective-n SE gives the correct **0 of 20**.

That field is retained in the artifact but must never be used for significance.

### 3. Error bars from the wrong model

`predictions.json` once paired probabilities from the **frozen paper model** with
AUCs read from `horizons.json`, which belongs to the **backtest model** — a
different fit trained on different data. The error bars described a predictor
that had never been measured. Skill is now computed from the frozen model's own
logits over its own out-of-sample period.

### The pattern

All three made results look *better*. Bugs that flatter you are the ones you do
not go looking for. Every statistical helper here now has a test that asserts the
conservative answer.

## What would count as a real edge

| Requirement | Threshold |
|-------------|-----------|
| Single horizon significant | AUC ≈ 0.70 at h=20 |
| Surviving Bonferroni across 20 | AUC ≈ 0.80 |
| Realistic honest edge | AUC 0.52–0.55 |

The gap between the last row and the first two **is the whole problem**. It is a
sample-size problem, not a modelling problem. Chasing an AUC threshold by
retrying models until one clears it does not produce an edge — it produces a
false positive, which is why that approach was refused during development.

Continue to [Results](07-results.md).
