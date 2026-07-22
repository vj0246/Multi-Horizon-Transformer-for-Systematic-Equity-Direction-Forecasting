# 9. Research Gaps

A literature review of deep-learning stock-prediction papers raised the critiques
below. Each is mapped to what this repository actually does, honestly, including
where it falls short.

## Summary table

| Critique | Status |
|----------|--------|
| Overfitting is pervasive | **Addressed** — dropout 0.4, early stopping, seed ensemble, validation-only selection |
| No model generalises across stocks/regimes | **Confirmed** — walk-forward Sharpe 0.277 ± 0.364 |
| Ensembles reduce overfitting | **Implemented** — 3-seed ensemble |
| Metaheuristic HPO (SSA, TLBO) | **Not implemented** — deliberate, see below |
| Sentiment improves volatile-regime accuracy | **Built, disabled** — data availability |
| Risk metrics beyond accuracy | **Implemented** — Sharpe, Sortino, Calmar, drawdown, DSR |
| Loss minimisation ≠ utility maximisation | **Addressed** — financial metrics are primary |
| Decomposition preprocessing (EMD/ICEEMDAN) | **Not implemented** — leakage risk, see below |
| Data drift handling (DAELM) | **Partially** — walk-forward measures it, no adaptive model |
| Hybrid architectures (CNN-BiLSTM-LGBM) | **Partially** — LightGBM baseline exists |
| Transformers capture high-lag dependencies | **Implemented** — 60-day attention |
| True multimodal fusion vs concatenation | **Valid criticism** — see below |
| No alternative data | **Confirmed gap** |
| Expanding-window CV reduces but does not eliminate look-ahead | **Addressed** — purged/embargoed CV |
| Survivorship bias rarely addressed | **Acknowledged, not fixed** — data-blocked |
| Latency ignored | **Not applicable** — daily horizon |
| Single-model reliance is risky | **Addressed** — ensemble + published baselines |

---

## 1. Overfitting

**Implemented:** dropout 0.4 (validation-selected, deliberately high), early
stopping with patience 7 and best-weight restore, a small model (~120k params, 2
layers, d_model 64), 3-seed ensembling, and strict validation-only architecture
selection via `scripts/select_model.py`.

**Honest note:** overfitting is *not* this project's failure mode. The model does
not overfit to a spurious signal — it finds no signal at all. Training and
validation curves converge to roughly the same near-chance performance. The
constraint is signal-to-noise, not capacity.

## 2. Metaheuristic hyperparameter optimisation (SSA, TLBO)

**Not implemented, deliberately.**

Sparrow Search and Teaching-Learning-Based Optimisation search the
hyperparameter space more thoroughly than a grid. But a wider search over a
fixed validation set **increases** the risk of selecting on noise, and the
existing evidence says the architecture is not binding: LightGBM and the
Transformer land in the same place, and a validation grid over dropout and
learning rate moved nothing.

Adding a metaheuristic search would very likely produce a configuration with a
better validation number and the same test performance. If it *were* added, it
would need its trial count fed into the deflated Sharpe calculation — searching
200 configurations and reporting the best without deflation is precisely how
backtests overstate performance.

**Verdict:** low expected value here, real risk of manufacturing a false positive.

## 3. Sentiment under volatile conditions

**Built and working, disabled on purpose.** `Source/News/` implements NewsAPI
fetching plus FinBERT scoring, and `features.use_sentiment` wires it in as an
extra feature.

**Why it is off:** NewsAPI's free tier serves ~30 days of history. Sentiment
cannot be backfilled to 2007. Training the historical backtest on zero-filled
sentiment would feed the model a feature that did not exist for 99% of the
sample — inventing a variable, which is worse than omitting one.

**To enable honestly** you need a historical news archive with point-in-time
timestamps (RavenPack, GDELT, or a self-maintained forward-collected archive).
Forward collection from today is free and is the recommended path.

## 4. Metrics beyond accuracy

**Fully implemented.** Sortino, Calmar, Sharpe, maximum drawdown, profit factor,
and hit rate all live in `financial_metrics`. Risk-adjusted returns are the
primary reporting frame — the site leads with Sharpe and excess return, not
accuracy.

The critique that papers minimise loss rather than maximise utility is correct
and taken seriously here: a model with 0.5132 AUC is reported as **having no
edge**, because the metric that matters is whether it beats buy-and-hold net of
costs. It does not.

## 5. Decomposition preprocessing (EMD, ICEEMDAN)

**Not implemented, with a specific technical reason.**

EMD and its variants decompose a series into intrinsic mode functions. Applied
naively, **they leak the future**: the decomposition of the full series uses
information from the entire sample, so IMFs at time `t` are computed partly from
data after `t`. Many published results using EMD preprocessing are contaminated
this way, which is a plausible contributor to the strong reported performance.

Doing it correctly requires decomposing **only the trailing window** at each
point in time, recomputed per step — expensive, and it changes the IMF basis at
every timestep.

**Verdict:** worth trying, but only with strictly causal per-window
decomposition, and any result should be treated with suspicion until the leakage
audit passes.

## 6. Data drift (DAELM)

**Partially addressed.** The 8-fold walk-forward validation *measures* drift
directly: mean Sharpe 0.277 with standard deviation 0.364 across folds. The
standard deviation exceeding the mean is quantified evidence of exactly the
regime-instability the literature describes.

There is a concrete, documented instance: the frozen validation-threshold rule
went degenerate on test because the signal's *level* shifted (validation mean
0.00 versus test mean −0.79). The fix was a level-invariant rolling-percentile
rule — a drift-robust design, arrived at empirically.

**Not implemented:** an adaptive/online learner such as DAELM. This is a
reasonable next step, though with no edge present there is nothing yet for
adaptation to preserve.

## 7. Hybrid architectures

**Partially implemented.** `scripts/gbdt_baseline.py` and
`gbdt_cross_section.py` provide LightGBM baselines on both tracks. Result:
LightGBM ≈ Transformer ≈ 0.50.

**Not implemented:** a CNN-BiLSTM-attention stack with an LGBM head. Given that
two structurally very different model families already converge to the same
near-chance result, a third hybrid is unlikely to differ. **When a tree ensemble
and a Transformer agree, the data is speaking, not the architecture.**

## 8. "True multimodal fusion, or merely feature concatenation?"

**This is the sharpest critique in the list, and it lands.**

The macro features here are **concatenated**, not fused. India VIX, USDINR,
crude, and overnight S&P are appended as additional columns in the same input
matrix and processed by the same attention stack. There is no cross-modal
attention, no separate encoder per modality, no learned fusion gate.

Whether genuine fusion would help is untested. It is an honest architectural
gap, and it is now stated as one rather than described as multimodal.

## 9. Alternative data

**Confirmed gap.** No satellite imagery, no earnings-call transcripts, no
macroeconomic index panel. The project uses free daily OHLCV plus four free macro
series.

This is a budget constraint, not an oversight — paid feeds were explicitly out of
scope. It is also, plausibly, **the actual reason for the negative result**:
daily OHLCV is the most heavily mined dataset in finance, and any edge there was
arbitraged away long ago. The genuine gap between academic practice and
deployment that the literature identifies is largely a *data* gap, not a model
gap.

## 10. Look-ahead bias and expanding-window CV

**Addressed beyond the standard.** Expanding-window CV alone does leave leakage
through overlapping labels at fold boundaries. This project adds:

- **Purged** cross-validation — training samples whose labels overlap the test
  window are dropped
- **Embargo** — a gap after each test fold before training resumes
- **Uniqueness weights** — samples weighted 1/h to reflect overlap

Plus the seven leakage rules in [Data](03-data.md#leakage-rules), each with a
test.

## 11. Survivorship bias

**Acknowledged, not fixed, and stated plainly in the artifacts.**

The 85-name universe is today's large caps. Companies that delisted, were
acquired, or collapsed are absent, biasing cross-sectional returns upward.

Fixing it needs a point-in-time index constituent history — commercially
available, not free. Rather than fake it, the bias is disclosed in
`cross_section.json`'s caveats, on the site, and in this documentation.

This is also why the equal-weight benchmark at +41.6% should itself be read as
optimistic: it holds the same survivors.

## 12. Latency

**Not applicable.** Predictions are for 1–20 *day* horizons on daily closes.
Execution latency is irrelevant at this frequency. It would matter immediately if
the project moved to intraday data — which is the main proposed direction below.

## 13. Single-model reliance

**Addressed.** Three seeds are ensembled; LightGBM baselines are published on
both tracks; four cross-sectional configurations are published with none
cherry-picked; and Diebold-Mariano compares against a naive baseline (p = 0.215).

---

## What would actually be worth doing next

Ranked by expected value, given everything measured here.

### 1. Higher-frequency data — the only change that attacks the real constraint

The binding limit is ~33 independent observations at h=20. Hourly bars would
multiply effective sample size by roughly 6x; 5-minute bars by ~75x. **Every
other item on this list is cosmetic next to this one.**

Cost: paid intraday NSE history. Free sources do not go back far enough.

### 2. Forward-collected sentiment

Start archiving headlines daily now. In 12 months there is a genuine
point-in-time sentiment series with no backfill dishonesty. Free, requires
patience.

### 3. Point-in-time fundamentals

Unlocks the entire quality/value factor family, which has far better-documented
predictive power than price-only direction models. Requires a paid
survivorship-clean source.

### 4. Causal decomposition preprocessing

EMD/ICEEMDAN computed strictly per trailing window. Moderate cost, genuinely
uncertain payoff, and demands a careful leakage audit.

### 5. True multimodal fusion

Separate encoders per modality with cross-attention. Interesting, but the data
gap likely dominates the architecture gap.

### Explicitly not recommended

- **Metaheuristic HPO** — searches harder over a space that is not binding while
  raising false-positive risk
- **More architectures** — two families already agree
- **Retrying until a metric clears a threshold** — this manufactures false
  positives; it was refused during development and should stay refused

## The meta-lesson

The literature reports strong results; this project reports none. The most likely
explanations, in order:

1. **Publication bias** — negative results rarely get published
2. **Leakage** — three separate bugs in *this* project each manufactured apparent
   skill, and all three flattered the result (see
   [Evaluation](06-evaluation.md#three-bugs-that-manufactured-skill)). It is
   reasonable to expect similar bugs in work that does not audit for them
3. **Multiple testing** — 20 horizons, 85 stocks, many configurations; something
   always looks significant uncorrected
4. **No transaction costs** — many papers report gross returns. At 9.58 bps
   round-trip, a marginal edge disappears

The apparatus in this repository exists to make those four failure modes hard to
commit accidentally. That is the contribution — not the alpha, which is absent.
