# 14. Full Evaluation — every track, every number

Consolidated results across all four tracks and both live layers. Every figure is
read from a committed artifact; nothing here is typed by hand.

## The verdict in one table

| Track | Primary metric | Value | Baseline | Significant? |
|-------|----------------|-------|----------|--------------|
| **Index, daily** | mean AUC | **0.5033** | 0.50 | **No** — 0/20 |
| **Index, frozen model** | mean AUC | **0.5132** | 0.50 | **No** — 0/20 over 675 OOS days |
| **Intraday, hourly** | mean AUC | **0.5031** | 0.50 | **No** — 0/20 |
| **Cross-section** | pooled IC | **−0.0260** | 0.00 | **No** — and wrong-signed |

Four independent experiments, four null results. That consistency is itself
informative: it is what a genuine absence of signal looks like, as opposed to one
unlucky configuration.

## Track 1 — Index, daily bars

| Metric | Value |
|--------|-------|
| Mean AUC | 0.5033 |
| Mean IC (Spearman) | +0.0210 |
| Net Sharpe | **+0.630**, 95% CI **[−0.58, 1.94]** |
| Total return | +11.2% |
| Buy-and-hold | **+30.2%** |
| Walk-forward Sharpe | **+0.277 ± 0.364** over 8 folds |

Two things kill this. The Sharpe confidence interval **spans zero**, so the point
estimate is not distinguishable from no skill. And the walk-forward standard
deviation **exceeds its mean**, meaning fold-to-fold variation is larger than the
effect — no stable edge across regimes.

The strategy also loses to simply holding the index by 19 percentage points.

## Track 2 — Index, frozen model on its own OOS period

Measured on the frozen paper model's own 675 out-of-sample days, not borrowed
from the backtest model.

| Horizon | P(up) | AUC | 95% CI | eff n | Actionable |
|---------|-------|-----|--------|-------|------------|
| 1 | 49.1% | 0.446 | [0.402, 0.489] | 673 | no |
| 7 | 68.0% | 0.471 | [0.354, 0.589] | 95 | no |
| 10 | 69.7% | 0.483 | [0.342, 0.624] | 66 | no |
| 13 | 59.8% | 0.584 | [0.426, 0.741] | 51 | no |
| 20 | 62.7% | 0.553 | [0.349, 0.758] | 33 | no |

**Mean AUC 0.5132, 0 of 20 horizons actionable.**

The h=10 row is the one to study. A calibrated **69.7% P(up)** looks like a strong
call. Its AUC interval is **[0.342, 0.624]** — it spans 0.50, so there is no
evidence the model ranks better than chance at that horizon. A confident-looking
probability with no evidence behind it is precisely what this apparatus exists to
expose.

## Track 3 — Intraday, hourly bars

| Metric | Value |
|--------|-------|
| Mean AUC | 0.5031 |
| Significant horizons | 0/20 |
| Effective n at primary horizon | **37** (vs 32 daily) |
| Typical 20-bar move | 178 bps |
| Round-trip cost | 9.58 bps |
| **Break-even win rate** | **52.7%** |

**A correction that matters:** this track was projected at eff n ≈ 253. That
figure assumed the whole 730-day history as test data; a proper 70/15/15 split
leaves 738 test windows and **eff n = 37**. The statistical power barely moved,
because the binding constraint was never bar size — it is that Yahoo caps hourly
history at 730 days.

The economics are the healthy part: costs are only 5% of the typical move, so
break-even is 52.7%. **If an edge existed it would be tradable.** None was found.

## Track 4 — Cross-section, 85 NSE names

| Metric | Value | Reading |
|--------|-------|---------|
| Pooled rank IC | **−0.0260** | slightly **backwards** |
| Pooled AUC h=20 | 0.4876 | below coin flip |
| Quintile fwd-20 returns | 1.38 / 1.27 / 1.22 / 0.93 / **0.92** % | **inverted** |
| Long/short spread | **−20.4%** | loses money |
| Long-only basket | +19.1% | |
| Equal-weight benchmark | **+41.6%** | beats every model basket |

The model ranks stocks slightly the *wrong* way, and equal-weighting the universe
beats every model-constructed basket.

**The trap here is obvious and is deliberately not taken:** flipping the signal
because test IC is negative would be look-ahead bias — you only know the sign
*from the test set*. It is left unflipped.

## Live layer — paper trading

676 out-of-sample days from a model frozen before any of them.

| Metric | Value |
|--------|-------|
| Paper return | **+18.5%** |
| Buy-and-hold | **+23.8%** |
| **Excess** | **−5.2%** |
| Sharpe | 0.69 |
| Round trips | 13 |
| Time in market | 40% |

**Sharpe 0.69 is beta, not alpha.** The book is long 40% of the time during a
rising market; excess return is negative. When a long/flat strategy shows a
positive Sharpe in a bull market, the default assumption should be exposure —
check the excess.

## Live layer — trade journal

| Metric | Value |
|--------|-------|
| Closed trades | 6 |
| Hit rate | **66.7%** |
| Exact binomial p | **0.688** |
| Significant? | **No** |
| Categories | 4 win · 1 noise · 1 signal_error |

A 67% hit rate looks like skill. On 6 trades it is entirely consistent with a coin
flip, and the journal says so rather than celebrating it. The test is an **exact**
binomial — the normal approximation is badly wrong at this n, which is exactly
where an over-confident p-value would promote noise to a lesson.

**Bandit:** the leader is best with probability **0.41** against a **0.12**
coin-flip baseline. Nothing is distinguishable; the argmax is noise.

## Live layer — drift

| Detector | Alarms | Over |
|----------|--------|------|
| ADWIN | 11 | 926 OOS observations |
| Page-Hinkley | 1 | 926 |

Both were calibrated empirically to **zero false alarms** on stationary N(0,1)
input while still catching a 3σ shift 40/40. The largest alarms are real level
shifts (2023-12-19: −1.03 → −3.43), which vindicates the past-only rolling
threshold — a fixed threshold went degenerate on exactly this behaviour.

**Champion:** `v1-frozen`, 1 trial. The quarterly gate has **refused every
challenger**, correctly: a 126-day evaluation block carries 6.3 independent
observations, below the 30 minimum, so no AUC difference measured there is
distinguishable from noise.

## Why: the binding constraint

| Setup | eff n | AUC needed to detect |
|-------|-------|----------------------|
| Daily, 20-day horizon | 32 | **0.686** |
| Hourly, 20-bar horizon | 37 | 0.675 |
| Hourly, full history as test | 253 | 0.571 |
| 5-min, 1 year forward-collected | 950 | **0.537** |

*A realistic honest edge in index direction is **0.52–0.55**.*

Every configuration currently available needs an effect **larger than any
plausible real edge** to reach significance. The experiment cannot resolve what
it is looking for. That is a sample-size problem, and no model, feature, or
retraining schedule fixes it.

## Frequency and horizon must move together

The obvious fix — "get finer data" — fails on its own:

| Setup | eff n |
|-------|-------|
| Daily bars, 20-**day** horizon | 32 |
| Hourly bars, 20-**day** horizon | **5 — worse** |
| Hourly bars, 20-**bar** horizon | 37 |

Sampling finer does not create more independent 20-day periods; the calendar span
bounds that. And costs push the other way:

| Bar | Hold | Cost / move | Win rate to break even |
|-----|------|-------------|------------------------|
| 5-min | 1 bar | **64%** | **82.0%** — impossible |
| 5-min | 20 bars | 14% | 57.1% |
| 1-hour | 20 bars | 4% | **52.1%** |
| 1-day | 20 bars | 2% | 50.8% |

More data at 5-minute bars buys statistical power and spends it on transaction
costs, ending up **harder** than daily.

## Baselines and specification tests

| Test | Result |
|------|--------|
| LightGBM, index | ≈ 0.50 |
| LightGBM, cross-section | ≈ 0 |
| Diebold-Mariano vs naive | **p = 0.215** — not distinguishable |
| Deflated Sharpe | 0.916 |

**Architecture is not the bottleneck.** A gradient-boosted tree and a Transformer
land in the same place, which is what happens when signal-to-noise binds rather
than model capacity.

## Everything tried that did not work

| Lever | Outcome |
|-------|---------|
| Better model (LightGBM, purged CV) | ≈ 0.50 |
| More data (86k panel rows, 28×) | no improvement |
| More features (8 macro series) | no improvement |
| Cross-sectional features | moved metrics right, still noise |
| Architecture tuning (validation grid) | no improvement |
| Regression objective (Huber) | **worse** — IC −0.012 |
| Positioning data | **worse** |
| Seed ensembling | reduced variance, not bias |
| Conviction gating | no significant edge |
| Volatility targeting | changed risk, not skill |
| Hourly frequency | **0.5031 — same finding** |

## Three bugs that manufactured skill

Each produced apparent edge that was not real. Each is now a regression test.

1. **The i.i.d. AUC standard error** — used raw `n`, reported 2 horizons as
   Bonferroni-significant. Corrected for overlap: **0 of 20**.
2. **The Spearman IC p-value** — same i.i.d. error in a different metric. Fed
   into multiple-testing it reported **4 of 20 actionable**. Correct answer: 0.
3. **Error bars from the wrong model** — `predictions.json` paired frozen-model
   probabilities with backtest-model AUCs, describing a predictor never measured.

**All three flattered the result.** Bugs that flatter you are the ones you do not
go looking for, which is why every statistical helper here now has a test
asserting the conservative answer.

A fourth, caught by the design itself: the champion/challenger gate promoted a
challenger on +0.0786 mean AUC — against a 0.25 standard error, i.e. **0.3σ**,
with the challenger actually *worse* on the traded horizon.

## The h=1 anti-signal: a false positive, investigated and rejected

Worth recording in full, because it is the most tempting result the project has
produced and every step of the temptation is instructive.

**The observation.** At h=1 the frozen model scores **AUC 0.4430** — 2.6 standard
errors *below* chance, two-sided **p = 0.0102** on 675 OOS days. And h=1 is the
**only horizon with zero forward-label overlap**, so its effective n is 675
rather than ~33. Clean statistics, large sample, nominally significant.

**Why the original test could not see it.** `auc_pvalue` was **one-sided**
(H1: AUC > 0.5). Under it, h=1 reports **p = 0.9949** — maximally insignificant.
A one-sided test is structurally blind to anti-skill: a model ranking *backwards*
is detecting real structure, and reporting that as "no information" answers the
wrong question. **This has been fixed — the test is now two-sided by default.**

**Why it is still not a finding.** Four independent reasons, any one sufficient:

| Check | Result |
|-------|--------|
| Multiple testing within the family of 20 | **0 survive** Bonferroni (threshold 0.0025) or BH |
| Sub-period stability | first half p = **0.0021**, second half p = **0.4054** |
| By year | 2024 p=0.073, 2025 p=0.392, 2026 p=0.331 — decays to noise |
| Effect magnitude | 5.7pp AUC deviation on a deep, continuously-arbitraged index future is implausibly large |

The "h=1 is a single pre-registered test" argument is **post-hoc**. It is true
that h=1 is structurally special, but it was never designated as the primary
hypothesis before the result appeared — and large effective n mechanically
produces the smallest p-value in a family, which is precisely what makes
selecting it after the fact illegitimate.

**Two errors in the first analysis of it**, both caught in review and both
instances of the exact bug classes this project exists to prevent:

1. **Look-ahead in the threshold.** The paired trading result used the
   *full-sample median* of the OOS window as the live decision threshold. Under
   defensible causal rules the Sharpe ranges from **−0.61** (sign of the logit)
   to **+1.46** (rolling 60-day median). A headline number that moves that much
   on an undisclosed choice is not a number.
2. **Effective-n mismatch in the Sharpe.** The position flips only **41 times in
   675 days**, so there are ~42 independent bets, not 675. Annualising daily
   overlapping returns inflated it. Corrected, the deflated Sharpe falls to
   **0.06–0.26** against this project's real trial count.

**Verdict: artifact.** The shipped artifacts continue to report **0 of 20
actionable**, and a regression test now asserts that any horizon promoted to
actionable must have a confidence interval strictly above 0.50.

## What would count as real

| Requirement | Threshold |
|-------------|-----------|
| Single horizon significant | AUC ≈ 0.70 at h=20 |
| Surviving Bonferroni across 20 | AUC ≈ 0.80 |
| Realistic honest edge | **0.52–0.55** |

The gap between the last row and the first two **is the entire problem**.
Retrying models until one clears a bar does not produce an edge — it produces a
false positive, which is why that approach was refused when requested.

## How to read all of this

**Daily Nifty 50 direction is not predictable from daily OHLCV, these macro
series, and news tone, at this sample size.**

That is a real and useful finding. It is not a claim that markets are efficient,
nor that the approach fails at higher frequency or with better data. It is a
statement about what this dataset can support — and the apparatus is now good
enough to say so with confidence rather than hope.

Back to the [index](README.md).
