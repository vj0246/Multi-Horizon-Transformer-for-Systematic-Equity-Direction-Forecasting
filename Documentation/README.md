# Documentation

Complete guide to the Multi-Horizon Transformer project. Written so that someone
who has never seen this repository can understand every file, reproduce every
number, and correctly interpret what the results do and do not show.

Read in order if you are new:

| # | Document | What it answers |
|---|----------|-----------------|
| 1 | [Overview](01-overview.md) | What this project is, what it found, why the finding is negative |
| 2 | [Getting Started](02-getting-started.md) | Install, run, regenerate every artifact |
| 3 | [Data](03-data.md) | Every data source, schema, and the leakage rules that govern it |
| 4 | [Architecture](04-architecture.md) | Pipeline and model diagrams, layer by layer |
| 5 | [File Reference](05-file-reference.md) | Every file in the repo, what it does, how to run it |
| 6 | [Evaluation](06-evaluation.md) | Every metric, its formula, and why it is used |
| 7 | [Results](07-results.md) | The actual numbers, with confidence intervals |
| 8 | [Instrument Choice](08-instrument-choice.md) | Why `^NSEI`, and whether a single stock is better |
| 9 | [Research Gaps](09-research-gaps.md) | Literature critique mapped to what is and is not implemented |
| 10 | [Adaptive Retraining](10-adaptive-retraining.md) | Drift detection, versioning, and the champion/challenger gate |
| 11 | [Journal & Advisor](11-journal-and-advisor.md) | P&L attribution, why not RL, the bandit, and the LLM guardrails |

## The one-paragraph summary

A Transformer encoder predicts whether the Nifty 50 index will close higher over
each of 20 forward horizons (1 to 20 days). It is trained on ~4,500 trading days
of engineered price, volume, and macro features with a strict temporal split.
**The model has no statistically detectable edge.** Mean out-of-sample AUC is
0.5123 against a 0.50 coin-flip baseline, and zero of 20 horizons survive
multiple-testing correction. The system is published as an honest negative
result, with a live paper-trading book that demonstrates the same conclusion
forward on real prices. Nothing here should be traded.

## Why a negative result is the deliverable

Daily index direction is close to unpredictable from price history alone. That is
the expected outcome, not a failure of implementation. The value of this project
is the measurement apparatus: leakage audits, overlap-corrected confidence
intervals, deflated Sharpe ratios, and multiple-testing correction that together
make it very hard to fool yourself into seeing an edge that is not there.

Three bugs found during development each *manufactured* apparent skill, and each
is now a regression test. They are documented in
[Evaluation](06-evaluation.md#three-bugs-that-manufactured-skill) because they
are the most transferable lesson in the repository.
