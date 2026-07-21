# 11. Trade Journal, Bandit, and LLM Commentary

Answering "monitor the P&L, learn from mistakes, don't repeat them" — and an
honest account of why the obvious implementation would make things worse.

## Why not reinforcement learning

The intuitive design is RL: watch the P&L, punish losers, stop repeating them.
Two problems, and the second is the one that matters.

### 1. The sample size is three to five orders of magnitude short

| Algorithm | Decisions needed | This book has | Shortfall |
|-----------|------------------|---------------|-----------|
| Tabular Q-learning | ~10,000 | 13 round trips | **769×** |
| DQN | ~1,000,000 | 13 round trips | **77,000×** |
| PPO | ~1,000,000 | 13 round trips | **77,000×** |

674 out-of-sample days, 13 round trips. That is the entire training set for a
policy learner.

### 2. The premise is wrong

RL assumes **a loss is a mistake**. In a market where the model has no measured
edge, most losses are *noise realising against you*.

An agent that learns not to repeat a random loss has fitted noise. It will
perform **worse** than the rule it replaced, while producing a convincing
narrative about what it learned. This is the exact failure mode the rest of the
repository is built to prevent — and RL would be the most sophisticated-looking
way yet to commit it.

So the journal implements the parts of the idea the data genuinely supports.

## Part 1 — P&L attribution

`Source/Journal/attribution.py`. Every closed trade is decomposed into a category
based on **the size of the move relative to the signal's resolution**, not on
profit alone:

| Category | Meaning | Actionable? |
|----------|---------|-------------|
| `win` | right direction, beyond the noise floor | — |
| `signal_error` | wrong direction, beyond the noise floor | diagnostic, but noisy |
| `cost_drag` | right direction, fees ate the move | **yes — arithmetic** |
| `noise` | move smaller than one holding-period sigma | **no — not a mistake** |

The noise floor is one standard deviation of the holding-period move. A trade
inside it carries no directional information: its sign is a coin flip, so
counting it as an error and "learning" from it is precisely how a feedback loop
fits randomness.

**Cost drag is the one component that can be acted on immediately.** It is
arithmetic, not inference: those trades had the direction right and lost to fees.
Fewer, longer holds fix it with no model change and no significance test.

### The hit rate must survive a test before it is a lesson

Current journal output:

| Metric | Value |
|--------|-------|
| Closed trades | 6 |
| Hit rate | **66.7%** |
| Exact binomial p | **0.688** |
| Significant? | **No** |

A 67% hit rate looks like skill. With 6 trades it is entirely consistent with a
coin flip. The journal reports the p-value beside it and refuses to call it
learning — because a feedback system that promotes this would start reinforcing
noise immediately.

The test is an **exact** binomial, not the normal approximation, which is badly
wrong at n≈13 — exactly the regime where an over-confident p-value would turn
noise into a lesson.

## Part 2 — a bandit, which is RL sized to the data

`Source/Journal/bandit.py`. Thompson sampling over **fixed, already-validated
rules**. It does not learn a policy; it only allocates between a handful of arms,
so the thing being estimated is a few scalars rather than a network.

Thompson sampling specifically — rather than ε-greedy or UCB — because it carries
its own uncertainty as a posterior. With this little data that *is* the point.

### The guard that matters: `overlap()`

Any bandit eventually names a winner; that is what `argmax` does, evidence or
not. Reporting that winner without reporting posterior overlap is how a bandit
launders noise into a decision.

Current output:

> No arm is distinguishable: the leader (`sign`) is best with only probability
> **0.41** against a **0.12** coin-flip baseline. The argmax here is noise, not a
> decision.

The honest answer is "these rules cannot be told apart yet", and the report says
so in the same breath as naming the leader.

## Part 3 — LLM commentary

`Source/Advisor/`. **Optional, off by default.** With no API key the journal
writes deterministic template text, so CI needs no secret and the site is never
blocked on a paid service.

### Should an LLM give advice here? No.

An LLM narrating a no-edge model's output as "advice" would manufacture
confidence that the statistics do not support. That is actively harmful, and no
amount of prompt engineering fixes it.

An LLM **explaining** the statistics is genuinely useful and structurally safe.
That is what this is.

### The guardrail is structural, not textual

The system prompt forbids advice, but prompts are not a security boundary.
The real guarantee is in `Journal/run.py`:

```python
payload = {
    "closed_trades": j["summary"],
    "bandit_separation": (b.get("separation") or {}).get("verdict"),
    "note": "historical results only; no forward prediction is included",
}
```

**The model is never given a forward prediction.** It has nothing to trade on, so
it cannot emit a trade call even if an injected instruction asked it to. A test
asserts the published commentary contains no forward-looking content.

### The rest of the guardrails

| Layer | Implementation |
|-------|----------------|
| Input | 12,000-char cap; regex screening for injection patterns; refuses rather than sanitises |
| Output | Caller-supplied schema validation, one retry with the error fed back, then deterministic fallback — never a partial artifact |
| Failure | Explicit 30s timeout; bounded exponential backoff on 429/5xx |
| Cost | `max_tokens` always set (900); retries bounded at 3; token usage returned and logged |
| Provider | Abstracted (`groq` \| `anthropic`); no vendor SDK in business logic |
| Prompts | Versioned in `prompts.py`, never inline; version stamped into the artifact |

A malformed artifact on a site about statistical honesty is worse than no
artifact, which is why validation failure falls back to deterministic text rather
than publishing partial output.

## Running it

```bash
python -m Source.Journal.run          # deterministic commentary, no API key
python -m Source.Journal.run --llm    # needs GROQ_API_KEY or ANTHROPIC_API_KEY
```

Writes `frontend/public/data/journal.json`. The daily CI job runs the
deterministic form.

```yaml
advisor:
  provider: groq        # groq | anthropic
  model: null           # null = provider default
```

## Honest limits

- **Nothing here creates an edge.** Attribution explains realised P&L; it does
  not improve prediction.
- **The only reliable lessons are the deterministic ones.** Cost drag and
  exposure are arithmetic. Direction quality needs a significance test that, at
  this sample size, it currently fails.
- **The bandit has not learned anything and says so.** With arms this close and
  pulls this few, its leader is noise.
- **Commentary is explanation, never advice**, and cannot become advice without
  changing the payload filter — which is the line to defend in review.

Back to the [index](README.md).
