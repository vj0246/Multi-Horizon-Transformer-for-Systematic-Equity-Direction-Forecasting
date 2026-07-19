# 8. Instrument Choice

Is `^NSEI` a good instrument for a single-instrument model, and would a single
stock be better?

## First, a correction

**`^NSEI` is not a stock.** It is the Nifty 50 index — a capitalisation-weighted
basket of 50 companies. That distinction is the reason it is the best choice
here, so it matters.

## Measured properties

Computed over 4,619 trading days from the committed data, against the 85-stock
NSE universe for contrast.

| Property | `^NSEI` | Universe median | Universe range |
|----------|---------|-----------------|----------------|
| Annualised volatility | **20.6%** | 35.8% | 25.4% – 72.9% |
| Up-day rate | 53.1% | — | — |
| 20-day windows that rose | **60.1%** | 56.4% | — |
| Return autocorr(1) | +0.0371 | 0.0216 (abs) | up to 0.232 (abs) |
| Sign autocorr(1) | +0.0599 | 0.0107 (abs) | up to 0.062 (abs) |

Calmest individual names for comparison: COLPAL 25.4%, BRITANNIA 25.8%,
HINDUNILVR 26.0%, ASIANPAINT 26.7%, ITC 27.0% — every one still noticeably more
volatile than the index.

## Why the index is the right choice

**1. Lowest volatility available.** At 20.6% it is below every single name in the
universe. Diversification cancels idiosyncratic noise. For a direction model, a
lower noise floor means a given edge is easier to detect and easier to hold.

**2. Strongest and most reliable drift.** 60.1% of 20-day windows closed higher,
against a 56.4% median for individual stocks. This matters more than it looks —
see the warning below.

**3. Cheapest to trade.** Nifty futures cost **9.58 bps** round-trip in this
project's cost model, versus **28.22 bps** for delivery equity. That is a 3x cost
advantage, and cost is a certainty while edge is a hypothesis.

**4. No single-name catastrophe risk.** No fraud, no earnings gap, no regulatory
action, no promoter pledge unwinding. An index cannot go to zero overnight.

**5. Deepest liquidity.** Nifty futures are among the most liquid contracts in
the world. Slippage assumptions are realistic rather than optimistic.

**6. Least survivorship bias.** The index handles its own reconstitution. A
single stock chosen today is chosen *because* it survived — that selection is
invisible and unfixable in the backtest.

**Verdict: `^NSEI` is the correct instrument choice.** For a single-instrument
direction model on free daily Indian data, there is no better option available.

## But the instrument is not the problem

A good instrument does not create an edge. On this well-chosen instrument, with
every leakage control in place:

- Mean OOS AUC **0.5123**, coin flip is 0.50
- **0 of 20** horizons significant after multiple-testing correction
- Paper trading **+19.2%** vs buy-and-hold **+24.5%** over 674 forward days

**The model does not beat holding the index.** Choosing the right instrument was
necessary, not sufficient.

## The drift trap

That 60.1% up-rate is the single most seductive number on this page. It is a real
edge — and it is **fully capturable by buying and holding**. It requires no
model, no GPU, and pays no transaction costs beyond one entry.

Any long/flat timing strategy that is in the market part of the time inherits a
*fraction* of that drift and pays costs on every switch. That is exactly what the
paper book shows: Sharpe 0.72 looks respectable, but it is 40%-of-the-time
exposure to a rising market, and excess return is **−5.3%**.

**When you see a positive Sharpe on a long/flat strategy in a bull market, the
default assumption should be beta, not alpha.** Check the excess return.

## Would a single stock be better?

**No — and selecting one from this project's output would be actively harmful.**

### The cross-sectional evidence

The model was run across all 85 names specifically to test relative ranking:

| Measure | Result |
|---------|--------|
| Pooled rank IC | **−0.026** (backwards) |
| Quintile forward returns | **inverted** — 1.38 / 1.27 / 1.22 / 0.93 / 0.92 % |
| Long/short spread | **−20.4%** |
| Equal-weight benchmark | **+41.6%** |

The model ranks stocks slightly *wrong*. Picking "the stock the model likes most"
selects from a ranking with no demonstrated validity.

### The selection-bias trap

Suppose you ignore the above and instead rank all 85 stocks by backtest AUC and
pick the winner. **That is data snooping**, and it is the most common way
retail-facing backtests lie.

With 85 candidates tested at α=0.05, roughly **4 will look significant by pure
chance**. The best of 85 will show an impressive AUC even if every stock is a
coin flip. That is what the Bonferroni and Benjamini-Hochberg machinery in
`Source/Evaluation/suite.py` exists to prevent, and it is the same reason a
request to loop until some horizon exceeded AUC 0.58 was refused during
development: searching until a metric clears a bar manufactures a false positive
rather than discovering an edge.

Note also that the highest raw return autocorrelation in the universe belongs to
DABUR at **−0.232** — strongly *negative*, i.e. mean-reverting, and almost
certainly reflecting bid-ask bounce and stale pricing rather than a tradable
pattern. Microstructure artifacts are not alpha.

### If you still want a single stock

Then the honest ranking criterion is **tradability, not backtest performance**:

1. Liquidity — tight spreads, deep book, F&O availability
2. Low volatility — the noise floor you must beat
3. No single-name event risk you cannot model
4. Cost — futures over delivery where available

By those criteria the large-cap consumer names (HINDUNILVR, ITC, ASIANPAINT,
BRITANNIA, COLPAL) are the calmest. **But this project provides no evidence the
model has skill on any of them individually**, and the cross-sectional result
suggests the opposite.

## The bottom line

| Question | Answer |
|----------|--------|
| Is `^NSEI` a good instrument? | **Yes — the best available here.** Lowest vol, strongest drift, cheapest, most liquid, no blowup risk |
| Does that make the model tradable? | **No.** AUC 0.5123, 0/20 significant, loses to buy-and-hold |
| Should you switch to a single stock? | **No.** Higher vol, higher cost, event risk, and the model ranks stocks backwards |
| Should you trade any of this? | **No.** There is no validated edge to trade |
| What does the 60.1% drift mean? | Buy and hold captures it. A model is not needed and here subtracts value |

If the goal is exposure to Indian equities, the evidence in this repository
points at a low-cost index fund, not this model. The project's value is the
measurement apparatus and the honest negative result — not a signal to trade.

Continue to [Research Gaps](09-research-gaps.md).
