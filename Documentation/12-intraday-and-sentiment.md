# 12. Intraday Track and Working Sentiment

Two additions aimed at the same target: getting enough independent observations
that the experiment can resolve a real edge.

## Why hourly, and why not 5-minute

Two constraints pull in opposite directions.

**Statistical power** wants more observations — sample finer.
**Transaction costs** want fewer, larger moves — hold longer.

Measured against the real India futures round trip (9.58 bps):

| Bar | Hold | Typical move | Cost / move | Win rate to break even |
|-----|------|--------------|-------------|------------------------|
| 5-min | 1 bar | 15 bps | **64%** | **82.0%** — impossible |
| 5-min | 20 bars | 67 bps | 14% | 57.1% |
| 15-min | 20 bars | 116 bps | 8% | 54.1% |
| **1-hour** | **20 bars** | **232 bps** | **4%** | **52.1%** |
| 1-day | 20 bars | 580 bps | 2% | 50.8% |

Sampling finer buys statistical power and spends it on costs. At 5-minute bars
the typical move is 15 bps against a 9.58 bps round trip — costs eat 64% of the
move, so you would need to be right 82% of the time. Even held 20 bars, 5-minute
demands a **57.1%** win rate, which is *harder than daily* despite far more data.

Hourly bars held ~20 bars is the band where both constraints are satisfiable.

## The trap: frequency and horizon must move together

Sampling more finely does **not** create more independent observations at a fixed
economic horizon. The number of independent 20-day periods in three years is ~37
however finely you slice it.

| Setup | Effective n |
|-------|-------------|
| Daily bars, 20-**day** horizon | 32 |
| Hourly bars, 20-**day** horizon (=140 bars) | **5 — worse** |
| Hourly bars, 20-**bar** horizon (~4 days) | 37 |

Switching to hourly while keeping a 20-day horizon makes the statistics *worse*,
not better. The horizon has to shrink with the bar size.

## What it actually delivered — and a correction

The initial projection for the hourly track was **eff n ≈ 253**. That figure
assumed the entire 730-day history as test data, which is not achievable: a
proper 70/15/15 split leaves 738 test windows.

| | Daily track | Hourly track |
|---|---|---|
| Mean AUC | 0.5033 | **0.5031** |
| Horizons significant (BH) | 0/20 | **0/20** |
| Effective n at primary horizon | 32 | **37** |
| Break-even win rate | 50.8% | 52.7% |

**Same finding, and the statistical power barely moved.** The binding constraint
was never the bar size — it is that Yahoo caps hourly history at 730 days. More
power requires more *history* at hourly frequency, which only forward collection
provides.

The economics are healthier than expected: a typical 20-bar move is 178 bps
against 9.58 bps of cost, so break-even sits at 52.7%. If an edge existed it
would be tradable. None was found.

## Intraday features

The architecture is unchanged — LightGBM and the Transformer already agree at
daily, so the model was never the constraint. What changes is the **input**.

| Feature | Why it only exists intraday |
|---------|------------------------------|
| `gap`, `gap_z` | ~17 hours of world news lands between the 15:30 close and 09:15 open. Averaged away in a daily bar. |
| `bar_of_day`, `minutes_from_open` | Opening auction imbalance and closing rebalance flow behave differently from mid-session. |
| `range_pos` | Where the bar closed inside its own high/low — a standard pressure proxy. |
| `rel_volume` | Volume against the **same time of day** historically, not a flat mean. |
| `close_vs_vwap` | Deviation from session VWAP, the reference most execution algorithms track. |

`rel_volume` deserves a note: intraday volume is strongly U-shaped, so
normalising against a flat mean would flag every open and close as anomalous and
smuggle a pure time-of-day signal into the model. It is normalised per
time-of-day bucket, using an expanding mean that is shifted so the current bar
never contributes to its own baseline.

A test asserts causality directly: perturbing a **future** bar must not change
any earlier feature row.

## Sentiment that can actually be backfilled

The existing NewsAPI path could never add value, for a reason no amount of
engineering fixes: the free tier serves ~30 days, so a feature built from it
exists for 30 of ~4,600 training days. Training on that zero-fills 99% of history
— inventing a feature rather than adding one. It stays off.

**GDELT DOC 2.0** solves exactly that gap:

| | NewsAPI | GDELT |
|---|---------|-------|
| Cost | free tier | free |
| API key | required | **none** |
| History | ~30 days | **2017 onward** |
| Resolution | article | 15-minute tone series |

`Source/News/gdelt.py` fetches an average-tone series and merges it onto bars
with a strict backward lag.

### Two operational quirks, both handled

1. **Aggressive rate limiting** (HTTP 429) with no documented budget — every call
   goes through bounded exponential backoff.
2. **OR'd query terms must be parenthesised.** An unparenthesised query returns
   **HTTP 200 with a plain-text error body**, so a malformed query looks like a
   successful empty result. The client raises instead, and a test asserts the
   default query is parenthesised.

### Leakage rule

```python
merged = pd.merge_asof(bars, tone, direction="backward")   # never look forward
b["news_tone"] = merged["tone"].shift(lag_bars)            # and one bar further back
```

Both steps are mandatory. GDELT timestamps are *publication* times, so a story
published at 10:00 can describe a 09:50 move — without the shift, a bar would see
news published inside its own interval. A test injects a tone spike and asserts
it appears only on the *following* bar.

## Running it

```bash
# hourly bars (free; 730d is Yahoo's cap for 1h)
python -m Source.Intraday.fetch --interval 1h --period 730d

# historical news tone (free, no key; rate-limited so it takes a few minutes)
python -m Source.News.gdelt --days 730

# train + evaluate, same encoder and same honesty apparatus
python -m Source.Intraday.run --interval 1h --horizons 20
python -m Source.Intraday.run --interval 1h --horizons 20 --sentiment
```

## Plugging in a different data source

`Source/Intraday/fetch.py` keeps sources behind one interface:

```python
SOURCES = {"yfinance": from_yfinance, "csv": from_csv}
```

Any provider — a broker API, or an NSE site publishing bars — that can produce
`datetime, open, high, low, close, volume` flows through unchanged:

```bash
python -m Source.Intraday.fetch --source csv --csv path/to/bars.csv
```

## Honest limits

- **The hourly track found no edge either.** Mean AUC 0.5031, 0/20 significant.
- **The power gain was marginal** (eff n 37 vs 32), because 730 days of history
  is the real cap. Forward collection is the only fix that does not cost money.
- **Sentiment is now fusable, not proven useful.** Being able to backfill it is a
  precondition for testing it, not evidence it helps.
- **Live data feeds solve execution, not research.** You cannot backtest on data
  you have not collected. Find an edge on history first; buy the fast feed to
  execute it second.

Back to the [index](README.md).
