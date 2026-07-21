import { data, fmtNum, fmtPct, fmtSigned } from "@/lib/data";
import { Callout, Panel, Section, Stat } from "@/components/ui";
import {
  AttentionChart,
  CalibrationChart,
  CSEquity,
  CSICSeries,
  CSQuintiles,
  DecileChart,
  DriftTimeline,
  EquityCurve,
  HorizonAUC,
  HorizonIC,
  PaperEquity,
  PredictionTable,
  PriceChart,
  SharpeExplorer,
  StockSignals,
  ThresholdSweep,
  TrainingHistory,
  YearlyChart,
} from "@/components/charts";

const s = data.summary;

const NAV = [
  ["overview", "Overview"],
  ["paper", "Paper Trading"],
  ["predictions", "Predictions"],
  ["architecture", "Architecture"],
  ["results", "Results"],
  ["backtest", "Backtest"],
  ["explorer", "Sharpe Explorer"],
  ["walkforward", "Walk-Forward"],
  ["crosssection", "Cross-Section"],
  ["signals", "Stock Signals"],
  ["attention", "Attention"],
  ["journal", "Journal"],
  ["adaptive", "Adaptive"],
  ["docs", "Docs"],
  ["method", "Method"],
];

export default function Page() {
  const prim = data.strategies.timing_ensemble;
  const timingH20 = data.strategies.timing_h20;
  const timingBest = data.strategies.timing_best;
  const quantile = data.strategies.quantile;
  const sign = data.strategies.sign;
  const bh = data.strategies.buy_and_hold;
  const cb = s.cost_breakdown;
  const cs2 = data.crossSection;
  const [ciLo, ciHi] = s.sharpe_ci95 ?? [NaN, NaN];
  const meanAuc = s.mean_auc ?? 0;
  const meanIc = s.mean_ic ?? 0;

  return (
    <main className="min-h-screen">
      {/* Nav */}
      <header className="sticky top-0 z-30 border-b border-edge/60 bg-ink/70 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
          <div className="text-sm font-semibold text-white">
            MHT<span className="text-accent">·</span>Nifty50
          </div>
          <nav className="hidden gap-5 text-xs text-muted md:flex">
            {NAV.map(([id, label]) => (
              <a key={id} href={`#${id}`} className="hover:text-white">
                {label}
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <a
              href="https://github.com/vj0246/Multi-Horizon-Transformer-for-Systematic-Equity-Direction-Forecasting/tree/main/Documentation"
              className="hidden rounded-md border border-edge px-3 py-1 text-xs text-muted transition-colors hover:border-edge2 hover:text-white sm:block"
            >
              Docs
            </a>
            <a
              href="https://github.com/vj0246/Multi-Horizon-Transformer-for-Systematic-Equity-Direction-Forecasting"
              className="rounded-md border border-edge px-3 py-1 text-xs text-muted transition-colors hover:border-edge2 hover:text-white"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Hero */}
      <div className="grid-lines border-b border-edge/50">
        <div className="mx-auto max-w-6xl px-5 py-16 md:py-24">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-edge bg-panel px-3 py-1 text-xs text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            {s.date_start} → {s.date_end} · {s.n_trading_days.toLocaleString()} trading days
          </div>
          <h1 className="max-w-3xl text-3xl font-bold leading-tight tracking-tight text-white md:text-5xl">
            Multi-Horizon Transformer for{" "}
            <span className="bg-gradient-to-r from-accent to-accent2 bg-clip-text text-transparent">
              Nifty 50
            </span>{" "}
            Direction Forecasting
          </h1>
          <p className="mt-4 max-w-2xl text-base font-medium text-white/90 md:text-lg">
            Multi-horizon Transformer for Nifty 50 direction, with rigorous evidence it
            has no edge.
          </p>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted md:text-base">
            A single Transformer encoder predicts whether the Nifty 50 index will close
            higher — simultaneously across {s.horizons} forward horizons (1 to {s.horizons}{" "}
            days). Trained on {s.n_features} engineered features over{" "}
            {s.n_samples.toLocaleString()} sequences with a strict temporal split. Every
            number below comes from a real trained model on held-out test data.
          </p>

          <div className="mt-6 flex flex-wrap gap-2 text-xs">
            {[
              ["Transformer encoder", "2 blocks · 4 heads · d64"],
              ["No look-ahead", "8 audited leakage rules"],
              ["Real India costs", `${s.roundtrip_cost_bps.toFixed(2)}bps round-trip`],
              ["Live paper trading", "updated every weekday"],
            ].map(([k, v]) => (
              <span
                key={k}
                className="rounded-full border border-edge bg-panel/70 px-3 py-1 text-muted"
              >
                <span className="text-white">{k}</span> · {v}
              </span>
            ))}
          </div>

          <div className="mt-9 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat
              label="Mean AUC (test)"
              value={fmtNum(meanAuc, 4)}
              sub="20-horizon average · 0.5 = coin flip"
              tone={meanAuc >= 0.5 ? "good" : "bad"}
            />
            <Stat
              label="Mean IC (Spearman)"
              value={fmtSigned(meanIc, 4)}
              sub="signal vs realized return"
              tone={meanIc >= 0 ? "good" : "bad"}
            />
            <Stat
              label="Timing Sharpe (net)"
              value={fmtSigned(prim.sharpe_net, 2)}
              sub={`95% CI [${ciLo.toFixed(2)}, ${ciHi.toFixed(2)}] · ${s.roundtrip_cost_bps.toFixed(1)}bps round-trip`}
              tone={prim.sharpe_net >= 0 ? "good" : "bad"}
            />
            <Stat
              label="Max Drawdown"
              value={fmtPct(prim.max_drawdown, 1)}
              sub="long/flat timing equity"
              tone="default"
            />
          </div>

          <div className="mt-6 max-w-3xl">
            <Callout tone="warn" title="The honest verdict — read this first">
              This model has <span className="text-white">no statistically detectable edge</span>.
              Mean AUC is {fmtNum(meanAuc, 4)} against a 0.50 coin flip, and{" "}
              <span className="text-white">
                {data.predictions.verdict.n_actionable} of {data.predictions.verdict.n_horizons}
              </span>{" "}
              horizons survive multiple-testing correction. The site is published as a
              negative result: the contribution is the measurement apparatus — leakage
              audits, overlap-corrected intervals, deflated Sharpe — not alpha. Nothing
              here should be traded.
            </Callout>
          </div>
        </div>
      </div>

      {/* Overview */}
      <Section id="overview" eyebrow="Overview" title="What this system does">
        <div className="grid gap-4 md:grid-cols-3">
          <Panel title="One model, 20 answers">
            <p className="text-sm leading-relaxed text-muted">
              A shared Transformer encoder outputs {s.horizons} independent logits — one per
              horizon — so short- and long-range signal inform each other through shared
              gradients instead of {s.horizons} separate models.
            </p>
          </Panel>
          <Panel title="No look-ahead leakage">
            <p className="text-sm leading-relaxed text-muted">
              Chronological {Math.round((s.split.train / s.n_samples) * 100)}/
              {Math.round((s.split.val / s.n_samples) * 100)}/
              {Math.round((s.split.test / s.n_samples) * 100)} split, no shuffling.
              StandardScaler is fit on training data only. Targets are strictly forward-looking.
            </p>
          </Panel>
          <Panel title="Signal, not just accuracy">
            <p className="text-sm leading-relaxed text-muted">
              All 20 horizon logits are ensembled into one signal, Platt-calibrated on
              validation data, and stress-tested as a long/flat timing strategy with
              India costs, bootstrap confidence intervals, decile attribution, and
              walk-forward validation.
            </p>
          </Panel>
        </div>

        <div className="mt-4">
          <Panel title={`${s.ticker} price context`} subtitle={`${s.date_start} → ${s.date_end}`}>
            <PriceChart />
          </Panel>
        </div>
      </Section>

      {/* Paper trading */}
      <Section id="paper" eyebrow="Live" title="Paper trading — the strategy, forward, on real prices">
        {(() => {
          const pt = data.paperTrading;
          const s = pt.summary;
          return (
            <>
              <div className="mb-5 rounded-lg border border-danger/40 bg-danger/5 px-4 py-3 text-xs leading-relaxed text-muted">
                <span className="font-semibold text-danger">Paper trading — simulated, no real money.</span>{" "}
                {pt.disclaimer.replace("PAPER TRADING - simulated, no real money. ", "")}
              </div>
              <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Paper return" value={fmtPct(s.total_return, 1)} sub={`${s.n_days} trading days`} tone={s.total_return >= 0 ? "good" : "bad"} />
                <Stat label="vs Buy & Hold" value={fmtPct(s.excess_return, 1)} sub={`BH ${fmtPct(s.buy_hold_return, 1)}`} tone={s.excess_return >= 0 ? "good" : "bad"} />
                <Stat label="Sharpe" value={fmtSigned(s.sharpe, 2)} sub={`${s.n_trades} round-trips`} tone={s.sharpe >= 0 ? "good" : "bad"} />
                <Stat label="Position now" value={s.current_position} sub={`${fmtPct(s.time_in_market, 0)} time in market`} />
              </div>
              <Panel
                title={`Paper equity (start = 100) vs buy-and-hold`}
                subtitle={`${pt.meta.strategy ?? "primary strategy"} · ${pt.meta.cost_roundtrip_bps}bps round-trip · seeded from ${pt.meta.seeded_from ?? "out-of-sample predictions"} · as of ${pt.as_of}`}
              >
                <PaperEquity />
              </Panel>
              <p className="mt-4 max-w-3xl text-xs leading-relaxed text-muted">
                This trades the exact primary strategy forward, out-of-sample, on real Nifty
                closes with the full India futures cost stack. It is not tuned to look good —
                it under-performs simply holding the index, which is what the no-edge finding
                predicts. That is the point: honest forward proof, updated as new data arrives.
              </p>
            </>
          );
        })()}
      </Section>

      {/* Current predictions */}
      <Section
        id="predictions"
        eyebrow="Predictions"
        title="What the model says today — and what that is worth"
      >
        {(() => {
          const pr = data.predictions;
          const pos = pr.position;
          const v = pr.verdict;
          const long = pos.stance === "LONG";
          // pick the live worked example straight from the artifact — hardcoding a
          // probability here would drift out of sync the next time the cron runs
          const boldest = pr.horizons.reduce((a, b) => (b.prob_up > a.prob_up ? b : a));
          return (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Stance" value={pos.stance} sub={`as of ${pr.as_of}`} tone={long ? "good" : "default"} />
                <Stat label="Signal" value={fmtSigned(pos.signal_z, 2)} sub={`${pos.percentile_of_trailing.toFixed(0)}th pct of ${pos.window_days}d`} />
                <Stat label="Entry threshold" value={fmtSigned(pos.threshold_z, 2)} sub={`${pos.quantile_rule}th-pct rule`} />
                <Stat
                  label="Actionable horizons"
                  value={`${v.n_actionable} / ${v.n_horizons}`}
                  sub="after multiple-testing correction"
                  tone={v.n_actionable > 0 ? "good" : "bad"}
                />
              </div>

              <div className="mb-5 rounded-lg border border-danger/40 bg-danger/5 px-4 py-3 text-xs leading-relaxed text-muted">
                <span className="font-semibold text-danger">Read the intervals, not the probabilities.</span>{" "}
                {v.headline} The boldest row below reads{" "}
                {(boldest.prob_up * 100).toFixed(1)}% P(up) at {boldest.horizon}d, yet its
                AUC interval [{fmtNum(boldest.auc_ci95[0], 3)}, {fmtNum(boldest.auc_ci95[1], 3)}]
                {boldest.actionable ? " clears" : " spans"} 0.50 — a confident-looking number
                with {boldest.actionable ? "measured skill behind it" : "no evidence behind it"}.
                The table is laid out so that is visible at a glance rather than buried.
              </div>

              <Panel
                title={`Forward probabilities across all ${v.n_horizons} horizons`}
                subtitle={`frozen ensemble (${pr.model.n_seeds} seeds, trained through ${pr.model.frozen_through}) · ${pr.model.calibration} · skill measured on this model's own ${v.oos_days_scored} out-of-sample days · last close ${fmtNum(pr.last_close, 0)}`}
              >
                <PredictionTable />
              </Panel>

              <p className="mt-4 max-w-3xl text-xs leading-relaxed text-muted">
                {pos.rationale}. The stance is a mechanical rule, not a conviction
                call: it fires whenever the trailing-quantile condition is met,
                independent of whether any horizon shows measurable skill. Publishing
                both together is deliberate — it shows exactly how much (or how
                little) statistical backing the live position has.
              </p>
            </>
          );
        })()}
      </Section>

      {/* Architecture */}
      <Section id="architecture" eyebrow="Architecture" title="Model architecture">
        <div className="grid gap-4 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <Panel title="Data flow">
              <div className="flex flex-col gap-2 text-xs">
                {[
                  `Input sequence  (${s.lookback} days × ${s.n_features} features)`,
                  `Dense projection → d_model = ${s.model.d_model}`,
                  "＋ Sinusoidal positional encoding",
                  `${s.model.num_layers} × Encoder block  ·  ${s.model.num_heads} heads  ·  FFN ${s.model.ff_dim}`,
                  s.model.pooling === "attention"
                    ? "Attention pooling (learned softmax over the 60 steps)"
                    : "GlobalAveragePooling1D",
                  `Dense(${s.horizons}) → ${s.horizons} raw logits`,
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-edge bg-panel2 text-[10px] text-accent2">
                      {i + 1}
                    </div>
                    <div className="flex-1 rounded-lg border border-edge bg-panel2/60 px-3 py-2 text-muted">
                      {step}
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
          <div className="lg:col-span-2">
            <Panel title="Specification">
              <dl className="divide-y divide-edge/60 text-sm">
                {[
                  ["Lookback window", `${s.lookback} days`],
                  ["Input features", `${s.n_features}`],
                  ["d_model", `${s.model.d_model}`],
                  ["Attention heads", `${s.model.num_heads}`],
                  ["Encoder blocks", `${s.model.num_layers}`],
                  ["FFN hidden dim", `${s.model.ff_dim}`],
                  ["Pooling", s.model.pooling === "attention" ? "Attention (learned)" : "GlobalAverage"],
                  ["Output horizons", `${s.horizons}`],
                  ["Loss", "BCE (from logits)"],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between py-2">
                    <dt className="text-muted">{k}</dt>
                    <dd className="tag text-white">{v}</dd>
                  </div>
                ))}
              </dl>
            </Panel>
          </div>
        </div>

        <div className="mt-4">
          <Panel title="Input features" subtitle={`${data.features.length} stationary signals per day — raw OHLCV price levels are excluded (non-stationary out-of-sample)`}>
            <div className="flex flex-wrap gap-2">
              {data.features.map((f) => (
                <span
                  key={f}
                  className="rounded-md border border-edge bg-panel2/60 px-2.5 py-1 text-xs text-muted"
                >
                  {f}
                </span>
              ))}
            </div>
          </Panel>
        </div>
      </Section>

      {/* Results */}
      <Section id="results" eyebrow="Results" title="Per-horizon predictive skill (test set)"
        lede="AUC measures ranking ability independent of any threshold; 0.50 is a coin flip. Read these beside the confidence intervals in the Predictions section — a bar above 0.50 is not evidence of skill on its own.">
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="ROC-AUC by horizon" subtitle="teal = above the 0.50 coin-flip baseline · rose = below">
            <HorizonAUC />
          </Panel>
          <Panel title="Information Coefficient by horizon" subtitle="Spearman rank corr: signal vs realized forward return">
            <HorizonIC />
          </Panel>
        </div>
        <p className="mt-4 max-w-3xl text-xs leading-relaxed text-muted">
          Daily index direction is close to efficient — AUCs hover near 0.5 and IC is small,
          as expected for a liquid benchmark. The value is in the aggregate ranking of the
          signal, tested next in a cost-aware backtest.
        </p>

        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <Panel
            title={`Probability calibration · horizon ${data.calibration.horizon}`}
            subtitle="reliability diagram on test — closer to the diagonal is better"
          >
            <CalibrationChart />
          </Panel>
          <Panel title="Why calibrate?">
            <p className="text-sm leading-relaxed text-muted">
              The model is trained for classification, so its raw sigmoid outputs rank
              market states well but are not trustworthy probabilities. A Platt scaler is
              fit per horizon on the <span className="text-white">validation set only</span>{" "}
              and applied to test predictions, so &quot;P(up) = 0.6&quot; means the market
              actually rose about 60% of the time at that score. All probability
              thresholds on this page use the calibrated values; rank metrics (AUC, IC)
              are unaffected.
            </p>
          </Panel>
        </div>
      </Section>

      {/* Backtest */}
      <Section id="backtest" eyebrow="Backtest" title="Long/flat timing backtest — calibrated 20-horizon ensemble"
        lede="All 20 logits are z-scored on validation statistics and averaged into one signal. The strategy goes long when that signal clears a trailing percentile, and is charged the full India futures cost stack on every switch.">
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-muted">
          On a single index, long-short quantile spreads are a cross-sectional idea that
          does not transfer; the honest framing is <span className="text-white">market
          timing</span>: hold the index when the ensemble signal is in its top{" "}
          {100 - 70}% (threshold fixed on validation data), sit in cash otherwise.
        </p>
        {s.threshold_rule_note && (
          <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-500/5 px-4 py-3 text-xs leading-relaxed text-muted">
            <span className="font-semibold text-amber-400">Selection caveat (read this):</span>{" "}
            {s.threshold_rule_note}
          </div>
        )}
        {prim.avg_exposure === 0 && (
          <div className="mb-4 rounded-lg border border-edge bg-panel2/60 px-4 py-3 text-xs leading-relaxed text-muted">
            <span className="text-white">Honest out-of-sample outcome:</span> on this run the
            ensemble signal never crossed its validation-fixed entry threshold during the test
            window, so the disciplined strategy <span className="text-white">stayed entirely in
            cash</span> — zero trades, zero Sharpe. Not a bug: with a threshold fixed on
            past data and only {prim.n_trades} non-overlapping test periods, &quot;never good
            enough to trade&quot; is a legitimate result, and a truer verdict on the edge than
            forcing positions. The individual horizon and cross-sectional signals below still
            carry the analysis; the Sharpe explorer lets you compare the strategies that did trade.
          </div>
        )}
        <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Net Sharpe" value={fmtSigned(prim.sharpe_net, 2)} sub={`95% CI [${ciLo.toFixed(2)}, ${ciHi.toFixed(2)}]`} tone={prim.sharpe_net >= 0 ? "good" : "bad"} />
          <Stat label="Total Return" value={fmtPct(prim.total_return, 1)} sub={`${prim.n_trades} non-overlapping trades`} tone={prim.total_return >= 0 ? "good" : "bad"} />
          <Stat label="Hit Rate" value={fmtPct(prim.hit_rate, 1)} sub="profitable trades" />
          <Stat label="Avg Exposure" value={fmtPct(prim.avg_exposure, 0)} sub="time in market vs cash" />
        </div>

        <Panel title="Equity curve — long/flat timing vs buy-and-hold" subtitle={`net of ${s.roundtrip_cost_bps.toFixed(2)}bps round-trip India ${s.instrument ?? ""} costs · non-overlapping ${s.primary_horizon}-day holds`}>
          <EquityCurve />
        </Panel>

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Panel title="vs passive Nifty benchmark" subtitle="does the signal beat simply holding the index?">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <div className="text-[11px] uppercase tracking-wider text-muted">Strategy</div>
                <div className={`mt-1 text-lg font-semibold tag ${prim.total_return >= 0 ? "text-accent" : "text-danger"}`}>{fmtPct(prim.total_return, 1)}</div>
                <div className="text-[11px] text-muted">Sharpe {fmtSigned(prim.sharpe_net, 2)}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wider text-muted">Buy &amp; Hold</div>
                <div className={`mt-1 text-lg font-semibold tag ${bh.total_return >= 0 ? "text-accent" : "text-danger"}`}>{fmtPct(bh.total_return, 1)}</div>
                <div className="text-[11px] text-muted">Sharpe {fmtSigned(bh.sharpe_net, 2)}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wider text-muted">Excess</div>
                <div className={`mt-1 text-lg font-semibold tag ${s.strategy_excess_return >= 0 ? "text-accent" : "text-danger"}`}>{fmtPct(s.strategy_excess_return, 1)}</div>
                <div className="text-[11px] text-muted">over holding</div>
              </div>
            </div>
            <p className="mt-4 text-xs leading-relaxed text-muted">
              The Nifty has a strong upward drift, so a long-biased signal can look good on its own.
              The honest test is the excess over passively holding the index, net of costs.
            </p>
          </Panel>
          {cb && (
            <Panel title={`India cost model · ${cb.instrument}`} subtitle={`round-trip ${cb.roundtrip_bps.toFixed(2)}bps charged on every trade`}>
              <table className="w-full text-xs tag">
                <tbody>
                  {[
                    ["STT (securities transaction tax)", cb.stt_bps],
                    ["Stamp duty", cb.stamp_duty_bps],
                    ["Slippage (bid-ask + impact)", cb.slippage_bps],
                    ["Exchange transaction", cb.exchange_txn_bps],
                    ["Brokerage", cb.brokerage_bps],
                    ["GST", cb.gst_bps],
                    ["SEBI turnover", cb.sebi_bps],
                  ].map(([k, v]: any) => (
                    <tr key={k} className="border-b border-edge/40">
                      <td className="py-1.5 text-muted">{k}</td>
                      <td className="py-1.5 text-right text-white">{Number(v).toFixed(2)} bps</td>
                    </tr>
                  ))}
                  <tr>
                    <td className="py-2 font-semibold text-white">Round-trip total</td>
                    <td className="py-2 text-right font-semibold text-accent">{cb.roundtrip_bps.toFixed(2)} bps</td>
                  </tr>
                </tbody>
              </table>
            </Panel>
          )}
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <Panel title="Strategy comparison">
            <table className="w-full text-xs">
              <thead className="text-muted">
                <tr className="border-b border-edge">
                  <th className="py-2 text-left font-normal">Strategy</th>
                  <th className="py-2 text-right font-normal">Sharpe</th>
                  <th className="py-2 text-right font-normal">Return</th>
                </tr>
              </thead>
              <tbody className="tag">
                {[
                  ["Timing · ensemble", prim],
                  ["Timing · h20 only", timingH20],
                  [`Timing · best-val h${s.best_val_horizon}`, timingBest],
                  ["Quantile L/S (ref)", quantile],
                  ["Sign (ref)", sign],
                  ["Buy & Hold", bh],
                ].map(([name, r]: any) => (
                  <tr key={name} className="border-b border-edge/40">
                    <td className="py-2 text-white">{name}</td>
                    <td className={`py-2 text-right ${r.sharpe_net >= 0 ? "text-accent" : "text-danger"}`}>
                      {fmtSigned(r.sharpe_net, 2)}
                    </td>
                    <td className={`py-2 text-right ${r.total_return >= 0 ? "text-accent" : "text-danger"}`}>
                      {fmtPct(r.total_return, 1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-[11px] leading-relaxed text-muted">
              Quantile L/S and sign are cross-sectional constructs kept for reference —
              shorting a structurally drifting index is not a meaningful strategy.
            </p>
          </Panel>
          <Panel title="Threshold sweep" subtitle="long/flat Sharpe vs calibrated P(up) threshold (thresholds from validation)">
            <ThresholdSweep />
          </Panel>
          <Panel title="Decile attribution" subtitle="mean forward return by calibrated-probability decile">
            <DecileChart />
          </Panel>
        </div>

        <div className="mt-4">
          <Panel title="Yearly Sharpe" subtitle="regime stability of the long/flat timing strategy">
            <YearlyChart />
          </Panel>
        </div>
      </Section>

      {/* Sharpe explorer */}
      <Section
        id="explorer"
        eyebrow="Interactive"
        title="Sharpe explorer — sweep the cost, watch the ratios move"
      >
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-muted">
          A Sharpe ratio is only meaningful next to its assumptions. Drag the transaction
          cost and every strategy&apos;s net Sharpe and equity curve recompute live from the
          raw per-trade returns — so you can see exactly how fragile (or robust) each ratio
          is to costs, and how the strategies re-rank against each other. The default is the
          model&apos;s India per-side cost.
        </p>
        <Panel title="Index-track strategies (horizon 20)" subtitle="net Sharpe = mean/std of non-overlapping period returns, annualized; recomputed in-browser">
          <SharpeExplorer />
        </Panel>
      </Section>

      {/* Walk-forward */}
      <Section id="walkforward" eyebrow="Robustness" title="Walk-forward validation"
        lede="Eight expanding-window folds, each retrained from scratch. If an edge were real and stable, fold Sharpes would cluster above zero. The spread across folds is the honest measure of regime robustness.">
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-muted">
          The single split can get lucky. Here the model is retrained from scratch on an
          expanding window and evaluated out-of-sample on the next block —{" "}
          {data.walkforward.length} folds, each using the long/flat ensemble timing rule
          with thresholds fixed on its own training carve. Mean net Sharpe:{" "}
          <span className={`tag ${(s.walk_forward_mean_sharpe ?? 0) >= 0 ? "text-accent" : "text-danger"}`}>
            {s.walk_forward_mean_sharpe !== null ? fmtSigned(s.walk_forward_mean_sharpe, 2) : "n/a"}
          </span>
          {s.walk_forward_sharpe_std !== null && (
            <span className="text-muted"> ± {s.walk_forward_sharpe_std.toFixed(2)} across folds</span>
          )}
          .
        </p>
        <div className="grid gap-3 md:grid-cols-4">
          {data.walkforward.map((f) => (
            <div key={f.fold} className="panel p-4">
              <div className="text-[11px] uppercase tracking-wider text-muted">Fold {f.fold}</div>
              <div className={`mt-1 text-xl font-semibold tag ${f.sharpe_net >= 0 ? "text-accent" : "text-danger"}`}>
                {fmtSigned(f.sharpe_net, 2)}
              </div>
              <div className="mt-2 space-y-1 text-[11px] text-muted">
                <div>AUC h{s.primary_horizon}: <span className="tag text-white">{fmtNum(f.auc_h20, 3)}</span></div>
                <div>Return: <span className="tag text-white">{fmtPct(f.total_return, 1)}</span></div>
                <div>Train n: <span className="tag text-white">{f.train_size}</span></div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* Cross-section */}
      <Section
        id="crosssection"
        eyebrow="Cross-Section"
        title={`Where direction models can earn: ranking ${cs2.universe_size} stocks against each other`}
      >
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-muted">
          Timing one near-efficient index failed the honest test above. This track asks a
          better question: on the same date, which of {cs2.universe_size} NSE large caps
          will do <em>relatively</em> better? The same shared-weight Transformer scores
          every stock; each rebalance goes long the top 20% and short the bottom 20% —
          a genuine cross-sectional quantile spread, the construct the single-index
          section could only imitate.
          {cs2.target_mode === "relative" && (
            <>
              {" "}Targets here are <span className="text-white">relative</span>: did the
              stock beat the cross-sectional median return that date — the canonical label
              for a ranking task (absolute direction labels saturate to &quot;up&quot; in a
              bull window).
            </>
          )}
          {cs2.use_xs_features && (
            <>
              {" "}The model also sees <span className="text-white">{cs2.n_features} features</span>{" "}
              including cross-sectional ones — momentum and returns demeaned by the
              universe, per-date percentile ranks, and sector-relative momentum — the
              relative signals a ranking task needs.
            </>
          )}
          {cs2.objective === "regression" && (
            <>
              {" "}The head is a <span className="text-white">continuous excess-return
              regression</span> (Huber loss): the model is trained on <em>how much</em> a
              name out- or underperforms the universe median, not just the binary sign — a
              richer gradient for ranking.
            </>
          )}
          {" "}Every configuration (absolute/relative targets, per-stock/cross-sectional
          features, classification/regression head) is published; none is cherry-picked.
        </p>

        <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat
            label="Mean daily IC"
            value={fmtSigned(cs2.mean_daily_ic, 3)}
            sub={`IR ${cs2.ic_ir.toFixed(2)} · ${(cs2.pct_days_ic_positive * 100).toFixed(0)}% days positive`}
            tone={cs2.mean_daily_ic >= 0 ? "good" : "bad"}
          />
          <Stat
            label="L/S Spread Sharpe"
            value={fmtSigned(cs2.spread.sharpe, 2)}
            sub={`95% CI [${cs2.spread.sharpe_ci95[0].toFixed(2)}, ${cs2.spread.sharpe_ci95[1].toFixed(2)}] · net futures`}
            tone={cs2.spread.sharpe >= 0 ? "good" : "bad"}
          />
          <Stat
            label="Long-only Top 20%"
            value={fmtPct(cs2.long_only.total_return, 1)}
            sub={`vs EW universe ${fmtPct(cs2.ew_benchmark.total_return, 1)} (gross)`}
            tone={cs2.long_only.total_return >= cs2.ew_benchmark.total_return ? "good" : "bad"}
          />
          <Stat
            label="Test Window"
            value={`${cs2.spread.n_rebalances} rebal.`}
            sub={`${cs2.test_start} → ${cs2.test_end} · 20-day holds`}
          />
        </div>

        <Panel
          title="Equity curves — spread, long-only, and the passive benchmark"
          subtitle={`legs charged real India costs: futures ${cs2.costs.futures_roundtrip_bps.toFixed(1)}bps, delivery ${cs2.costs.delivery_roundtrip_bps.toFixed(1)}bps round-trip`}
        >
          <CSEquity />
        </Panel>

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Panel title="Quintile attribution" subtitle="mean 20-day forward return by signal quintile — monotonic = signal ranks correctly">
            <CSQuintiles />
          </Panel>
          <Panel title="Daily cross-sectional IC" subtitle={`Spearman rank corr across stocks, per test date${cs2.pooled_ic !== undefined ? ` · pooled rank IC ${fmtSigned(cs2.pooled_ic, 3)}` : ""} (overlapping horizon — disclosed)`}>
            <CSICSeries />
          </Panel>
        </div>

        <div className="mt-4">
          <Panel title="Honest caveats">
            <ul className="list-inside list-disc space-y-2 text-sm text-muted">
              {cs2.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </Panel>
        </div>
      </Section>

      {/* Stock signals */}
      <Section
        id="signals"
        eyebrow="Live signals"
        title="Per-stock predictions across all 20 horizons — and pick your risk"
      >
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-muted">
          The cross-sectional model&apos;s latest output for every name in the universe:
          each row is a stock, each column a forecast horizon (1–20 days), colored by the
          calibrated probability that the stock outperforms the universe median. Click any
          stock for its horizon curve. Then choose a risk profile — each is a{" "}
          <span className="text-white">real backtested construction labelled by its actual
          historical Sharpe</span>, showing the basket it would hold from today&apos;s signal.
        </p>
        <Panel title="Signal heatmap · risk profiles" subtitle={`objective: ${data.stockSignals.objective} · calibrated probabilities`}>
          <StockSignals />
        </Panel>
      </Section>

      {/* Attention */}
      <Section id="attention" eyebrow="Interpretability" title="What the model attends to">
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Average attention by lookback distance" subtitle="2nd encoder block · averaged over samples, heads, queries">
            <AttentionChart />
          </Panel>
          <Panel title="Training history" subtitle="binary cross-entropy loss · early stopping restores best weights">
            <TrainingHistory />
          </Panel>
        </div>
      </Section>




      {/* Trade journal */}
      <Section
        id="journal"
        eyebrow="Feedback"
        title="Learning from mistakes — only where a mistake can be identified"
        lede="Every closed trade is decomposed into signal error, cost drag, and noise. The distinction matters: cost drag is arithmetic and can be acted on immediately, while a loss inside the noise floor is not a mistake at all, and 'learning' from it means fitting randomness."
      >
        {(() => {
          const j = data.journal;
          const a = j.attribution;
          const sep = j.bandit.separation;
          const cats = Object.entries(a.categories ?? {});
          return (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Closed trades" value={String(a.n_trades)}
                  sub={`noise floor ${fmtPct(a.noise_floor, 1)}`} />
                <Stat label="Hit rate" value={fmtPct(a.hit_rate, 0)}
                  sub={`binomial p = ${a.hit_rate_pvalue.toFixed(3)}`}
                  tone={a.hit_rate_is_significant ? "good" : "warn"} />
                <Stat label="Cost drag" value={fmtPct(Math.abs(a.cost_return_sum), 2)}
                  sub={`${a.actionable.cost_drag_trades} trades right but fee-eaten`}
                  tone="bad" />
                <Stat label="Inside noise floor" value={String(a.not_actionable.noise_trades)}
                  sub="carry no directional information" tone="neutral" />
              </div>

              <div className="mb-5">
                <Callout tone={a.hit_rate_is_significant ? "good" : "warn"}
                  title="Is this evidence of learning?">
                  {a.verdict}
                </Callout>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <Panel title="Where the P&L actually came from"
                  subtitle="categories are assigned by size of move relative to the noise floor, not by profit alone">
                  <div className="space-y-2">
                    {cats.map(([k, v]) => (
                      <div key={k} className="flex items-center gap-3">
                        <span className="w-28 shrink-0 text-xs text-muted">
                          {k.replace(/_/g, " ")}
                        </span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-edge">
                          <div
                            className={`h-full rounded-full ${
                              k === "win" ? "bg-accent"
                              : k === "signal_error" ? "bg-danger"
                              : k === "cost_drag" ? "bg-warn" : "bg-muted"
                            }`}
                            style={{ width: `${(v / a.n_trades) * 100}%` }}
                          />
                        </div>
                        <span className="tag w-6 text-right text-xs text-white">{v}</span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-4 text-[11px] leading-relaxed text-muted">
                    {a.actionable.note}
                  </p>
                </Panel>

                <Panel title="Bandit over the published strategy rules"
                  badge="Thompson sampling"
                  subtitle="Reinforcement learning sized to the data: it allocates between fixed, already-validated rules rather than learning a policy, because policy learning needs ~1,000,000 decisions and this book has made a few dozen.">
                  {sep ? (
                    <>
                      <div className="space-y-1.5">
                        {Object.entries(sep.p_best)
                          .sort((x, y) => y[1] - x[1])
                          .map(([arm, p]) => (
                            <div key={arm} className="flex items-center gap-3">
                              <span className="w-32 shrink-0 truncate text-xs text-muted">{arm}</span>
                              <div className="h-2 flex-1 overflow-hidden rounded-full bg-edge">
                                <div className="h-full rounded-full bg-accent2"
                                  style={{ width: `${p * 100}%` }} />
                              </div>
                              <span className="tag w-10 text-right text-xs text-white">
                                {(p * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                      </div>
                      <p className="mt-4 text-[11px] leading-relaxed text-muted">
                        P(arm is best) from the posteriors. Coin-flip baseline is{" "}
                        {(sep.uniform_baseline * 100).toFixed(0)}%. {sep.verdict}
                      </p>
                    </>
                  ) : (
                    <p className="text-xs text-muted">{j.bandit.skipped}</p>
                  )}
                </Panel>
              </div>

              <div className="mt-4">
                <Panel title={j.commentary.headline}
                  badge={j.commentary.source === "llm"
                    ? `${j.commentary.provider} · ${j.commentary.model}` : "deterministic"}
                  subtitle="Commentary explains realised history only — it is never shown a forward prediction, so it cannot give trading advice.">
                  <p className="text-sm leading-relaxed text-muted">{j.commentary.what_happened}</p>
                  <p className="mt-2 text-sm leading-relaxed text-muted">{j.commentary.what_it_means}</p>
                  <ul className="mt-3 list-inside list-disc space-y-1 text-xs text-muted">
                    {j.commentary.caveats.map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                </Panel>
              </div>
            </>
          );
        })()}
      </Section>

      {/* Adaptive retraining */}
      <Section
        id="adaptive"
        eyebrow="Adaptation"
        title="Retraining without manufacturing an edge"
        lede="Layers are sized by parameter count against the independent observations their cadence actually delivers - not by clock speed. A week carries ~0.25 independent observations at a 20-day horizon, so weekly gradient updates to a 69,589-parameter backbone would fit noise, not signal."
      >
        {(() => {
          const a = data.adaptive;
          const reg = a.registry;
          const rt = a.retrain;
          const alarms = Object.values(a.drift.detectors)
            .reduce((n, d) => n + d.n_alarms, 0);
          return (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Drift alarms" value={String(alarms)}
                  sub={`over ${a.drift.n_observations} OOS days`} tone="warn" />
                <Stat label="Recalibrations" value={String(a.recalibration.n_events)}
                  sub={`${a.recalibration.window_days}d trailing window`} />
                <Stat label="Champion" value={reg.champion ?? "none"}
                  sub={`trained through ${reg.champion_cutoff ?? "-"}`} />
                <Stat label="Trials counted" value={String(reg.n_trials)}
                  sub="feeds the deflated Sharpe" tone="neutral" />
              </div>

              <div className="mb-5 grid gap-3 md:grid-cols-4">
                {[
                  ["Daily", "Drift detection", "0 params", "monitors only, never retrains"],
                  ["Daily", "Entry threshold", "0 params", "past-only rolling percentile"],
                  ["Monthly", "Platt calibration", `~${a.design.decision_layer_params} params`, "trailing window, label-embargoed"],
                  ["Quarterly", "Backbone refit", `${a.design.backbone_params.toLocaleString()} params`, "purged, embargoed, gated"],
                ].map(([when, what, params, how]) => (
                  <Panel key={what} title={what} badge={when}>
                    <div className="tag text-lg text-accent">{params}</div>
                    <p className="mt-1 text-xs leading-relaxed text-muted">{how}</p>
                  </Panel>
                ))}
              </div>

              {rt.gate && (
                <div className="mb-5">
                  <Callout tone={rt.gate.promote ? "good" : "warn"}
                    title={`Champion/challenger gate: ${rt.gate.promote ? "PROMOTED" : "REJECTED"}`}>
                    {String(rt.gate.reason)}. The gate fails closed - a challenger that
                    cannot be shown to be better is not promoted, and every rejected
                    challenger still increments the trial count that deflates the Sharpe.
                  </Callout>
                </div>
              )}

              <Panel
                title="Drift alarms on the live signal"
                subtitle={`ADWIN + Page-Hinkley over ${a.drift.n_observations} out-of-sample days. Thresholds were calibrated empirically to zero false alarms on stationary input while still catching a 3-sigma shift.`}
              >
                <DriftTimeline />
                <p className="mt-3 text-[11px] leading-relaxed text-muted">
                  {a.drift.note}
                </p>
              </Panel>

              <p className="mt-4 max-w-3xl text-xs leading-relaxed text-muted">
                {reg.note}
              </p>
            </>
          );
        })()}
      </Section>

      {/* Documentation */}
      <Section
        id="docs"
        eyebrow="Documentation"
        title="Read the whole thing"
        lede="Every file, every metric formula, every leakage rule and every number is documented in the repository — written so someone who has never seen the project can reproduce it end to end."
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {[
            ["Overview", "What it is, what it found, why the finding is negative", "01-overview.md"],
            ["Getting Started", "Install, run, regenerate artifacts, WSL2 GPU setup", "02-getting-started.md"],
            ["Data", "Sources, schemas and the eight audited leakage rules", "03-data.md"],
            ["Architecture", "Pipeline and model diagrams, layer by layer", "04-architecture.md"],
            ["File Reference", "Every file in the repo and how to run it", "05-file-reference.md"],
            ["Evaluation", "Every metric, plus three bugs that faked skill", "06-evaluation.md"],
            ["Results", "The numbers, with confidence intervals", "07-results.md"],
            ["Instrument Choice", "Why the index, and whether a single stock is better", "08-instrument-choice.md"],
            ["Research Gaps", "Literature critique vs what is implemented", "09-research-gaps.md"],
          ].map(([title, desc, file]) => (
            <a
              key={file}
              href={`https://github.com/vj0246/Multi-Horizon-Transformer-for-Systematic-Equity-Direction-Forecasting/blob/main/Documentation/${file}`}
              className="panel block p-4 transition-colors hover:border-edge2"
            >
              <div className="text-sm font-semibold text-white">{title}</div>
              <div className="mt-1 text-xs leading-relaxed text-muted">{desc}</div>
            </a>
          ))}
        </div>
      </Section>

      {/* Method */}
      <Section id="method" eyebrow="Method" title="How to read these results — honestly">
        <div className="grid gap-4 md:grid-cols-2">
          <Panel title="What is real here">
            <ul className="list-inside list-disc space-y-2 text-sm text-muted">
              <li>Every metric is computed on held-out test data from a freshly trained model.</li>
              <li>India {s.instrument} costs ({s.roundtrip_cost_bps.toFixed(1)}bps round-trip: STT, stamp, slippage, brokerage, exchange, GST) are charged on every trade.</li>
              <li>Entry thresholds, signal z-scoring, and Platt calibration are all fit on validation data only — nothing is tuned on the test set.</li>
              <li>Sharpe is reported with a bootstrap 95% CI; results are benchmarked against passively holding the Nifty.</li>
              <li>Inputs are restricted to stationary features; returns are non-overlapping; {data.walkforward.length}-fold walk-forward retraining checks it is not a single-split fluke.</li>
            </ul>
          </Panel>
          <Panel title="Honest limitations">
            <ul className="list-inside list-disc space-y-2 text-sm text-muted">
              <li>A liquid index is near-efficient at daily frequency — skill is thin by design.</li>
              <li>~30 non-overlapping test trades: the CI is wide and any point estimate is fragile.</li>
              <li>Single-asset backtest; no borrow costs or capacity modeling (slippage is included).</li>
              <li>FinBERT sentiment fusion is wired in (config-gated) but off: NewsAPI can&apos;t backfill history, so no fabricated features enter the backtest.</li>
              <li>Research artifact, not a deployed trading system.</li>
            </ul>
          </Panel>
        </div>
      </Section>

      <footer className="border-t border-edge/60">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-2 px-5 py-8 text-xs text-muted md:flex-row md:items-center">
          <div>
            Multi-Horizon Transformer · {s.ticker} · {s.date_start}–{s.date_end}
          </div>
          <div>Built with TensorFlow · Next.js · Recharts. Results regenerated from real training runs.</div>
        </div>
      </footer>
    </main>
  );
}
