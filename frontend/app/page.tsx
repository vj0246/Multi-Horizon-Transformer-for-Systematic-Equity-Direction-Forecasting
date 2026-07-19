import { data, fmtNum, fmtPct, fmtSigned } from "@/lib/data";
import { Panel, Section, Stat } from "@/components/ui";
import {
  AttentionChart,
  CalibrationChart,
  CSEquity,
  CSICSeries,
  CSQuintiles,
  DecileChart,
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
          <a
            href="https://github.com"
            className="rounded-md border border-edge px-3 py-1 text-xs text-muted hover:text-white"
          >
            {s.ticker}
          </a>
        </div>
      </header>

      {/* Hero */}
      <div className="grid-lines border-b border-edge/50">
        <div className="mx-auto max-w-6xl px-5 py-16 md:py-24">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-edge bg-panel px-3 py-1 text-xs text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            {s.date_start} → {s.date_end} · {s.n_trading_days.toLocaleString()} trading days
          </div>
          <h1 className="max-w-3xl text-3xl font-bold leading-tight text-white md:text-5xl">
            Multi-Horizon Transformer for{" "}
            <span className="text-accent">Nifty 50</span> Direction Forecasting
          </h1>
          <p className="mt-5 max-w-2xl text-sm leading-relaxed text-muted md:text-base">
            A single Transformer encoder predicts whether the Nifty 50 index will close
            higher — simultaneously across {s.horizons} forward horizons (1 to {s.horizons}{" "}
            days). Trained on {s.n_features} engineered features over{" "}
            {s.n_samples.toLocaleString()} sequences with a strict temporal split. Every
            number below comes from a real trained model on held-out test data.
          </p>

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
                {v.headline} A calibrated 63% P(up) whose AUC interval spans 0.50 is
                a confident-looking number with no evidence behind it — the table
                below is laid out so that is visible at a glance rather than buried.
              </div>

              <Panel
                title="Forward probabilities across all 20 horizons"
                subtitle={`frozen ensemble (${pr.model.n_seeds} seeds, trained through ${pr.model.frozen_through}) · ${pr.model.calibration} · last close ${fmtNum(pr.last_close, 0)}`}
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
      <Section id="results" eyebrow="Results" title="Per-horizon predictive skill (test set)">
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="ROC-AUC by horizon" subtitle="green = above 0.5 coin-flip baseline">
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
      <Section id="backtest" eyebrow="Backtest" title="Long/flat timing backtest — calibrated 20-horizon ensemble">
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
      <Section id="walkforward" eyebrow="Robustness" title="Walk-forward validation">
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
