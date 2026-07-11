import { data, fmtNum, fmtPct, fmtSigned } from "@/lib/data";
import { Panel, Section, Stat } from "@/components/ui";
import {
  AttentionChart,
  DecileChart,
  EquityCurve,
  HorizonAUC,
  HorizonIC,
  PriceChart,
  ThresholdSweep,
  TrainingHistory,
  YearlyChart,
} from "@/components/charts";

const s = data.summary;

const NAV = [
  ["overview", "Overview"],
  ["architecture", "Architecture"],
  ["results", "Results"],
  ["backtest", "Backtest"],
  ["walkforward", "Walk-Forward"],
  ["attention", "Attention"],
  ["method", "Method"],
];

export default function Page() {
  const q = data.strategies.quantile;
  const sign = data.strategies.sign;
  const long = data.strategies.long;
  const bh = data.strategies.buy_and_hold;
  const cb = s.cost_breakdown;
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
              label="L/S Sharpe (net)"
              value={fmtSigned(q.sharpe_net, 2)}
              sub={`net of ${s.roundtrip_cost_bps.toFixed(1)}bps round-trip · h=${s.primary_horizon}`}
              tone={q.sharpe_net >= 0 ? "good" : "bad"}
            />
            <Stat
              label="Max Drawdown"
              value={fmtPct(q.max_drawdown, 1)}
              sub="quantile long-short equity"
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
              Raw logits are treated as a continuous alpha signal and stress-tested with
              quant metrics: Information Coefficient, quantile long-short Sharpe with
              transaction costs, decile attribution, and walk-forward validation.
            </p>
          </Panel>
        </div>

        <div className="mt-4">
          <Panel title={`${s.ticker} price context`} subtitle={`${s.date_start} → ${s.date_end}`}>
            <PriceChart />
          </Panel>
        </div>
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
                  "GlobalAveragePooling1D",
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
          <Panel title="Input features" subtitle={`${data.features.length} engineered signals per day`}>
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
      </Section>

      {/* Backtest */}
      <Section id="backtest" eyebrow="Backtest" title={`Alpha-signal backtest · horizon ${s.primary_horizon}`}>
        <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Net Sharpe" value={fmtSigned(q.sharpe_net, 2)} sub="quantile long-short" tone={q.sharpe_net >= 0 ? "good" : "bad"} />
          <Stat label="Total Return" value={fmtPct(q.total_return, 1)} sub={`${q.n_trades} non-overlapping trades`} tone={q.total_return >= 0 ? "good" : "bad"} />
          <Stat label="Hit Rate" value={fmtPct(q.hit_rate, 1)} sub="profitable trades" />
          <Stat label="Avg Exposure" value={fmtPct(q.avg_exposure, 0)} sub="fraction of capital deployed" />
        </div>

        <Panel title="Equity curve — quantile long-short vs buy-and-hold" subtitle={`net of ${s.roundtrip_cost_bps.toFixed(2)}bps round-trip India ${s.instrument ?? ""} costs · non-overlapping ${s.primary_horizon}-day holds`}>
          <EquityCurve />
        </Panel>

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Panel title="vs passive Nifty benchmark" subtitle="does the signal beat simply holding the index?">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <div className="text-[11px] uppercase tracking-wider text-muted">Strategy</div>
                <div className={`mt-1 text-lg font-semibold tag ${q.total_return >= 0 ? "text-accent" : "text-danger"}`}>{fmtPct(q.total_return, 1)}</div>
                <div className="text-[11px] text-muted">Sharpe {fmtSigned(q.sharpe_net, 2)}</div>
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
                  ["Sign", sign],
                  ["Quantile L/S", q],
                  ["Long-only", long],
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
          </Panel>
          <Panel title="Threshold sweep" subtitle="long-only Sharpe vs entry threshold">
            <ThresholdSweep />
          </Panel>
          <Panel title="Decile attribution" subtitle="mean forward return by signal decile">
            <DecileChart />
          </Panel>
        </div>

        <div className="mt-4">
          <Panel title="Yearly Sharpe" subtitle="regime stability of the quantile long-short strategy">
            <YearlyChart />
          </Panel>
        </div>
      </Section>

      {/* Walk-forward */}
      <Section id="walkforward" eyebrow="Robustness" title="Walk-forward validation">
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-muted">
          The single split can get lucky. Here the model is retrained from scratch on an
          expanding window and evaluated out-of-sample on the next block —{" "}
          {data.walkforward.length} folds. Mean net Sharpe:{" "}
          <span className={`tag ${(s.walk_forward_mean_sharpe ?? 0) >= 0 ? "text-accent" : "text-danger"}`}>
            {s.walk_forward_mean_sharpe !== null ? fmtSigned(s.walk_forward_mean_sharpe, 2) : "n/a"}
          </span>
          .
        </p>
        <div className="grid gap-3 md:grid-cols-5">
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
              <li>Results are benchmarked against passively holding the Nifty, net of costs.</li>
              <li>Returns are non-overlapping; walk-forward retraining checks it is not a single-split fluke.</li>
            </ul>
          </Panel>
          <Panel title="Honest limitations">
            <ul className="list-inside list-disc space-y-2 text-sm text-muted">
              <li>A liquid index is near-efficient at daily frequency — skill is thin by design.</li>
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
