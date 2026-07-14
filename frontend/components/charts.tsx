"use client";

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { data, type StrategyReport } from "@/lib/data";

const AX = { stroke: "#8b98ad", fontSize: 11 };
const GRID = "#232c3d";

const tip = {
  contentStyle: {
    background: "#0f1420",
    border: "1px solid #232c3d",
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#e6edf5" },
};

export function HorizonAUC() {
  const d = data.horizons.map((h) => ({ h: h.horizon, auc: h.auc, base: 0.5 }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={d} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="h" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis domain={[0.4, 0.65]} tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(4)} />
        <ReferenceLine y={0.5} stroke="#f87171" strokeDasharray="4 4" />
        <Bar dataKey="auc" radius={[3, 3, 0, 0]}>
          {d.map((row, i) => (
            <Cell key={i} fill={row.auc >= 0.5 ? "#4ade80" : "#f87171"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function HorizonIC() {
  const d = data.horizons.map((h) => ({ h: h.horizon, ic: h.ic }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={d} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="h" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(4)} />
        <ReferenceLine y={0} stroke="#8b98ad" />
        <Bar dataKey="ic" radius={[3, 3, 0, 0]}>
          {d.map((row, i) => (
            <Cell key={i} fill={row.ic >= 0 ? "#38bdf8" : "#f87171"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function EquityCurve() {
  const eq = data.strategies.timing_ensemble?.equity_curve ?? [];
  const bh = data.strategies.buy_and_hold?.equity_curve ?? [];
  const d = eq.map((v, i) => ({ i, strategy: v, buyhold: bh[i] }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={d} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#4ade80" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#4ade80" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="i" tick={AX} tickLine={false} axisLine={{ stroke: GRID }}
          label={{ value: "trade #", fill: "#8b98ad", fontSize: 11, dy: 12 }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v?.toFixed(4)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine y={1} stroke="#8b98ad" strokeDasharray="3 3" />
        <Area type="monotone" dataKey="strategy" name="long/flat timing (ensemble)" stroke="#4ade80" strokeWidth={2} fill="url(#eq)" />
        <Area type="monotone" dataKey="buyhold" name="buy & hold Nifty" stroke="#8b98ad" strokeWidth={1.6} strokeDasharray="4 3" fill="none" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function AttentionChart() {
  const d = [...data.attention].sort((a, b) => a.days_back - b.days_back);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={d} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="days_back" tick={AX} tickLine={false} axisLine={{ stroke: GRID }}
          reversed label={{ value: "days back", fill: "#8b98ad", fontSize: 11, dy: 12 }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(5)} />
        <Line type="monotone" dataKey="weight" stroke="#38bdf8" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ThresholdSweep() {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data.thresholdSweep} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="threshold" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} />
        <ReferenceLine y={0} stroke="#8b98ad" />
        <Line type="monotone" dataKey="sharpe" stroke="#4ade80" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function DecileChart() {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data.decile} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="decile" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(4)} />
        <ReferenceLine y={0} stroke="#8b98ad" />
        <Bar dataKey="mean_return" radius={[3, 3, 0, 0]}>
          {data.decile.map((row, i) => (
            <Cell key={i} fill={row.mean_return >= 0 ? "#4ade80" : "#f87171"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function YearlyChart() {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data.yearly} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="year" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} />
        <ReferenceLine y={0} stroke="#8b98ad" />
        <Bar dataKey="sharpe" radius={[3, 3, 0, 0]}>
          {data.yearly.map((row, i) => (
            <Cell key={i} fill={row.sharpe >= 0 ? "#4ade80" : "#f87171"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function TrainingHistory() {
  const t = data.training;
  const d = t.loss.map((l, i) => ({ epoch: i + 1, loss: l, val_loss: t.val_loss[i] }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={d} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="epoch" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(4)} />
        <Line type="monotone" dataKey="loss" stroke="#38bdf8" strokeWidth={2} dot={false} name="train loss" />
        <Line type="monotone" dataKey="val_loss" stroke="#4ade80" strokeWidth={2} dot={false} name="val loss" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CalibrationChart() {
  const bins = new Map<number, { x: number; raw?: number; calibrated?: number }>();
  data.calibration.pre.forEach((b) => {
    bins.set(b.bin_mid, { x: b.bin_mid, raw: b.observed });
  });
  data.calibration.post.forEach((b) => {
    const e = bins.get(b.bin_mid) ?? { x: b.bin_mid };
    e.calibrated = b.observed;
    bins.set(b.bin_mid, e);
  });
  const d = [...bins.values()].sort((a, b) => a.x - b.x);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={d} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="x" type="number" domain={[0, 1]} tick={AX} tickLine={false}
          axisLine={{ stroke: GRID }}
          label={{ value: "predicted P(up)", fill: "#8b98ad", fontSize: 11, dy: 12 }} />
        <YAxis domain={[0, 1]} tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v?.toFixed(3)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#8b98ad" strokeDasharray="4 4" />
        <Line type="monotone" dataKey="raw" name="raw sigmoid" stroke="#f87171" strokeWidth={1.6} strokeDasharray="5 3" connectNulls />
        <Line type="monotone" dataKey="calibrated" name="Platt-calibrated" stroke="#4ade80" strokeWidth={2} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CSQuintiles() {
  const d = data.crossSection.quintile_mean_fwd20.map((v, i) => ({
    q: `Q${i + 1}`,
    ret: v,
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={d} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="q" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
        <ReferenceLine y={0} stroke="#8b98ad" />
        <Bar dataKey="ret" radius={[3, 3, 0, 0]}>
          {d.map((row, i) => (
            <Cell key={i} fill={row.ret >= 0 ? "#4ade80" : "#f87171"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CSEquity() {
  const cs = data.crossSection;
  const n = Math.max(
    cs.spread.equity_curve.length,
    cs.long_only.equity_curve.length,
    cs.ew_benchmark.equity_curve.length
  );
  const d = Array.from({ length: n }, (_, i) => ({
    i,
    spread: cs.spread.equity_curve[i],
    longonly: cs.long_only.equity_curve[i],
    ew: cs.ew_benchmark.equity_curve[i],
  }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={d} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="i" tick={AX} tickLine={false} axisLine={{ stroke: GRID }}
          label={{ value: "rebalance #", fill: "#8b98ad", fontSize: 11, dy: 12 }} />
        <YAxis tick={AX} tickLine={false} axisLine={false} domain={["auto", "auto"]} />
        <Tooltip {...tip} formatter={(v: number) => v?.toFixed(4)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine y={1} stroke="#8b98ad" strokeDasharray="3 3" />
        <Line type="monotone" dataKey="spread" name="L/S spread (net, futures)" stroke="#4ade80" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="longonly" name="long-only top 20% (net, delivery)" stroke="#38bdf8" strokeWidth={1.8} dot={false} />
        <Line type="monotone" dataKey="ew" name="equal-weight universe (gross)" stroke="#8b98ad" strokeWidth={1.6} strokeDasharray="4 3" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CSICSeries() {
  const d = data.crossSection.ic_series.map((r) => ({ date: r.date, ic: r.ic }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={d} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="ic" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="date" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} minTickGap={60} />
        <YAxis tick={AX} tickLine={false} axisLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v?.toFixed(3)} />
        <ReferenceLine y={0} stroke="#8b98ad" />
        <Area type="monotone" dataKey="ic" stroke="#38bdf8" strokeWidth={1.4} fill="url(#ic)" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ---- Live stock signals: per-stock, all-horizons heatmap + risk profiles ---
function probColor(p: number): string {
  // Diverging around 0.5: red (underperform) -> neutral -> green (outperform).
  const t = Math.max(-1, Math.min(1, (p - 0.5) * 6));
  return t >= 0
    ? `rgba(74, 222, 128, ${(t * 0.85).toFixed(3)})`
    : `rgba(248, 113, 113, ${(-t * 0.85).toFixed(3)})`;
}

export function StockSignals() {
  const sig = data.stockSignals;
  const H = sig.horizons;
  const [sel, setSel] = useState(sig.stocks[0]?.ticker ?? "");
  const [risk, setRisk] = useState(sig.risk_profiles[1]?.key ?? sig.risk_profiles[0]?.key ?? "");

  const selRow = sig.stocks.find((s) => s.ticker === sel) ?? sig.stocks[0];
  const profile = sig.risk_profiles.find((p) => p.key === risk) ?? sig.risk_profiles[0];
  const curve = selRow?.probs.map((p, i) => ({ h: i + 1, p })) ?? [];

  return (
    <div>
      <div className="mb-5 rounded-lg border border-danger/40 bg-danger/5 px-4 py-3 text-xs leading-relaxed text-muted">
        <span className="font-semibold text-danger">Research demonstration — not investment advice.</span>{" "}
        {sig.disclaimer.replace("RESEARCH DEMONSTRATION ONLY - NOT INVESTMENT ADVICE. ", "")}
      </div>

      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <span>Model signal as of <span className="tag text-white">{sig.as_of}</span> · {sig.n_stocks} names · calibrated P(outperform universe median) per horizon</span>
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: probColor(0.62) }} /> outperform</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: probColor(0.38) }} /> underperform</span>
        </span>
      </div>

      {/* Heatmap: stocks x 20 horizons */}
      <div className="overflow-x-auto">
        <div className="min-w-[640px]">
          <div className="flex text-[9px] text-muted">
            <div className="w-20 shrink-0" />
            {Array.from({ length: H }, (_, i) => (
              <div key={i} className="flex-1 text-center">{i + 1}</div>
            ))}
          </div>
          {sig.stocks.map((s) => (
            <button
              key={s.ticker}
              onClick={() => setSel(s.ticker)}
              className={`flex w-full items-center ${s.ticker === sel ? "ring-1 ring-accent2" : ""}`}
              title={`${s.ticker} · rank ${(s.rank_pct * 100).toFixed(0)}%`}
            >
              <div className={`w-20 shrink-0 truncate pr-1 text-left text-[10px] ${s.ticker === sel ? "text-white" : "text-muted"}`}>{s.ticker}</div>
              {s.probs.map((p, i) => (
                <div key={i} className="h-3.5 flex-1" style={{ background: probColor(p) }} title={`h${i + 1}: ${(p * 100).toFixed(0)}%`} />
              ))}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        {/* Selected stock horizon curve */}
        <div>
          <div className="mb-2 text-xs text-muted">
            <span className="tag text-white">{selRow?.ticker}</span> ({selRow?.sector}) · P(outperform) across horizons
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={curve} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="h" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
              <YAxis domain={[0.3, 0.7]} tick={AX} tickLine={false} axisLine={false} />
              <Tooltip {...tip} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
              <ReferenceLine y={0.5} stroke="#8b98ad" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="p" stroke="#38bdf8" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Risk profile selector */}
        <div>
          <div className="mb-2 text-xs text-muted">Pick a risk profile — each is a real backtested construction, labelled by its historical Sharpe</div>
          <div className="mb-3 flex gap-2">
            {sig.risk_profiles.map((p) => (
              <button
                key={p.key}
                onClick={() => setRisk(p.key)}
                className={`flex-1 rounded-md border px-2 py-2 text-xs ${p.key === risk ? "border-accent2 bg-panel2 text-white" : "border-edge text-muted hover:text-white"}`}
              >
                <div className="font-semibold">{p.label}</div>
                <div className={`tag ${p.sharpe >= 0 ? "text-accent" : "text-danger"}`}>Sharpe {p.sharpe >= 0 ? "+" : ""}{p.sharpe.toFixed(2)}</div>
              </button>
            ))}
          </div>
          <div className="rounded-lg border border-edge bg-panel2/40 p-3 text-xs">
            <div className="mb-2 text-muted">{profile?.construction}</div>
            <div className="mb-3 grid grid-cols-3 gap-2 tag">
              <div><div className="text-[10px] uppercase text-muted">Hist. return</div><div className={profile?.total_return >= 0 ? "text-accent" : "text-danger"}>{((profile?.total_return ?? 0) * 100).toFixed(1)}%</div></div>
              <div><div className="text-[10px] uppercase text-muted">Sharpe</div><div className={profile?.sharpe >= 0 ? "text-accent" : "text-danger"}>{(profile?.sharpe ?? 0).toFixed(2)}</div></div>
              <div><div className="text-[10px] uppercase text-muted">Max DD</div><div className="text-danger">{((profile?.max_drawdown ?? 0) * 100).toFixed(1)}%</div></div>
            </div>
            <div className="mb-1 text-[10px] uppercase tracking-wider text-accent">Long today</div>
            <div className="mb-2 flex flex-wrap gap-1">
              {profile?.long.slice(0, 12).map((t) => (
                <span key={t} className="rounded border border-accent/30 bg-accent/5 px-1.5 py-0.5 text-[10px] text-accent">{t}</span>
              ))}
              {profile && profile.long.length > 12 && <span className="text-[10px] text-muted">+{profile.long.length - 12} more</span>}
            </div>
            {profile && profile.short.length > 0 && (
              <>
                <div className="mb-1 text-[10px] uppercase tracking-wider text-danger">Short today</div>
                <div className="flex flex-wrap gap-1">
                  {profile.short.map((t) => (
                    <span key={t} className="rounded border border-danger/30 bg-danger/5 px-1.5 py-0.5 text-[10px] text-danger">{t}</span>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---- Interactive Sharpe explorer ------------------------------------------
// Recomputes each strategy's Sharpe and equity live as the user sweeps the
// per-side transaction cost. Uses the raw gross returns and exposures exported
// per strategy: net_i = gross_i - abs_pos_i * 2 * (bps / 1e4).
const EXPLORER_STRATS: { key: string; label: string; color: string }[] = [
  { key: "timing_ensemble", label: "Timing · ensemble", color: "#4ade80" },
  { key: "timing_h20", label: "Timing · h20", color: "#38bdf8" },
  { key: "quantile", label: "Quantile L/S", color: "#f59e0b" },
  { key: "sign", label: "Sign", color: "#f87171" },
];

function annSharpe(net: number[], ppy: number): number {
  if (net.length < 2) return 0;
  const mean = net.reduce((a, b) => a + b, 0) / net.length;
  const variance = net.reduce((a, b) => a + (b - mean) ** 2, 0) / net.length;
  const sd = Math.sqrt(variance);
  return sd === 0 ? 0 : (mean / sd) * Math.sqrt(ppy);
}

function applyCost(strat: StrategyReport, bps: number): number[] {
  const g = strat.gross_returns ?? [];
  const a = strat.abs_pos ?? [];
  return g.map((gi, i) => gi - (a[i] ?? 0) * 2 * (bps / 1e4));
}

export function SharpeExplorer() {
  const roundtripDefault = Math.round((data.summary.per_side_cost_bps ?? 5) * 10) / 10;
  const [bps, setBps] = useState(roundtripDefault);

  const strats = useMemo(
    () => EXPLORER_STRATS.map((s) => ({ ...s, r: data.strategies[s.key] }))
      .filter((s) => s.r && s.r.gross_returns && s.r.gross_returns.length > 1),
    []
  );

  const computed = useMemo(
    () =>
      strats.map((s) => {
        const net = applyCost(s.r, bps);
        const ppy = s.r.periods_per_year ?? 12.6;
        let eq = 1;
        const curve = net.map((n) => (eq *= 1 + n));
        return {
          key: s.key,
          label: s.label,
          color: s.color,
          sharpe: annSharpe(net, ppy),
          total: (curve.length ? curve[curve.length - 1] : 1) - 1,
          curve,
        };
      }),
    [strats, bps]
  );

  const bars = computed.map((c) => ({ label: c.label, sharpe: c.sharpe, color: c.color }));
  const maxLen = Math.max(...computed.map((c) => c.curve.length), 0);
  const equity = Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number> = { i };
    computed.forEach((c) => (row[c.key] = c.curve[i]));
    return row;
  });

  return (
    <div>
      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <label className="text-xs text-muted">
          Transaction cost:{" "}
          <span className="tag text-white">{bps.toFixed(1)} bps/side</span>{" "}
          <span className="text-muted">({(bps * 2).toFixed(1)} bps round-trip)</span>
        </label>
        <input
          type="range"
          min={0}
          max={30}
          step={0.5}
          value={bps}
          onChange={(e) => setBps(parseFloat(e.target.value))}
          className="h-1 w-full cursor-pointer appearance-none rounded bg-edge accent-accent sm:w-1/2"
          aria-label="transaction cost basis points per side"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <div className="mb-2 text-xs text-muted">Net Sharpe by strategy at this cost</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 12, left: 40, bottom: 0 }}>
              <CartesianGrid stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
              <YAxis type="category" dataKey="label" tick={{ ...AX, fontSize: 10 }} tickLine={false} axisLine={false} width={90} />
              <Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} />
              <ReferenceLine x={0} stroke="#8b98ad" />
              <Bar dataKey="sharpe" radius={[0, 3, 3, 0]}>
                {bars.map((b, i) => (
                  <Cell key={i} fill={b.sharpe >= 0 ? b.color : "#f87171"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="mb-2 text-xs text-muted">Equity curves at this cost (net)</div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={equity} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="i" tick={AX} tickLine={false} axisLine={{ stroke: GRID }} />
              <YAxis tick={AX} tickLine={false} axisLine={false} />
              <Tooltip {...tip} formatter={(v: number) => v?.toFixed(3)} />
              <ReferenceLine y={1} stroke="#8b98ad" strokeDasharray="3 3" />
              {computed.map((c) => (
                <Line key={c.key} type="monotone" dataKey={c.key} name={c.label} stroke={c.color} strokeWidth={1.8} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <table className="mt-4 w-full text-xs tag">
        <thead className="text-muted">
          <tr className="border-b border-edge">
            <th className="py-2 text-left font-normal">Strategy</th>
            <th className="py-2 text-right font-normal">Net Sharpe</th>
            <th className="py-2 text-right font-normal">Total return</th>
          </tr>
        </thead>
        <tbody>
          {computed.map((c) => (
            <tr key={c.key} className="border-b border-edge/40">
              <td className="py-2" style={{ color: c.color }}>{c.label}</td>
              <td className={`py-2 text-right ${c.sharpe >= 0 ? "text-accent" : "text-danger"}`}>
                {c.sharpe >= 0 ? "+" : ""}{c.sharpe.toFixed(2)}
              </td>
              <td className={`py-2 text-right ${c.total >= 0 ? "text-accent" : "text-danger"}`}>
                {(c.total * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PriceChart() {
  const d = data.price;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={d} margin={{ top: 8, right: 8, left: -4, bottom: 0 }}>
        <defs>
          <linearGradient id="px" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="date" tick={AX} tickLine={false} axisLine={{ stroke: GRID }}
          minTickGap={60} />
        <YAxis tick={AX} tickLine={false} axisLine={false} domain={["auto", "auto"]} width={52} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(0)} />
        <Area type="monotone" dataKey="close" stroke="#38bdf8" strokeWidth={1.6} fill="url(#px)" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
