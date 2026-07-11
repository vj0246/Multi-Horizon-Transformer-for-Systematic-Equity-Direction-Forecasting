"use client";

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
import { data } from "@/lib/data";

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
  const eq = data.strategies.quantile?.equity_curve ?? [];
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
        <Area type="monotone" dataKey="strategy" name="quantile L/S" stroke="#4ade80" strokeWidth={2} fill="url(#eq)" />
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
