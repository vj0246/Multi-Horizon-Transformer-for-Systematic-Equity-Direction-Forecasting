import { ReactNode } from "react";

export function Section({
  id,
  eyebrow,
  title,
  children,
}: {
  id?: string;
  eyebrow?: string;
  title?: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="mx-auto w-full max-w-6xl px-5 py-14 md:py-20">
      {eyebrow && (
        <div className="mb-2 text-xs uppercase tracking-[0.2em] text-accent2">{eyebrow}</div>
      )}
      {title && (
        <h2 className="mb-8 text-2xl font-semibold text-white md:text-3xl">{title}</h2>
      )}
      {children}
    </section>
  );
}

export function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`panel p-5 ${className}`}>
      {title && (
        <div className="mb-1 text-sm font-semibold text-white">{title}</div>
      )}
      {subtitle && <div className="mb-4 text-xs text-muted">{subtitle}</div>}
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "default" | "good" | "bad";
}) {
  const color =
    tone === "good" ? "text-accent" : tone === "bad" ? "text-danger" : "text-white";
  return (
    <div className="panel p-4">
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tag ${color}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-muted">{sub}</div>}
    </div>
  );
}
