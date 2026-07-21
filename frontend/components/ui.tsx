import { ReactNode } from "react";

export function Section({
  id,
  eyebrow,
  title,
  lede,
  children,
}: {
  id?: string;
  eyebrow?: string;
  title?: string;
  lede?: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="mx-auto w-full max-w-6xl scroll-mt-16 px-5 py-14 md:py-20">
      {eyebrow && (
        <div className="mb-2 flex items-center gap-2">
          <span className="h-px w-6 bg-accent/60" />
          <span className="text-xs uppercase tracking-[0.2em] text-accent2">{eyebrow}</span>
        </div>
      )}
      {title && (
        <h2 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">{title}</h2>
      )}
      {lede && <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">{lede}</p>}
      <div className={title || lede ? "mt-8" : ""}>{children}</div>
    </section>
  );
}

export function Panel({
  title,
  subtitle,
  badge,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  badge?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`panel p-5 ${className}`}>
      {(title || badge) && (
        <div className="mb-1 flex items-start justify-between gap-3">
          {title && <div className="text-sm font-semibold text-white">{title}</div>}
          {badge && (
            <span className="shrink-0 rounded-full border border-edge bg-ink/60 px-2 py-0.5 text-[10px] uppercase tracking-wider text-muted">
              {badge}
            </span>
          )}
        </div>
      )}
      {subtitle && <div className="mb-4 text-xs leading-relaxed text-muted">{subtitle}</div>}
      {children}
    </div>
  );
}

const TONE = {
  default: "text-white",
  good: "text-accent",
  bad: "text-danger",
  warn: "text-warn",
  neutral: "text-muted",
} as const;

export function Stat({
  label,
  value,
  sub,
  tone = "default",
  hint,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: keyof typeof TONE;
  hint?: string;
}) {
  return (
    <div className="panel p-4" title={hint}>
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tag ${TONE[tone]}`}>{value}</div>
      {sub && <div className="mt-1 text-xs leading-snug text-muted">{sub}</div>}
    </div>
  );
}

/** Callout for verdicts, warnings and honesty notes. */
export function Callout({
  tone = "note",
  title,
  children,
}: {
  tone?: "note" | "warn" | "danger" | "good";
  title?: string;
  children: ReactNode;
}) {
  const box = {
    note: "border-accent2/40 bg-accent2/5",
    good: "border-accent/40 bg-accent/5",
    warn: "border-warn/40 bg-warn/5",
    danger: "border-danger/40 bg-danger/5",
  }[tone];
  const dot = {
    note: "bg-accent2",
    good: "bg-accent",
    warn: "bg-warn",
    danger: "bg-danger",
  }[tone];
  const head = {
    note: "text-accent2",
    good: "text-accent",
    warn: "text-warn",
    danger: "text-danger",
  }[tone];
  return (
    <div className={`rounded-lg border px-4 py-3 text-xs leading-relaxed text-muted ${box}`}>
      {title && (
        <div className={`mb-1 flex items-center gap-2 font-semibold ${head}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
          {title}
        </div>
      )}
      {children}
    </div>
  );
}
