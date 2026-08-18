import { Link } from "@tanstack/react-router";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { Rating, ReviewStatus } from "@/lib/fwa";

export function PageHeading({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="animate-rise mb-8 flex flex-col gap-4 text-center md:flex-row md:items-end md:justify-between md:text-left">
      <div>
        {eyebrow ? (
          <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-accent">{eyebrow}</p>
        ) : null}
        <h1 className="text-gradient-brand mt-2 text-3xl font-semibold tracking-[0.04em] sm:text-4xl">
          {title}
        </h1>
        {subtitle ? <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap justify-center gap-2 md:justify-end">{actions}</div> : null}
    </div>
  );
}

export function Panel({
  title,
  description,
  icon: Icon,
  actions,
  className,
  bodyClassName,
  children,
}: {
  title?: string;
  description?: string;
  icon?: LucideIcon;
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("surface-panel animate-rise overflow-hidden", className)}>
      {title ? (
        <header className="flex flex-wrap items-center gap-3 border-b border-border/70 bg-secondary/40 px-5 py-4">
          {Icon ? (
            <span className="flex size-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <Icon className="size-4.5" />
            </span>
          ) : null}
          <div className="min-w-0">
            <h2 className="text-sm font-semibold tracking-[0.1em] text-foreground uppercase">{title}</h2>
            {description ? (
              <p className="text-xs text-muted-foreground">{description}</p>
            ) : null}
          </div>
          {actions ? <div className="ml-auto flex flex-wrap gap-2">{actions}</div> : null}
        </header>
      ) : null}
      <div className={cn("p-5", bodyClassName)}>{children}</div>
    </section>
  );
}

export function ActionCard({
  to,
  icon: Icon,
  title,
  description,
  meta,
  delay = 0,
}: {
  to: string;
  icon: LucideIcon;
  title: string;
  description: string;
  meta?: string;
  delay?: number;
}) {
  return (
    <Link
      to={to}
      style={{ animationDelay: `${delay}ms` }}
      className="surface-panel animate-rise group relative flex flex-col gap-4 p-7 transition-all duration-500 hover:-translate-y-1.5 hover:shadow-lift"
    >
      <span className="gradient-navy pointer-events-none absolute inset-x-0 top-0 h-1 rounded-t-2xl opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
      <span className="gradient-navy flex size-14 items-center justify-center rounded-2xl text-navy-foreground shadow-float transition-transform duration-500 group-hover:scale-105">
        <Icon className="size-7" />
      </span>
      <div>
        <h3 className="text-lg font-semibold tracking-[0.08em] text-foreground uppercase">{title}</h3>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>
      </div>
      {meta ? (
        <p className="mt-auto text-[11px] font-medium uppercase tracking-[0.18em] text-accent">{meta}</p>
      ) : null}
    </Link>
  );
}

export function KpiCard({
  label,
  value,
  hint,
  tone = "default",
  delay = 0,
}: {
  label: string;
  value: string | number;
  hint?: string;
  tone?: "default" | "high" | "medium" | "low" | "accent";
  delay?: number;
}) {
  const tones: Record<string, string> = {
    default: "text-foreground",
    high: "text-risk-high",
    medium: "text-risk-medium",
    low: "text-risk-low",
    accent: "text-accent",
  };
  return (
    <div
      style={{ animationDelay: `${delay}ms` }}
      className="surface-panel animate-rise p-4 transition-all duration-400 hover:-translate-y-1 hover:shadow-lift"
    >
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">{label}</p>
      <p className={cn("mt-2 text-2xl font-semibold tabular-nums", tones[tone])}>{value}</p>
      {hint ? <p className="mt-1 text-[11px] text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

export function RiskBadge({ rating }: { rating: Rating }) {
  const styles: Record<Rating, string> = {
    High: "bg-risk-high/12 text-risk-high border-risk-high/30",
    Medium: "bg-risk-medium/14 text-risk-medium border-risk-medium/30",
    Low: "bg-risk-low/12 text-risk-low border-risk-low/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.12em]",
        styles[rating],
      )}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {rating}
    </span>
  );
}

export function StatusBadge({ status }: { status: ReviewStatus }) {
  const styles: Record<ReviewStatus, string> = {
    "Pending Review": "bg-secondary text-secondary-foreground border-border",
    Accepted: "bg-risk-low/12 text-risk-low border-risk-low/30",
    Rejected: "bg-risk-high/12 text-risk-high border-risk-high/30",
    "Resubmission Required": "bg-risk-medium/14 text-risk-medium border-risk-medium/30",
    "Auto Approved": "bg-risk-low/12 text-risk-low border-risk-low/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium whitespace-nowrap",
        styles[status],
      )}
    >
      {status}
    </span>
  );
}

export function ScoreDial({ score }: { score: number }) {
  const angle = Math.round((score / 100) * 360);
  return (
    <div className="flex items-center gap-4">
      <div
        className="relative flex size-24 items-center justify-center rounded-full"
        style={{
          background: `conic-gradient(var(--primary) ${angle}deg, oklch(0.93 0.012 245) ${angle}deg)`,
        }}
      >
        <div className="flex size-[76px] flex-col items-center justify-center rounded-full bg-card shadow-inset-hair">
          <span className="text-2xl font-semibold tabular-nums text-foreground">{score}</span>
          <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">/ 100</span>
        </div>
      </div>
      <div className="text-sm text-muted-foreground">
        <p className="font-semibold text-foreground">Composite Risk Score</p>
        <p className="mt-1 max-w-[16rem] text-xs leading-relaxed">
          Blended anomaly score from utilization, payment and peer-deviation models.
        </p>
      </div>
    </div>
  );
}
