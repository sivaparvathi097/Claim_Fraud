import { BrainCircuit, Info, ShieldAlert } from "lucide-react";
import { Panel, RiskBadge } from "./primitives";
import type { Rating } from "@/lib/fwa";

export type ReportBlock = { label: string; body: string | string[] };

export function LlmReport({
  score,
  rating,
  assessment,
  blocks,
  footer,
}: {
  score: number;
  rating: Rating;
  assessment: string;
  blocks: ReportBlock[];
  footer?: React.ReactNode;
}) {
  return (
    <Panel
      title="AI Investigation Explanation"
      description="Generated explainability report"
      icon={BrainCircuit}
      bodyClassName="p-0"
    >
      <div className="gradient-navy flex flex-wrap items-center gap-4 px-5 py-4 text-navy-foreground">
        <div>
          <p className="text-[10px] uppercase tracking-[0.22em] opacity-75">Risk score</p>
          <p className="text-3xl font-semibold tabular-nums">{score}</p>
        </div>
        <div className="h-10 w-px bg-navy-foreground/25" />
        <div>
          <p className="text-[10px] uppercase tracking-[0.22em] opacity-75">Risk rating</p>
          <p className="text-lg font-semibold">{rating}</p>
        </div>
        <div className="ml-auto flex items-center gap-2 rounded-full bg-card/15 px-3 py-1.5 text-xs font-medium backdrop-blur">
          <ShieldAlert className="size-4" />
          {assessment}
        </div>
      </div>

      <div className="divide-y divide-border/70">
        {blocks.map((block, i) => (
          <div
            key={block.label}
            style={{ animationDelay: `${i * 70}ms` }}
            className="animate-rise grid gap-2 px-5 py-4 md:grid-cols-[190px_1fr]"
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">{block.label}</p>
            {Array.isArray(block.body) ? (
              <ul className="space-y-1.5 text-sm leading-relaxed text-muted-foreground">
                {block.body.map((line) => (
                  <li key={line} className="flex gap-2">
                    <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary/60" />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm leading-relaxed text-muted-foreground">{block.body}</p>
            )}
          </div>
        ))}
      </div>

      <div className="flex items-start gap-2.5 border-t border-border/70 bg-secondary/50 px-5 py-4">
        <Info className="mt-0.5 size-4 shrink-0 text-primary" />
        <p className="text-xs leading-relaxed text-muted-foreground">
          <span className="font-semibold text-foreground">
            AI-assisted assessment — Final decision requires domain-expert review.
          </span>{" "}
          The system detects statistical anomalies and explains contributing signals; it does not
          independently establish fraud. A qualified investigator performs final verification.
        </p>
      </div>

      {footer ? <div className="border-t border-border/70 px-5 py-4">{footer}</div> : null}
    </Panel>
  );
}

export function RatingLine({ rating }: { rating: Rating }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      Model rating <RiskBadge rating={rating} />
    </div>
  );
}
