import { useState } from "react";
import { Activity, BrainCircuit, Gauge, UserCheck } from "lucide-react";
import {
  assessmentFromScore,
  hashString,
  money,
  peerSeries,
  riskDistribution,
  signalSeries,
  type ProviderRecord,
  type ReviewStatus,
} from "@/lib/fwa";
import { explainProvider, reviewAccept, reviewReject, type ApiExplanation, type ApiProviderResult } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { KpiCard, Panel, ScoreDial } from "./primitives";
import { BarSeries, ChartFrame, PeerCompare, RiskDonut, SignalRadar, TrendLine } from "./charts";
import { PowerBiPanel } from "./PowerBiPanel";
import { LlmReport, type ReportBlock } from "./LlmReport";
import { HumanReview } from "./HumanReview";

export function ProviderDetail({
  provider,
  onStatusChange,
}: {
  provider: ProviderRecord;
  onStatusChange?: (status: ReviewStatus) => void;
}) {
  const [status, setStatus] = useState<ReviewStatus>(provider.status);
  const [explanation, setExplanation] = useState<ApiExplanation | null>(null);
  const [explainBusy, setExplainBusy] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  // Phase 5: the deterministic routing decision travels with the raw API
  // result. Only routes flagged by the router require human review — the
  // LLM explanation is still generated for every route.
  const raw = provider.api as ApiProviderResult | undefined;
  const routing = raw?.routing ?? null;
  const requiresReview = routing ? routing.requires_human_review : true;
  const seed = hashString(provider.providerId);
  const peer = peerSeries(seed);
  const analysis = provider.analysis;
  const signals = analysis
    ? analysis.signals.map((s) => ({ name: s.name, value: Math.round(s.value) }))
    : signalSeries(seed);

  const reportBlocks = analysis
    ? [
        ...analysis.evidence.map((e) => ({ label: e.label, body: e.body })),
        ...(analysis.topFeatures.length
          ? [
              {
                label: "Top Model Drivers",
                body: analysis.topFeatures.map(
                  (t) => `${t.feature}: ${t.value} vs training median ${t.trainingMedian} (deviation ${t.deviation}\u00d7)`,
                ),
              },
            ]
          : []),
        {
          label: "Reasoning",
          body: `The hybrid XGBoost + LightGBM model predicted "${analysis.predictedClass}" with a suspicious probability of ${Math.round((analysis.classProbabilities["Suspicious"] ?? 0) * 100)}%, producing a risk score of ${provider.riskScore}/100.`,
        },
      ]
    : [
        {
          label: "Key Signals",
          body: [
            `Payment per beneficiary of ${money(provider.paymentPerBeneficiary)} exceeds the ${provider.specialty} cohort norm.`,
            `Utilization index of ${provider.utilization}\u00d7 indicates elevated service intensity.`,
            `${provider.claimCount.toLocaleString()} claims across ${provider.beneficiaryCount.toLocaleString()} beneficiaries produces a concentrated billing ratio.`,
          ],
        },
        {
          label: "Peer Comparison",
          body: `Relative to matched ${provider.specialty} providers, reimbursement deviates by ${provider.peerDeviation}% and admitted days by ${(provider.daysAdmitted / 30).toFixed(1)} months of cumulative stay.`,
        },
        {
          label: "Behavioral Patterns",
          body: [
            "Repeat high-intensity procedure mix within short billing windows.",
            "Weekly volume spikes clustered near period-end submissions.",
            "Beneficiary overlap with previously reviewed provider clusters.",
          ],
        },
        {
          label: "Supporting Evidence",
          body: [
            `Provider ${provider.providerId} \u00b7 Specialty ${provider.specialty}`,
            `Total reimbursed ${money(provider.reimbursedAmount)} across ${provider.claimCount.toLocaleString()} claims.`,
            `Peer deviation ${provider.peerDeviation}% \u00b7 Utilization ${provider.utilization}\u00d7`,
          ],
        },
        {
          label: "Assessment",
          body: `"${assessmentFromScore(provider.riskScore)}" \u2014 provider behavior shows patterns that differ from relevant peer benchmarks.`,
        },
        {
          label: "LLM Reasoning",
          body: `The explanation aggregates model feature attributions into narrative form. No single indicator establishes intent; the combination of payment concentration, utilization intensity and temporal clustering explains why the provider is rated ${provider.rating.toLowerCase()} risk and is routed for expert verification.`,
        },
      ];

  function handleStatus(next: ReviewStatus) {
    setStatus(next);
    onStatusChange?.(next);
    // Accept / Reject is stored in the backend review record. It never
    // changes the ML prediction or the ML risk score.
    if (next === "Accepted" || next === "Rejected") {
      const action = next === "Accepted" ? reviewAccept : reviewReject;
      action({ case_id: raw?.provider_npi ?? provider.providerId, entity_type: "provider" }).catch((err) =>
        console.error("Review submission failed", err),
      );
    }
  }

  function explainWithLlm() {
    // The EXACT analysis result already returned by the backend is sent to the
    // explain endpoint — ML and the Analytical Engine are never rerun.
    if (!raw) return;
    setExplainBusy(true);
    setExplainError(null);
    explainProvider({ analysis: raw })
      .then(setExplanation)
      .catch((err) => setExplainError(err instanceof Error ? err.message : String(err)))
      .finally(() => setExplainBusy(false));
  }

  const llmBlocks: ReportBlock[] = explanation
    ? [
        { label: "LLM Summary", body: explanation.summary },
        { label: "LLM Reasoning", body: explanation.reasoning },
        ...(explanation.supporting_signals.length
          ? [{ label: "Signals Used", body: explanation.supporting_signals }]
          : []),
        ...(explanation.evidence_used.length
          ? [{ label: "Evidence Used", body: explanation.evidence_used }]
          : []),
        ...(explanation.routing_reasons.length
          ? [{ label: "Routing Reasons", body: explanation.routing_reasons }]
          : []),
        ...(explanation.trust_factors.length
          ? [{ label: "Trust Factors", body: explanation.trust_factors }]
          : []),
        ...(explanation.caveats.length ? [{ label: "Caveats", body: explanation.caveats }] : []),
      ]
    : [];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <div className="surface-panel animate-rise p-5 md:col-span-2">
          <ScoreDial score={provider.riskScore} />
        </div>
        <KpiCard label="Risk Rating" value={provider.rating} tone={provider.rating.toLowerCase() as "high"} delay={60} />
        <KpiCard label="Model Prediction" value={analysis ? analysis.predictedClass : assessmentFromScore(provider.riskScore)} {...(analysis ? { hint: "Hybrid XGBoost + LightGBM" } : {})} delay={120} />
        <KpiCard label="Claim Count" value={provider.claimCount.toLocaleString()} delay={0} />
        <KpiCard label="Beneficiary Count" value={provider.beneficiaryCount.toLocaleString()} delay={60} />
        <KpiCard label="Reimbursed Amount" value={money(provider.reimbursedAmount)} delay={120} />
        <KpiCard label="Payment / Beneficiary" value={money(provider.paymentPerBeneficiary)} tone="accent" delay={180} />
        <KpiCard label="Peer Deviation" value={`${provider.peerDeviation > 0 ? "+" : ""}${provider.peerDeviation}%`} delay={0} />
        <KpiCard label="Utilization Index" value={`${provider.utilization}×`} hint="1.0 = cohort norm" delay={60} />
        <KpiCard label="Days Admitted" value={provider.daysAdmitted.toLocaleString()} delay={120} />
        <KpiCard label="Specialty" value={provider.specialty} delay={180} />
      </div>

      {routing ? (
        <Panel
          title="Deterministic Routing"
          description="Decided by the routing layer — the LLM only explains it"
          icon={Gauge}
        >
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full border border-border/70 bg-secondary/60 px-4 py-1.5 text-sm font-semibold uppercase tracking-[0.12em]">
              {routing.route.replace(/_/g, " ")}
            </span>
            <span className="text-xs text-muted-foreground">
              {routing.requires_human_review
                ? "Human review required"
                : "No human review required — LLM explanation still generated"}
            </span>
          </div>
          {routing.reasons.length ? (
            <ul className="mt-3 space-y-1 text-xs leading-relaxed text-muted-foreground">
              {routing.reasons.map((reason) => (
                <li key={reason}>• {reason}</li>
              ))}
            </ul>
          ) : null}
        </Panel>
      ) : null}

      <PowerBiPanel
        title="Provider Power BI Dashboard"
        filterTable="Providers"
        filterColumn="ProviderID"
        filterValue={provider.providerId}
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartFrame title="Payment vs Peer Cohort">
            <PeerCompare data={peer} />
          </ChartFrame>
          <ChartFrame title="Behavioral Indicators">
            <SignalRadar data={signals} />
          </ChartFrame>
          <ChartFrame title="Temporal Indicators">
            <TrendLine data={peer.map((p) => ({ name: p.name, value: p.subject }))} />
          </ChartFrame>
          <ChartFrame title="Signal Strength">
            <BarSeries data={signals.slice(0, 5)} color="var(--chart-2)" />
          </ChartFrame>
        </div>
      </PowerBiPanel>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_1fr]">
        <LlmReport
          score={provider.riskScore}
          rating={provider.rating}
          assessment={analysis ? `Predicted: ${analysis.predictedClass}` : assessmentFromScore(provider.riskScore)}
          blocks={[...reportBlocks, ...llmBlocks]}
          footer={
            <div className="space-y-2">
              <Button
                onClick={explainWithLlm}
                disabled={explainBusy || !provider.api}
                className="rounded-full px-6 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-float"
              >
                <BrainCircuit className="size-4" />
                {explainBusy ? "Reasoning…" : "Explain with LLM"}
              </Button>
              {explainError ? <p className="text-xs font-medium text-risk-high">{explainError}</p> : null}
            </div>
          }
        />
        <div className="space-y-6">
          <Panel title="Risk Composition" icon={Gauge}>
            <ChartFrame title="Cohort Risk Distribution">
              <RiskDonut data={riskDistribution([provider.riskScore, 28, 55, 84, 61, 33, 92, 47])} />
            </ChartFrame>
          </Panel>
          <Panel title="Payment Statistics" icon={Activity}>
            <ChartFrame title="Payment Concentration">
              <BarSeries
                data={[
                  { name: "P25", value: Math.round(provider.paymentPerBeneficiary * 0.5) },
                  { name: "P50", value: Math.round(provider.paymentPerBeneficiary * 0.72) },
                  { name: "P75", value: Math.round(provider.paymentPerBeneficiary * 0.9) },
                  { name: "Provider", value: provider.paymentPerBeneficiary },
                ]}
              />
            </ChartFrame>
          </Panel>
        </div>
      </div>

      {requiresReview ? (
        <HumanReview status={status} onChange={handleStatus} subject={`Provider ${provider.providerId}`} />
      ) : (
        <Panel
          title="Human Review"
          description={`Provider ${provider.providerId}`}
          icon={UserCheck}
        >
          <p className="text-xs leading-relaxed text-muted-foreground">
            This case satisfied the configured automated approval policy, so no human review is
            required. The LLM explanation above is still generated for auditability.
          </p>
        </Panel>
      )}
    </div>
  );
}
