import { useState } from "react";
import { Activity, BrainCircuit, Gauge, UserCheck } from "lucide-react";
import {
  amountBuckets,
  assessmentFromScore,
  hashString,
  money,
  peerSeries,
  riskDistribution,
  signalSeries,
  type ClaimRecord,
  type ReviewStatus,
} from "@/lib/fwa";
import { explainClaim, reviewAccept, reviewReject, type ApiClaimResult, type ApiExplanation } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { KpiCard, Panel, ScoreDial } from "./primitives";
import { BarSeries, ChartFrame, PeerCompare, RiskDonut, SignalRadar, TrendLine } from "./charts";
import { PowerBiPanel } from "./PowerBiPanel";
import { LlmReport, type ReportBlock } from "./LlmReport";
import { HumanReview } from "./HumanReview";

export function ClaimDetail({
  claim,
  onStatusChange,
}: {
  claim: ClaimRecord;
  onStatusChange?: (status: ReviewStatus) => void;
}) {
  const [status, setStatus] = useState<ReviewStatus>(claim.status);
  const [explanation, setExplanation] = useState<ApiExplanation | null>(null);
  const [explainBusy, setExplainBusy] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  // Phase 5: the deterministic routing decision travels with the raw API
  // result. Only routes flagged by the router require human review — the
  // LLM explanation is still generated for every route.
  const raw = claim.api as ApiClaimResult | undefined;
  const routing = raw?.routing ?? null;
  const requiresReview = routing ? routing.requires_human_review : true;
  const seed = hashString(claim.claimId);
  const peer = peerSeries(seed);
  const analysis = claim.analysis;
  const signals = analysis
    ? analysis.signals.map((s) => ({ name: s.name, value: Math.round(s.value) }))
    : signalSeries(seed);
  const percentile = Math.min(99, Math.max(4, Math.round(claim.riskScore * 0.9 + 6)));
  const peerPayment = Math.round(claim.paymentAmount / (1 + claim.riskScore / 120));

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
          body: `The hybrid XGBoost + LightGBM model predicted "${analysis.predictedClass}". The blended class probability places this claim at a risk score of ${claim.riskScore}/100, so it is routed for expert documentation review.`,
        },
      ]
    : [
        {
          label: "Key Signals",
          body: [
            `Paid amount sits in the P${percentile} band for ${claim.claimType} claims of comparable duration.`,
            `Service count of ${claim.serviceCount} across ${claim.duration} days deviates from the cohort median.`,
            `Allowed-to-submitted ratio of ${(claim.allowedAmount / claim.claimAmount).toFixed(2)} is atypical for this provider.`,
          ],
        },
        {
          label: "Supporting Evidence",
          body: [
            `Claim ${claim.claimId} \u00b7 Provider ${claim.providerId} \u00b7 Beneficiary ${claim.beneficiaryId}`,
            `Submitted ${money(claim.claimAmount)}, allowed ${money(claim.allowedAmount)}, paid ${money(claim.paymentAmount)}, deductible ${money(claim.deductible)}.`,
            `Peer median payment for the matched cohort: ${money(peerPayment)}.`,
          ],
        },
        {
          label: "Assessment",
          body: `"${assessmentFromScore(claim.riskScore)}" \u2014 the claim demonstrates unusual payment and utilization patterns compared with relevant peer behavior.`,
        },
        {
          label: "Reasoning",
          body: `The model weighted payment deviation, service intensity and provider-level history. Individually these features are not conclusive, but jointly they place the claim above the ${claim.rating.toLowerCase()}-risk threshold and warrant documentation review before payment release.`,
        },
      ];

  function handleStatus(next: ReviewStatus) {
    setStatus(next);
    onStatusChange?.(next);
    // Accept / Reject is stored in the backend review record. It never
    // changes the ML prediction or the ML risk score.
    if (next === "Accepted" || next === "Rejected") {
      const action = next === "Accepted" ? reviewAccept : reviewReject;
      action({ case_id: raw?.claim_id ?? claim.claimId, entity_type: "claim" }).catch((err) =>
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
    explainClaim({ analysis: raw })
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
          <ScoreDial score={claim.riskScore} />
        </div>
        <KpiCard label="Risk Rating" value={claim.rating} tone={claim.rating.toLowerCase() as "high"} delay={60} />
        <KpiCard label="Model Prediction" value={analysis ? analysis.predictedClass : assessmentFromScore(claim.riskScore)} {...(analysis ? { hint: "Hybrid XGBoost + LightGBM" } : {})} delay={120} />
        <KpiCard label="Payment vs Peer" value={`${Math.round((claim.paymentAmount / peerPayment - 1) * 100)}%`} hint={`Peer median ${money(peerPayment)}`} tone="accent" delay={0} />
        <KpiCard label="Payment Percentile" value={`P${percentile}`} hint="Within specialty cohort" delay={60} />
        <KpiCard label="Service Count" value={claim.serviceCount} hint="Lines on claim" delay={120} />
        <KpiCard label="Claim Duration" value={`${claim.duration} days`} delay={180} />
        <KpiCard label="Claim Frequency" value={`${(1 + (seed % 40) / 10).toFixed(1)}×`} hint="vs beneficiary baseline" delay={0} />
        <KpiCard label="Submitted Amount" value={money(claim.claimAmount)} delay={60} />
        <KpiCard label="Allowed Amount" value={money(claim.allowedAmount)} delay={120} />
        <KpiCard label="Paid Amount" value={money(claim.paymentAmount)} delay={180} />
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
        title="Claim Power BI Dashboard"
        filterTable="Claims"
        filterColumn="ClaimID"
        filterValue={claim.claimId}
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartFrame title="Payment vs Peer Behavior">
            <PeerCompare data={peer} />
          </ChartFrame>
          <ChartFrame title="Behavioral Signals">
            <SignalRadar data={signals} />
          </ChartFrame>
          <ChartFrame title="Peer Signal Strength">
            <BarSeries data={signals.slice(0, 5)} color="var(--chart-2)" />
          </ChartFrame>
          <ChartFrame title="Utilization Trend">
            <TrendLine data={peer.map((p) => ({ name: p.name, value: p.subject }))} />
          </ChartFrame>
        </div>
      </PowerBiPanel>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_1fr]">
        <LlmReport
          score={claim.riskScore}
          rating={claim.rating}
          assessment={analysis ? `Predicted: ${analysis.predictedClass}` : assessmentFromScore(claim.riskScore)}
          blocks={[...reportBlocks, ...llmBlocks]}
          footer={
            <div className="space-y-2">
              <Button
                onClick={explainWithLlm}
                disabled={explainBusy || !claim.api}
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
              <RiskDonut data={riskDistribution([claim.riskScore, 30, 52, 81, 66, 24, 90, 44])} />
            </ChartFrame>
          </Panel>
          <Panel title="Amount Profile" icon={Activity}>
            <ChartFrame title="Claim Amount Distribution">
              <BarSeries data={amountBuckets([claim.claimAmount, 900, 4200, 13400, 26000, 41000])} />
            </ChartFrame>
          </Panel>
        </div>
      </div>

      {requiresReview ? (
        <HumanReview status={status} onChange={handleStatus} subject={`Claim ${claim.claimId}`} />
      ) : (
        <Panel
          title="Human Review"
          description={`Claim ${claim.claimId}`}
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
