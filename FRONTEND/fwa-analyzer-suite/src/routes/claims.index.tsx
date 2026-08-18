import { createFileRoute } from "@tanstack/react-router";
import { FileText, Layers } from "lucide-react";
import { AppShell } from "@/components/fwa/AppShell";
import { ActionCard, PageHeading } from "@/components/fwa/primitives";

export const Route = createFileRoute("/claims/")({
  head: () => ({
    meta: [
      { title: "Uncover Claims — FWA Risk Investigator" },
      { name: "description", content: "Run single-claim or batch claim FWA risk analysis." },
      { property: "og:title", content: "Uncover Claims — FWA Risk Investigator" },
      { property: "og:description", content: "Run single-claim or batch claim FWA risk analysis." },
    ],
  }),
  component: ClaimsPage,
});

function ClaimsPage() {
  return (
    <AppShell>
      <PageHeading
        eyebrow="Claim Investigation"
        title="UNCOVER CLAIMS"
        subtitle="Choose an investigation mode to begin scoring claim-level risk."
      />
      <div className="mx-auto grid max-w-4xl gap-6 md:grid-cols-2">
        <ActionCard
          to="/claims/single"
          icon={FileText}
          title="Single Claims"
          description="Enter claim attributes manually and receive an immediate risk score, dashboard and explainability report."
          meta="Interactive scoring"
        />
        <ActionCard
          to="/claims/batch"
          icon={Layers}
          title="Batch Claims"
          description="Upload a claims file to score the full dataset, review portfolio KPIs and triage a prioritized queue."
          meta="Dataset triage"
          delay={120}
        />
      </div>
    </AppShell>
  );
}
