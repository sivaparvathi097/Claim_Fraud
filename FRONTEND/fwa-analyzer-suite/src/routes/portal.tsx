import { createFileRoute } from "@tanstack/react-router";
import { Building2, FileSearch } from "lucide-react";
import { AppShell } from "@/components/fwa/AppShell";
import { ActionCard } from "@/components/fwa/primitives";

export const Route = createFileRoute("/portal")({
  head: () => ({
    meta: [
      { title: "Uncover FWA — Investigation Workspace" },
      {
        name: "description",
        content: "Launch AI-powered claim or provider investigations from the FWA workspace.",
      },
      { property: "og:title", content: "Uncover FWA — Investigation Workspace" },
      {
        property: "og:description",
        content: "Launch AI-powered claim or provider investigations from the FWA workspace.",
      },
    ],
  }),
  component: PortalPage,
});

function PortalPage() {
  return (
    <AppShell>
      <div className="mx-auto max-w-4xl pt-6 text-center">
        <p className="animate-fade text-[11px] font-semibold uppercase tracking-[0.28em] text-accent">
          Payment Integrity Command
        </p>
        <h1 className="text-gradient-brand animate-float mt-4 text-4xl font-semibold tracking-[0.1em] sm:text-5xl">
          UNCOVER FWA
        </h1>
        <p className="animate-rise mt-4 text-sm text-muted-foreground sm:text-base">
          AI-Powered Claim &amp; Provider Investigation
        </p>
      </div>

      <div className="mx-auto mt-12 grid max-w-4xl gap-6 md:grid-cols-2">
        <ActionCard
          to="/claims"
          icon={FileSearch}
          title="Analyse Claim"
          description="Score individual claims or entire submission batches for upcoding, duplication and utilization anomalies."
          meta="Single & batch claim analysis"
        />
        <ActionCard
          to="/providers"
          icon={Building2}
          title="Analyse Provider"
          description="Profile provider billing behavior against peer cohorts to surface systemic abuse patterns."
          meta="Single & batch provider analysis"
          delay={120}
        />
      </div>
    </AppShell>
  );
}
