import { createFileRoute } from "@tanstack/react-router";
import { Building, Network } from "lucide-react";
import { AppShell } from "@/components/fwa/AppShell";
import { ActionCard, PageHeading } from "@/components/fwa/primitives";

export const Route = createFileRoute("/providers/")({
  head: () => ({
    meta: [
      { title: "Uncover Providers — FWA Risk Investigator" },
      { name: "description", content: "Run single-provider or batch provider FWA risk analysis." },
      { property: "og:title", content: "Uncover Providers — FWA Risk Investigator" },
      { property: "og:description", content: "Run single-provider or batch provider FWA risk analysis." },
    ],
  }),
  component: ProvidersPage,
});

function ProvidersPage() {
  return (
    <AppShell>
      <PageHeading
        eyebrow="Provider Investigation"
        title="UNCOVER PROVIDERS"
        subtitle="Profile provider billing behavior individually or across an uploaded network."
      />
      <div className="mx-auto grid max-w-4xl gap-6 md:grid-cols-2">
        <ActionCard
          to="/providers/single"
          icon={Building}
          title="Single Provider"
          description="Score one provider from entered attributes and review peer deviation, utilization and behavioral signals."
          meta="Interactive scoring"
        />
        <ActionCard
          to="/providers/batch"
          icon={Network}
          title="Batch Provider"
          description="Upload a provider file to rank the full network by risk and route high-risk entities to investigation."
          meta="Network triage"
          delay={120}
        />
      </div>
    </AppShell>
  );
}
