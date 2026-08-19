import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ClipboardList, Sparkles } from "lucide-react";
import { AppShell } from "@/components/fwa/AppShell";
import { PageHeading, Panel } from "@/components/fwa/primitives";
import { ProviderDetail } from "@/components/fwa/ProviderDetail";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { type ProviderRecord } from "@/lib/fwa";
import { providerResultToRecord, scoreProvider } from "@/lib/api";

export const Route = createFileRoute("/providers/single")({
  head: () => ({
    meta: [
      { title: "Single Provider Analysis — FWA Risk Investigator" },
      { name: "description", content: "Score one provider and review peer deviation, utilization and AI explainability." },
      { property: "og:title", content: "Single Provider Analysis — FWA Risk Investigator" },
      { property: "og:description", content: "Score one provider and review peer deviation, utilization and AI explainability." },
    ],
  }),
  component: SingleProviderPage,
});

const DEFAULTS = {
  providerId: "PRV-41127",
  totalClaims: "1840",
  uniqueBeneficiaries: "612",
  averageClaimAmount: "3496",
  highValueClaimsPct: "15",
  chronicComplexPct: "24",
  repeatMultiplePct: "8",
  inpatientClaimSharePct: "32",
};

function SingleProviderPage() {
  const [form, setForm] = useState(DEFAULTS);
  const [provider, setProvider] = useState<ProviderRecord | null>(null);
  const [analysing, setAnalysing] = useState(false);

  function set(key: keyof typeof DEFAULTS, value: string) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function analyse() {
    setAnalysing(true);
    const totalClaims = Number(form.totalClaims) || 0;
    const avgClaimAmount = Number(form.averageClaimAmount) || 0;
    const highValueClaimsPct = Number(form.highValueClaimsPct) || 0;
    const chronicComplexPct = Number(form.chronicComplexPct) || 0;
    const repeatMultiplePct = Number(form.repeatMultiplePct) || 0;
    const inpatientClaimSharePct = Number(form.inpatientClaimSharePct) || 0;

    const features: Record<string, unknown> = {
      claim_count: totalClaims,
      cms_total_beneficiaries: Number(form.uniqueBeneficiaries) || 0,
      cms_weighted_avg_payment: avgClaimAmount,
      cms_weighted_avg_submitted_charge: avgClaimAmount * (1 + highValueClaimsPct / 100),
      services_per_beneficiary: chronicComplexPct / 10,
      cms_total_beneficiary_days: totalClaims * (1 + repeatMultiplePct / 100),
      treatment_service_percentile: inpatientClaimSharePct / 100,
    };
    scoreProvider({ provider_npi: form.providerId || null, features })
      .then((result) => setProvider(providerResultToRecord(result)))
      .catch((err) => console.error("Provider scoring failed", err))
      .finally(() => setAnalysing(false));
  }

  return (
    <AppShell>
      <PageHeading
        eyebrow="Provider Investigation"
        title="SINGLE PROVIDER ANALYSIS"
        subtitle="Enter provider attributes to generate a risk profile, dashboard and explainability report."
      />

      <Panel title="Provider Attributes" description="Provider scoring fields" icon={ClipboardList}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <TextField label="Provider ID" value={form.providerId} onChange={(v) => set("providerId", v)} />
          <TextField label="Total Claims" value={form.totalClaims} onChange={(v) => set("totalClaims", v)} />
          <TextField label="Unique Beneficiaries" value={form.uniqueBeneficiaries} onChange={(v) => set("uniqueBeneficiaries", v)} />
          <TextField label="Average Claim Amount" value={form.averageClaimAmount} onChange={(v) => set("averageClaimAmount", v)} />
          <TextField label="High-Value Claims %" value={form.highValueClaimsPct} onChange={(v) => set("highValueClaimsPct", v)} />
          <TextField label="Chronic/Complex Cases %" value={form.chronicComplexPct} onChange={(v) => set("chronicComplexPct", v)} />
          <TextField label="Repeat/Multiple Claims %" value={form.repeatMultiplePct} onChange={(v) => set("repeatMultiplePct", v)} />
          <TextField label="Inpatient Claim Share %" value={form.inpatientClaimSharePct} onChange={(v) => set("inpatientClaimSharePct", v)} />
        </div>

        <div className="mt-6 flex justify-end">
          <Button
            onClick={analyse}
            disabled={analysing}
            className="rounded-full px-7 py-5 text-sm font-semibold tracking-[0.1em] uppercase shadow-float transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lift"
          >
            <Sparkles className="size-4" />
            {analysing ? "Analysing…" : "Analyse Provider"}
          </Button>
        </div>
      </Panel>

      {provider ? (
        <div className="mt-8 animate-rise">
          <ProviderDetail provider={provider} />
        </div>
      ) : null}
    </AppShell>
  );
}


function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </Label>
      <Input value={value} onChange={(e) => onChange(e.target.value)} className="h-11 rounded-xl" />
    </div>
  );
}

function CheckField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-3 text-sm text-foreground">
      <Checkbox checked={checked} onCheckedChange={(v) => onChange(Boolean(v))} />
      {label}
    </label>
  );
}

function ToggleField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-3 text-sm text-foreground">
      <Switch checked={checked} onCheckedChange={onChange} />
      {label}
    </label>
  );
}
