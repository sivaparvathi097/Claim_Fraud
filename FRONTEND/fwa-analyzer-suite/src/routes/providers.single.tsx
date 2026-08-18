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
  specialty: "Internal Medicine",
  claimCount: "1840",
  providerBeneficiaryCount: "612",
  beneficiaryCount: "588",
  reimbursed: "2140000",
  deductible: "142000",
  daysAdmitted: "684",
  paymentPerBeneficiary: "3496",
  peerDeviation: "118",
  utilization: "1.84",
};

function SingleProviderPage() {
  const [form, setForm] = useState(DEFAULTS);
  const [flags, setFlags] = useState({
    priorInvestigation: true,
    highRiskSpecialty: true,
    sanctionList: false,
    telehealthHeavy: false,
    networkCluster: true,
  });
  const [provider, setProvider] = useState<ProviderRecord | null>(null);
  const [analysing, setAnalysing] = useState(false);

  function set(key: keyof typeof DEFAULTS, value: string) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function analyse() {
    setAnalysing(true);
    const features: Record<string, unknown> = {
      provider_type: form.specialty,
      claim_count: Number(form.claimCount) || 0,
      cms_total_beneficiaries: Number(form.beneficiaryCount) || 0,
      cms_total_beneficiary_days: Number(form.daysAdmitted) || 0,
      services_per_beneficiary: Number(form.utilization) || 0,
      peer_deviation_score: (Number(form.peerDeviation) || 0) / 100,
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

      <Panel title="Provider Attributes" description="Core scoring fields" icon={ClipboardList}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <TextField label="Provider ID / NPI" value={form.providerId} onChange={(v) => set("providerId", v)} />
          <TextField label="Specialty" value={form.specialty} onChange={(v) => set("specialty", v)} />
          <TextField label="Provider Claim Count" value={form.claimCount} onChange={(v) => set("claimCount", v)} />
          <TextField
            label="Provider Beneficiary Count"
            value={form.providerBeneficiaryCount}
            onChange={(v) => set("providerBeneficiaryCount", v)}
          />
          <TextField label="Beneficiary Count" value={form.beneficiaryCount} onChange={(v) => set("beneficiaryCount", v)} />
          <TextField label="Reimbursed Amount" value={form.reimbursed} onChange={(v) => set("reimbursed", v)} />
          <TextField label="Deductible Amount" value={form.deductible} onChange={(v) => set("deductible", v)} />
          <TextField label="Days Admitted" value={form.daysAdmitted} onChange={(v) => set("daysAdmitted", v)} />
          <TextField
            label="Payment Per Beneficiary"
            value={form.paymentPerBeneficiary}
            onChange={(v) => set("paymentPerBeneficiary", v)}
          />
          <TextField label="Peer Deviation (%)" value={form.peerDeviation} onChange={(v) => set("peerDeviation", v)} />
          <TextField label="Utilization Indicator" value={form.utilization} onChange={(v) => set("utilization", v)} />
        </div>

        <div className="mt-6 grid gap-3 rounded-xl border border-border/70 bg-secondary/40 p-4 sm:grid-cols-2 lg:grid-cols-3">
          <CheckField
            label="Prior investigation history"
            checked={flags.priorInvestigation}
            onChange={(v) => setFlags({ ...flags, priorInvestigation: v })}
          />
          <CheckField
            label="High-risk specialty"
            checked={flags.highRiskSpecialty}
            onChange={(v) => setFlags({ ...flags, highRiskSpecialty: v })}
          />
          <CheckField
            label="Sanction list match"
            checked={flags.sanctionList}
            onChange={(v) => setFlags({ ...flags, sanctionList: v })}
          />
          <ToggleField
            label="Telehealth-heavy billing"
            checked={flags.telehealthHeavy}
            onChange={(v) => setFlags({ ...flags, telehealthHeavy: v })}
          />
          <ToggleField
            label="Shared beneficiary cluster"
            checked={flags.networkCluster}
            onChange={(v) => setFlags({ ...flags, networkCluster: v })}
          />
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
