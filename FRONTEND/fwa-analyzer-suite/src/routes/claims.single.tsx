import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ClipboardList, Sparkles } from "lucide-react";
import { AppShell } from "@/components/fwa/AppShell";
import { PageHeading, Panel } from "@/components/fwa/primitives";
import { ClaimDetail } from "@/components/fwa/ClaimDetail";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type ClaimRecord } from "@/lib/fwa";
import { CLAIM_TYPE_MAP, claimResultToRecord, scoreClaim } from "@/lib/api";

export const Route = createFileRoute("/claims/single")({
  head: () => ({
    meta: [
      { title: "Single Claim Analysis — FWA Risk Investigator" },
      { name: "description", content: "Score an individual claim and review AI explainability with expert sign-off." },
      { property: "og:title", content: "Single Claim Analysis — FWA Risk Investigator" },
      { property: "og:description", content: "Score an individual claim and review AI explainability with expert sign-off." },
    ],
  }),
  component: SingleClaimPage,
});

const DEFAULTS = {
  claimId: "CLM-920481",
  providerId: "PRV-41127",
  beneficiaryId: "BEN-742930",
  claimType: "Inpatient",
  duration: "14",
  submitted: "28400",
  allowed: "21300",
  payment: "19850",
  deductible: "1200",
  serviceCount: "11",
};

function SingleClaimPage() {
  const [form, setForm] = useState(DEFAULTS);
  const [flags, setFlags] = useState({
    priorDenial: true,
    outOfNetwork: false,
    duplicateSuspected: true,
    priorAuth: false,
    emergency: false,
  });
  const [claim, setClaim] = useState<ClaimRecord | null>(null);
  const [analysing, setAnalysing] = useState(false);

  function set(key: keyof typeof DEFAULTS, value: string) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function analyse() {
    setAnalysing(true);
    const features: Record<string, unknown> = {
      Claim_Type: CLAIM_TYPE_MAP[form.claimType] ?? form.claimType.toLowerCase(),
      Claim_Duration_Days: Number(form.duration) || 0,
      Claim_Submitted_Amount: Number(form.submitted) || 0,
      Claim_Allowed_Amount: Number(form.allowed) || 0,
      Claim_Payment_Amount: Number(form.payment) || 0,
      Deductible_Amount: Number(form.deductible) || 0,
      Service_Count: Number(form.serviceCount) || 0,
    };
    scoreClaim({ claim_id: form.claimId || null, features })
      .then((result) => setClaim(claimResultToRecord(result)))
      .catch((err) => console.error("Claim scoring failed", err))
      .finally(() => setAnalysing(false));
  }

  return (
    <AppShell>
      <PageHeading
        eyebrow="Claim Investigation"
        title="SINGLE CLAIM ANALYSIS"
        subtitle="Provide claim attributes to generate a risk score, dashboard and explainability report."
      />

      <Panel title="Claim Attributes" description="10 core scoring fields" icon={ClipboardList}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          <TextField label="Claim ID" value={form.claimId} onChange={(v) => set("claimId", v)} />
          <TextField label="Provider ID" value={form.providerId} onChange={(v) => set("providerId", v)} />
          <TextField label="Beneficiary ID" value={form.beneficiaryId} onChange={(v) => set("beneficiaryId", v)} />
          <div className="space-y-1.5">
            <Label className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              Claim Type
            </Label>
            <Select value={form.claimType} onValueChange={(v) => set("claimType", v)}>
              <SelectTrigger className="h-11 w-full rounded-xl">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {["Inpatient", "Outpatient", "Professional", "Pharmacy", "DME"].map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <TextField label="Claim Duration (days)" value={form.duration} onChange={(v) => set("duration", v)} />
          <TextField label="Submitted Amount" value={form.submitted} onChange={(v) => set("submitted", v)} />
          <TextField label="Allowed Amount" value={form.allowed} onChange={(v) => set("allowed", v)} />
          <TextField label="Payment Amount" value={form.payment} onChange={(v) => set("payment", v)} />
          <TextField label="Deductible Amount" value={form.deductible} onChange={(v) => set("deductible", v)} />
          <TextField label="Service Count" value={form.serviceCount} onChange={(v) => set("serviceCount", v)} />
        </div>

        <div className="mt-6 grid gap-3 rounded-xl border border-border/70 bg-secondary/40 p-4 sm:grid-cols-2 lg:grid-cols-3">
          <CheckField
            label="Prior denial on record"
            checked={flags.priorDenial}
            onChange={(v) => setFlags({ ...flags, priorDenial: v })}
          />
          <CheckField
            label="Out-of-network service"
            checked={flags.outOfNetwork}
            onChange={(v) => setFlags({ ...flags, outOfNetwork: v })}
          />
          <CheckField
            label="Duplicate pattern suspected"
            checked={flags.duplicateSuspected}
            onChange={(v) => setFlags({ ...flags, duplicateSuspected: v })}
          />
          <ToggleField
            label="Prior authorization obtained"
            checked={flags.priorAuth}
            onChange={(v) => setFlags({ ...flags, priorAuth: v })}
          />
          <ToggleField
            label="Emergency admission"
            checked={flags.emergency}
            onChange={(v) => setFlags({ ...flags, emergency: v })}
          />
        </div>

        <div className="mt-6 flex justify-end">
          <Button
            onClick={analyse}
            disabled={analysing}
            className="rounded-full px-7 py-5 text-sm font-semibold tracking-[0.1em] uppercase shadow-float transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lift"
          >
            <Sparkles className="size-4" />
            {analysing ? "Analysing…" : "Analyse Claim"}
          </Button>
        </div>
      </Panel>

      {claim ? (
        <div className="mt-8 animate-rise">
          <ClaimDetail claim={claim} />
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
