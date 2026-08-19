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
  beneficiaryId: "BEN-742930",
  claimType: "Inpatient",
  reimbursement: "19850",
  deductible: "1200",
  duration: "14",
  diagnoses: "5",
  procedures: "3",
  previousClaims: "2",
};

function SingleClaimPage() {
  const [form, setForm] = useState(DEFAULTS);
  const [claim, setClaim] = useState<ClaimRecord | null>(null);
  const [analysing, setAnalysing] = useState(false);

  function set(key: keyof typeof DEFAULTS, value: string) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function analyse() {
    setAnalysing(true);
    const claimTypeLower = CLAIM_TYPE_MAP[form.claimType] ?? form.claimType.toLowerCase();
    const reimbursementVal = Number(form.reimbursement) || 0;
    const features: Record<string, unknown> = {
      Claim_Type: claimTypeLower,
      Claim_Duration_Days: form.claimType === "Outpatient" ? 0 : (Number(form.duration) || 0),
      Claim_Payment_Amount: reimbursementVal,
      Claim_Submitted_Amount: reimbursementVal,
      Claim_Allowed_Amount: reimbursementVal,
      Deductible_Amount: Number(form.deductible) || 0,
      Diagnosis_Count: Number(form.diagnoses) || 0,
      Procedure_Count: Number(form.procedures) || 0,
      Previous_Claim_Count: Number(form.previousClaims) || 0,
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

      <Panel title="Claim Attributes" description="Claim scoring fields" icon={ClipboardList}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <TextField label="Claim ID" value={form.claimId} onChange={(v) => set("claimId", v)} />
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
                {["Inpatient", "Outpatient"].map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <TextField label="Claim Reimbursement" value={form.reimbursement} onChange={(v) => set("reimbursement", v)} />
          <TextField label="Deductible Amount" value={form.deductible} onChange={(v) => set("deductible", v)} />
          {form.claimType !== "Outpatient" && (
            <TextField label="Length of Stay" value={form.duration} onChange={(v) => set("duration", v)} />
          )}
          <TextField label="Number of Diagnoses" value={form.diagnoses} onChange={(v) => set("diagnoses", v)} />
          <TextField label="Number of Procedures" value={form.procedures} onChange={(v) => set("procedures", v)} />
          <TextField label="Previous Claims" value={form.previousClaims} onChange={(v) => set("previousClaims", v)} />
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
