/**
 * API client for the FWA Risk Investigator backend (FastAPI).
 *
 * The backend runs the trained XGBoost + LightGBM hybrid models and returns the
 * prediction, a 0-100 risk score, and the analytical-engine signals/evidence.
 */
import type { ClaimRecord, ModelAnalysis, ProviderRecord, Rating } from "./fwa";

const API_BASE = (import.meta.env['VITE_API_BASE_URL'] as string | undefined) ?? "http://localhost:8000";

// -------------------------------------------------------------------- response
export interface ApiSignal {
  signal: string;
  value: number;
  interpretation: string;
  supporting_attributes: string[];
  severity: "High" | "Medium" | "Low";
  attribute_values: Record<string, number>;
}

export interface ApiEvidenceItem {
  signal: string;
  value: number;
  interpretation: string;
  supporting_attributes: string[];
  severity: "High" | "Medium" | "Low";
  attribute_values: Record<string, number>;
}

export interface ApiTopFeature {
  feature: string;
  value: number;
  training_median: number;
  deviation: number;
}

/** Phase 2 Trust Gate metadata (Phase 3: exposed in the API response). */
export interface ApiTrustFlags {
  model_agreement: number;
  low_agreement: boolean;
  confidence: number;
  low_confidence: boolean;
  in_distribution: boolean;
  out_of_distribution: boolean;
  feature_completeness: boolean;
}

/** Phase 4 deterministic routing decision (rule-based, after the Analytical Engine). */
export type ApiRoute = "auto_approve" | "fast_track" | "full_investigation";

export interface ApiRoutingDecision {
  route: ApiRoute;
  reasons: string[];
  requires_human_review: boolean;
  predicate_results: Record<string, boolean>;
}

export interface ApiClaimResult {
  claim_id: string | null;
  provider_id: string | null;
  beneficiary_id: string | null;
  claim_type: string | null;
  predicted_class: string;
  predicted_class_id: number;
  class_probabilities: Record<string, number>;
  risk_score: number;
  rating: Rating;
  suspicious: boolean;
  // Phase 3 upstream context (optional — added fields only, UI unchanged)
  raw_ml_risk_score?: number | null;
  calibrated_risk_score?: number | null;
  prediction?: string | null;
  prediction_label?: string | null;
  risk_rating?: Rating | null;
  trust_flags?: ApiTrustFlags | null;
  // Phase 4 deterministic routing decision (optional — added field only, UI unchanged)
  routing?: ApiRoutingDecision | null;
  submitted_amount: number | null;
  allowed_amount: number | null;
  payment_amount: number | null;
  deductible_amount: number | null;
  service_count: number | null;
  duration_days: number | null;
  diagnosis_count?: number | null;
  procedure_count?: number | null;
  previous_claim_count?: number | null;
  signals: ApiSignal[];

  evidence: ApiEvidenceItem[];
  top_features: ApiTopFeature[];
  status: string;
}

export interface ApiProviderResult {
  provider_npi: string | null;
  provider_state: string | null;
  provider_type: string | null;
  predicted_class: string;
  predicted_class_id: number;
  class_probabilities: Record<string, number>;
  suspicious_probability: number;
  risk_score: number;
  rating: Rating;
  suspicious: boolean;
  // Phase 3 upstream context (optional — added fields only, UI unchanged)
  raw_ml_risk_score?: number | null;
  calibrated_risk_score?: number | null;
  prediction?: string | null;
  prediction_label?: string | null;
  risk_rating?: Rating | null;
  trust_flags?: ApiTrustFlags | null;
  // Phase 4 deterministic routing decision (optional — added field only, UI unchanged)
  routing?: ApiRoutingDecision | null;
  total_beneficiaries: number | null;
  total_services: number | null;
  claim_count: number | null;
  weighted_avg_payment: number | null;
  services_per_beneficiary: number | null;
  peer_deviation_score: number | null;
  cms_weighted_avg_submitted_charge?: number | null;
  cms_total_beneficiary_days?: number | null;
  treatment_service_percentile?: number | null;
  signals: ApiSignal[];

  evidence: ApiEvidenceItem[];
  top_features: ApiTopFeature[];
  status: string;
}

export interface ApiBatchClaim {
  total: number;
  results: ApiClaimResult[];
  queue: { rank: number; claim_id: string | null; provider_id: string | null; risk_score: number; rating: Rating; predicted_class: string; status: string }[];
  summary: Record<string, number> & { routing_counts?: Record<string, number> };
}

export interface ApiBatchProvider {
  total: number;
  results: ApiProviderResult[];
  queue: { rank: number; provider_npi: string | null; provider_type: string | null; risk_score: number; rating: Rating; predicted_class: string; status: string }[];
  summary: Record<string, number> & { routing_counts?: Record<string, number> };
}

// --------------------------------------------------------------------- client
async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const raw = await response.text().catch(() => "");
    let detail = raw;
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown };
      if (parsed.detail !== undefined) detail = String(parsed.detail);
    } catch {
      /* keep raw body */
    }
    throw new Error(detail || `Backend ${path} failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Backend ${path} failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export interface ClaimScorePayload {
  claim_id?: string | null;
  features?: Record<string, unknown>;
  enrich_from_dataset?: boolean;
}

export interface ProviderScorePayload {
  provider_npi?: string | null;
  features?: Record<string, unknown>;
  enrich_from_dataset?: boolean;
}

export interface BatchPayload {
  use_dataset?: boolean;
  limit?: number;
  rows?: Record<string, unknown>[];
}

export const scoreClaim = (payload: ClaimScorePayload) => post<ApiClaimResult>("/api/claims/score", payload);
export const batchClaims = (payload: BatchPayload) => post<ApiBatchClaim>("/api/claims/batch", payload);
export const scoreProvider = (payload: ProviderScorePayload) => post<ApiProviderResult>("/api/providers/score", payload);
export const batchProviders = (payload: BatchPayload) => post<ApiBatchProvider>("/api/providers/batch", payload);

// ------------------------------------------------------------ CSV file upload
// Multipart upload of the ACTUAL selected CSV — the backend scores exactly the
// uploaded rows (never a training-dataset sample). No dataset paths are
// hardcoded here; the browser only ever sends the user's file.
async function uploadCsv<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  if (!response.ok) {
    const raw = await response.text().catch(() => "");
    let detail = raw;
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown };
      if (parsed.detail !== undefined) detail = String(parsed.detail);
    } catch {
      /* keep raw body */
    }
    throw new Error(detail || `Upload to ${path} failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const uploadClaimsCsv = (file: File) => uploadCsv<ApiBatchClaim>("/api/claims/batch", file);
export const uploadProvidersCsv = (file: File) => uploadCsv<ApiBatchProvider>("/api/providers/batch", file);

// Schema preflight: the upload card validates the ACTUAL file against the
// backend model contract before scoring — record count and verdict come from
// the backend, never from the client.
export interface ApiValidation {
  valid: boolean;
  record_count: number;
}

export const validateClaimsCsv = (file: File) => uploadCsv<ApiValidation>("/api/claims/validate", file);
export const validateProvidersCsv = (file: File) => uploadCsv<ApiValidation>("/api/providers/validate", file);

/** Deterministic route carried by a mapped record (null when absent). */
export function recordRoute(api: unknown): ApiRoute | null {
  return (api as { routing?: ApiRoutingDecision | null } | undefined)?.routing?.route ?? null;
}

/** Human-review requirement of a mapped record; defaults to true when absent. */
export function recordRequiresReview(api: unknown): boolean {
  const routing = (api as { routing?: ApiRoutingDecision | null } | undefined)?.routing;
  return routing ? routing.requires_human_review : true;
}

// ------------------------------------------------- LLM explanation + review
// The explain endpoints receive the ALREADY-GENERATED analysis result — the
// backend never reruns ML or the Analytical Engine for an explanation.
export interface ApiExplanation {
  case_id: string | null;
  entity_type: "claim" | "provider";
  // Deterministic route echoed from the input result — the LLM never decides it.
  route: ApiRoute | null;
  prediction: string;
  risk_score: number;
  risk_rating: Rating;
  raw_ml_risk_score: number | null;
  calibrated_risk_score: number | null;
  summary: string;
  reasoning: string[];
  supporting_signals: string[];
  evidence_used: string[];
  routing_reasons: string[];
  trust_factors: string[];
  caveats: string[];
  model_provider: string;
}

export interface ApiReviewRecord {
  case_id: string;
  entity_type: "claim" | "provider";
  review_status: "Accepted" | "Rejected";
  reviewed_at: string;
  reviewer_action: "accept" | "reject";
}

export interface ReviewPayload {
  case_id: string;
  entity_type: "claim" | "provider";
}

export const explainClaim = (payload: { analysis: ApiClaimResult }) =>
  post<ApiExplanation>("/api/claims/explain", payload);
export const explainProvider = (payload: { analysis: ApiProviderResult }) =>
  post<ApiExplanation>("/api/providers/explain", payload);
export const reviewAccept = (payload: ReviewPayload) => post<ApiReviewRecord>("/api/review/accept", payload);
export const reviewReject = (payload: ReviewPayload) => post<ApiReviewRecord>("/api/review/reject", payload);
export const llmStatus = () => get<{ configured: boolean; provider: string; model: string }>("/api/llm/status");

// -------------------------------------------------------------------- mapping
function toAnalysis(r: { predicted_class: string; class_probabilities: Record<string, number>; signals: ApiSignal[]; evidence: ApiEvidenceItem[]; top_features: ApiTopFeature[] }): ModelAnalysis {
  return {
    predictedClass: r.predicted_class,
    classProbabilities: r.class_probabilities,
    signals: r.signals.map((s) => ({ name: s.signal, value: s.value, detail: s.interpretation })),
    evidence: r.evidence.map((e) => ({
      label: e.signal,
      body: [
        e.interpretation,
        ...Object.entries(e.attribute_values).map(([attr, value]) => `${attr}: ${Number(value).toLocaleString()}`),
        `Signal strength ${e.value}/100 \u00b7 severity ${e.severity}`,
      ],
    })),
    topFeatures: r.top_features.map((t) => ({
      feature: t.feature,
      value: t.value,
      trainingMedian: t.training_median,
      deviation: t.deviation,
    })),
  };
}

export function claimResultToRecord(r: ApiClaimResult): ClaimRecord {
  return {
    claimId: r.claim_id ?? "—",
    providerId: r.provider_id ?? "—",
    beneficiaryId: r.beneficiary_id ?? "—",
    claimType: r.claim_type ?? "—",
    claimAmount: r.submitted_amount ?? 0,
    allowedAmount: r.allowed_amount ?? 0,
    paymentAmount: r.payment_amount ?? 0,
    deductible: r.deductible_amount ?? 0,
    serviceCount: r.service_count ?? 0,
    duration: r.duration_days ?? 0,
    diagnoses: r.diagnosis_count ?? 0,
    procedures: r.procedure_count ?? 0,
    previousClaims: r.previous_claim_count ?? 0,
    riskScore: Math.round(r.risk_score),
    rating: r.rating,
    status: r.routing && !r.routing.requires_human_review ? "Auto Approved" : "Pending Review",
    analysis: toAnalysis(r),
    api: r,
  };
}


export function providerResultToRecord(r: ApiProviderResult): ProviderRecord {
  const beneficiaries = r.total_beneficiaries ?? 0;
  const services = r.total_services ?? 0;
  const avgPayment = r.weighted_avg_payment ?? 0;
  const reimbursed = avgPayment * services;
  
  const submitted = r.cms_weighted_avg_submitted_charge ?? 0;
  const highValueClaimsPct = avgPayment > 0 ? Math.round(((submitted / avgPayment) - 1) * 100) : 0;
  
  const chronicComplexPct = Math.round((r.services_per_beneficiary ?? 0) * 10);
  
  const days = r.cms_total_beneficiary_days ?? 0;
  const claimCount = r.claim_count ?? 0;
  const repeatMultiplePct = claimCount > 0 ? Math.round(((days / claimCount) - 1) * 100) : 0;
  
  const inpatientClaimSharePct = Math.round((r.treatment_service_percentile ?? 0) * 100);

  return {
    providerId: r.provider_npi ?? "—",
    specialty: r.provider_type ?? "—",
    claimCount: r.claim_count ?? 0,
    beneficiaryCount: beneficiaries,
    reimbursedAmount: reimbursed,
    paymentPerBeneficiary: beneficiaries > 0 ? reimbursed / beneficiaries : 0,
    daysAdmitted: days,
    peerDeviation: Math.round((r.peer_deviation_score ?? 0) * 100),
    utilization: r.services_per_beneficiary ?? 0,
    riskScore: Math.round(r.risk_score),
    rating: r.rating,
    status: r.routing && !r.routing.requires_human_review ? "Auto Approved" : "Pending Review",
    highValueClaimsPct,
    chronicComplexPct,
    repeatMultiplePct,
    inpatientClaimSharePct,
    analysis: toAnalysis(r),
    api: r,
  };
}


/** Map the UI's friendly claim-type labels to the model's trained categories. */
export const CLAIM_TYPE_MAP: Record<string, string> = {
  Inpatient: "inpatient",
  Outpatient: "outpatient",
  DME: "dme",
  "Home Health": "hha",
  Hospice: "hospice",
  SNF: "snf",
  Carrier: "carrier",
};
