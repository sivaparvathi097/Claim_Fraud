export type Rating = "High" | "Medium" | "Low";
export type ReviewStatus = "Pending Review" | "Accepted" | "Rejected" | "Resubmission Required" | "Auto Approved";

/** Structured output returned by the trained hybrid model + analytical engine. */
export type ModelAnalysis = {
  predictedClass: string;
  classProbabilities: Record<string, number>;
  signals: { name: string; value: number; detail: string }[];
  evidence: { label: string; body: string[] }[];
  topFeatures: { feature: string; value: number; trainingMedian: number; deviation: number }[];
};

export type ClaimRecord = {
  claimId: string;
  providerId: string;
  beneficiaryId: string;
  claimType: string;
  claimAmount: number;
  allowedAmount: number;
  paymentAmount: number;
  deductible: number;
  serviceCount: number;
  duration: number;
  diagnoses?: number;
  procedures?: number;
  previousClaims?: number;
  riskScore: number;
  rating: Rating;
  status: ReviewStatus;
  analysis?: ModelAnalysis;
  /** The raw backend analysis result, used verbatim for LLM explanation. */
  api?: unknown;
};


export type ProviderRecord = {
  providerId: string;
  specialty: string;
  claimCount: number;
  beneficiaryCount: number;
  reimbursedAmount: number;
  paymentPerBeneficiary: number;
  daysAdmitted: number;
  peerDeviation: number;
  utilization: number;
  highValueClaimsPct?: number;
  chronicComplexPct?: number;
  repeatMultiplePct?: number;
  inpatientClaimSharePct?: number;
  riskScore: number;
  rating: Rating;
  status: ReviewStatus;
  analysis?: ModelAnalysis;
  /** The raw backend analysis result, used verbatim for LLM explanation. */
  api?: unknown;
};

const CLAIM_TYPES = ["Inpatient", "Outpatient", "Professional", "Pharmacy", "DME"];
const SPECIALTIES = [
  "Internal Medicine",
  "Orthopedics",
  "Cardiology",
  "Diagnostic Lab",
  "Home Health",
  "Physical Therapy",
];

/** Deterministic pseudo-random generator so results stay stable across renders. */
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashString(value: string) {
  let h = 2166136261;
  for (let i = 0; i < value.length; i++) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function ratingFromScore(score: number): Rating {
  if (score >= 75) return "High";
  if (score >= 45) return "Medium";
  return "Low";
}

export function isSuspicious(score: number) {
  return score >= 65;
}

export function assessmentFromScore(score: number) {
  return isSuspicious(score) ? "Potentially Suspicious" : "No Significant Anomaly Detected";
}

export function money(value: number) {
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

export function generateClaims(count: number, seed: number): ClaimRecord[] {
  const rand = mulberry(seed);
  return Array.from({ length: count }, (_, i) => {
    const riskScore = Math.round(12 + rand() * 88);
    const claimAmount = Math.round(400 + rand() * 46000);
    const allowedAmount = Math.round(claimAmount * (0.62 + rand() * 0.33));
    return {
      claimId: `CLM-${(920000 + i * 37 + Math.floor(rand() * 30)).toString()}`,
      providerId: `PRV-${(41000 + Math.floor(rand() * 260)).toString()}`,
      beneficiaryId: `BEN-${(700000 + Math.floor(rand() * 90000)).toString()}`,
      claimType: CLAIM_TYPES[Math.floor(rand() * CLAIM_TYPES.length)]!,
      claimAmount,
      allowedAmount,
      paymentAmount: Math.round(allowedAmount * (0.8 + rand() * 0.2)),
      deductible: Math.round(rand() * 1600),
      serviceCount: 1 + Math.floor(rand() * 22),
      duration: 1 + Math.floor(rand() * 26),
      riskScore,
      rating: ratingFromScore(riskScore),
      status: "Pending Review" as ReviewStatus,
    };
  });
}

export function generateProviders(count: number, seed: number): ProviderRecord[] {
  const rand = mulberry(seed);
  return Array.from({ length: count }, (_, i) => {
    const riskScore = Math.round(10 + rand() * 90);
    const claimCount = 40 + Math.floor(rand() * 2400);
    const beneficiaryCount = 20 + Math.floor(rand() * 900);
    const reimbursedAmount = Math.round(claimCount * (280 + rand() * 900));
    return {
      providerId: `PRV-${(41000 + i * 7 + Math.floor(rand() * 5)).toString()}`,
      specialty: SPECIALTIES[Math.floor(rand() * SPECIALTIES.length)]!,
      claimCount,
      beneficiaryCount,
      reimbursedAmount,
      paymentPerBeneficiary: Math.round(reimbursedAmount / beneficiaryCount),
      daysAdmitted: Math.floor(rand() * 900),
      peerDeviation: Math.round((rand() * 320 - 40) * 10) / 10,
      utilization: Math.round((0.4 + rand() * 2.4) * 100) / 100,
      riskScore,
      rating: ratingFromScore(riskScore),
      status: "Pending Review" as ReviewStatus,
    };
  });
}

export type Series = { name: string; value: number }[];

export function riskDistribution(scores: number[]): Series {
  return [
    { name: "Low", value: scores.filter((s) => s < 45).length },
    { name: "Medium", value: scores.filter((s) => s >= 45 && s < 75).length },
    { name: "High", value: scores.filter((s) => s >= 75).length },
  ];
}

export function bucketBy<T>(rows: T[], key: (row: T) => string): Series {
  const map = new Map<string, number>();
  for (const row of rows) map.set(key(row), (map.get(key(row)) ?? 0) + 1);
  return [...map.entries()].map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
}

export function amountBuckets(values: number[]): Series {
  const edges = [1000, 5000, 15000, 30000, Infinity];
  const labels = ["<1K", "1–5K", "5–15K", "15–30K", "30K+"];
  return labels.map((name, i) => ({
    name,
    value: values.filter((v) => v < edges[i]! && (i === 0 || v >= edges[i - 1]!)).length,
  }));
}

export function peerSeries(seed: number) {
  const rand = mulberry(seed);
  return ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"].map((name) => ({
    name,
    subject: Math.round(60 + rand() * 90),
    peer: Math.round(45 + rand() * 30),
  }));
}

export function signalSeries(seed: number) {
  const rand = mulberry(seed + 991);
  return [
    "Upcoding",
    "Volume Spike",
    "Peer Deviation",
    "Duplicate Pattern",
    "Utilization",
    "Temporal Anomaly",
  ].map((name) => ({ name, value: Math.round(20 + rand() * 80) }));
}

export const RISK_COLORS: Record<Rating, string> = {
  High: "var(--risk-high)",
  Medium: "var(--risk-medium)",
  Low: "var(--risk-low)",
};
