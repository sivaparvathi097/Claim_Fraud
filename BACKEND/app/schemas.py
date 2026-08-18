"""Pydantic request/response schemas for the FWA API.

Responses separate three distinct output types (never combined):
  * the ML model output  — prediction, class probabilities, risk score, rating
  * Analytical Engine signals — deterministic supporting signals (0-100)
  * feature deviations — vs the training medians stored in the artifact
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Rating = Literal["High", "Medium", "Low"]
Severity = Literal["High", "Medium", "Low"]
ReviewStatus = Literal["Pending Review", "Accepted", "Rejected", "Resubmission Required"]
EntityType = Literal["claim", "provider"]
ReviewerAction = Literal["accept", "reject"]
Route = Literal["auto_approve", "fast_track", "full_investigation"]


# --------------------------------------------------------------------- requests
class ClaimScoreRequest(BaseModel):
    """Score a single claim.

    ``claim_id`` — optional. ``features`` — optional mapping of model feature
    name -> value (model's exact feature names). With ``enrich_from_dataset``
    and a known claim_id, the full dataset row is used as the feature base.
    """

    claim_id: str | None = Field(default=None, description="Optional Claim_ID")
    features: dict[str, Any] = Field(default_factory=dict, description="Model feature overrides")
    enrich_from_dataset: bool = Field(
        default=False,
        description="If True and claim_id exists in the dataset, load the full feature row as the base",
    )


class ProviderScoreRequest(BaseModel):
    provider_npi: str | None = Field(default=None, description="Optional provider_npi")
    features: dict[str, Any] = Field(default_factory=dict, description="Model feature overrides")
    enrich_from_dataset: bool = Field(
        default=False,
        description="If True and provider_npi exists in the dataset, load the full feature row as the base",
    )


class BatchRequest(BaseModel):
    """Batch scoring — JSON records per the existing frontend requirements.

    Either score explicit ``rows`` (list of feature dicts using model feature
    names) or a sample drawn from the backing dataset (``use_dataset=True``).
    """

    use_dataset: bool = Field(default=True, description="Score a sample from the backing dataset")
    limit: int = Field(default=50, ge=1, le=500, description="Number of rows to score")
    rows: list[dict[str, Any]] | None = Field(default=None, description="Explicit rows to score")


# ------------------------------------------------------- analytical engine types
class Signal(BaseModel):
    """One deterministic Analytical Engine signal (NOT a model score)."""

    signal: str
    value: float = Field(description="Signal strength 0-100")
    interpretation: str = ""
    supporting_attributes: list[str] = Field(default_factory=list)
    severity: Severity = "Low"
    attribute_values: dict[str, float] = Field(default_factory=dict)


class EvidenceItem(Signal):
    """Structured evidence package item — same shape as a signal, ranked."""


class FeatureDeviation(BaseModel):
    feature: str
    value: float
    training_median: float
    deviation: float


class TrustFlags(BaseModel):
    """Phase 2 Trust Gate metadata — reliability/context facts about the
    prediction. Never a risk score, never consumed by signal calculations."""

    model_agreement: float
    low_agreement: bool
    confidence: float
    low_confidence: bool
    in_distribution: bool
    out_of_distribution: bool
    feature_completeness: bool


class RoutingDecision(BaseModel):
    """Phase 4 deterministic routing decision — produced AFTER the Analytical
    Engine by explicit configurable predicates (never the LLM). Adds nothing
    to the ML output, TrustFlags, signals or evidence."""

    route: Route
    reasons: list[str] = Field(description="Auditable, human-readable predicate outcomes")
    requires_human_review: bool = Field(
        description="False only for auto_approve; auto_approve does NOT bypass the future LLM explanation"
    )
    predicate_results: dict[str, bool] = Field(description="Every predicate individually represented")


# --------------------------------------------------------------------- responses
class ClaimScoreResult(BaseModel):
    claim_id: str | None = None
    provider_id: str | None = None
    beneficiary_id: str | None = None
    claim_type: str | None = None

    # ML model output (immutable through the pipeline)
    predicted_class: str = Field(description="Prediction label")
    predicted_class_id: int = Field(description="Prediction")
    class_probabilities: dict[str, float]
    risk_score: float = Field(description="0-100 risk score: (1 - P(Legitimate)) * 100")
    rating: Rating
    suspicious: bool

    # Phase 3 upstream context — raw ML risk preserved, calibrated companion,
    # Trust Gate metadata. `prediction`/`prediction_label` mirror
    # predicted_class and `risk_rating` mirrors rating for the Phase 3
    # contract; the existing fields remain the source of truth.
    raw_ml_risk_score: float | None = Field(
        default=None, description="Raw ML risk before calibration (never modified)"
    )
    calibrated_risk_score: float | None = Field(
        default=None, description="Output of the stored calibration artifact"
    )
    prediction: str | None = Field(default=None, description="Predicted class label")
    prediction_label: str | None = Field(default=None, description="Predicted class label")
    risk_rating: Rating | None = Field(default=None, description="Rating band of the raw ML risk")
    trust_flags: TrustFlags | None = Field(default=None, description="Trust Gate metadata")
    routing: RoutingDecision | None = Field(
        default=None, description="Phase 4 deterministic routing decision (rule-based, after the Analytical Engine)"
    )

    # Echoed core scoring fields (for the existing UI)
    submitted_amount: float | None = None
    allowed_amount: float | None = None
    payment_amount: float | None = None
    deductible_amount: float | None = None
    service_count: float | None = None
    duration_days: float | None = None

    # Analytical engine output (distinct from the ML score)
    signals: list[Signal] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    top_features: list[FeatureDeviation] = Field(default_factory=list)

    status: ReviewStatus = "Pending Review"


class ProviderScoreResult(BaseModel):
    provider_npi: str | None = None
    provider_state: str | None = None
    provider_type: str | None = None

    # ML model output (immutable through the pipeline)
    predicted_class: str
    predicted_class_id: int
    class_probabilities: dict[str, float]
    suspicious_probability: float
    risk_score: float = Field(description="0-100 risk score: P(Suspicious) * 100")
    rating: Rating
    suspicious: bool

    # Phase 3 upstream context — see ClaimScoreResult for the field contract.
    raw_ml_risk_score: float | None = Field(
        default=None, description="Raw ML risk before calibration (never modified)"
    )
    calibrated_risk_score: float | None = Field(
        default=None, description="Output of the stored calibration artifact"
    )
    prediction: str | None = Field(default=None, description="Predicted class label")
    prediction_label: str | None = Field(default=None, description="Predicted class label")
    risk_rating: Rating | None = Field(default=None, description="Rating band of the raw ML risk")
    trust_flags: TrustFlags | None = Field(default=None, description="Trust Gate metadata")
    routing: RoutingDecision | None = Field(
        default=None, description="Phase 4 deterministic routing decision (rule-based, after the Analytical Engine)"
    )

    # Echoed core fields
    total_beneficiaries: float | None = None
    total_services: float | None = None
    claim_count: float | None = None
    weighted_avg_payment: float | None = None
    services_per_beneficiary: float | None = None
    peer_deviation_score: float | None = None

    # Analytical engine output (distinct from the ML score)
    signals: list[Signal] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    top_features: list[FeatureDeviation] = Field(default_factory=list)

    status: ReviewStatus = "Pending Review"


class BatchClaimResponse(BaseModel):
    total: int
    results: list[ClaimScoreResult]
    queue: list[dict[str, Any]] = Field(description="Prioritized investigation queue (ML risk score, desc)")
    summary: dict[str, Any]


class BatchProviderResponse(BaseModel):
    total: int
    results: list[ProviderScoreResult]
    queue: list[dict[str, Any]]
    summary: dict[str, Any]


# --------------------------------------------------------- LLM explanation
# The explain endpoints receive the ALREADY-GENERATED analysis result — ML and
# the Analytical Engine are never rerun on this path. The response echoes the
# ML prediction / risk score / rating straight from that input; the LLM only
# adds human-readable reasoning next to them.
class ClaimExplainRequest(BaseModel):
    analysis: ClaimScoreResult = Field(
        description="The already-generated claim analysis result to explain"
    )


class ProviderExplainRequest(BaseModel):
    analysis: ProviderScoreResult = Field(
        description="The already-generated provider analysis result to explain"
    )


class ExplanationResult(BaseModel):
    """Structured LLM reasoning about an already-analyzed case (Phase 5).

    The route and every echoed score/rating/label come from the INPUT
    analysis result — the LLM never decides the route and never produces a
    second risk score."""

    case_id: str | None = None
    entity_type: EntityType
    # Deterministic routing outcome — echoed from the input analysis, never
    # from the LLM. Any mismatching LLM route is ignored in app/main.py.
    route: str | None = Field(
        default=None,
        description="Deterministic route of the explained case (router-owned)",
    )
    # Echoed ML output — taken from the input analysis, never from the LLM.
    prediction: str
    risk_score: float
    risk_rating: Rating
    raw_ml_risk_score: float | None = None
    calibrated_risk_score: float | None = None
    # LLM-authored explanation content.
    summary: str
    reasoning: list[str]
    supporting_signals: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    routing_reasons: list[str] = Field(
        default_factory=list,
        description="Deterministic routing reasons echoed from the input result",
    )
    trust_factors: list[str] = Field(
        default_factory=list,
        description="Trust Gate flags echoed from the input result",
    )
    caveats: list[str] = Field(default_factory=list)
    model_provider: str


# ------------------------------------------------------------- human review
# A review record stores the human decision NEXT TO the analysis result. It
# never overwrites the ML prediction or risk score.
class ReviewRequest(BaseModel):
    case_id: str = Field(min_length=1)
    entity_type: EntityType


class ReviewRecord(BaseModel):
    case_id: str
    entity_type: EntityType
    review_status: Literal["Accepted", "Rejected"]
    reviewed_at: str
    reviewer_action: ReviewerAction
