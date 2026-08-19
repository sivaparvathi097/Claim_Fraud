"""FastAPI application for the FWA Risk Investigator backend.

Run locally with:

    uvicorn app.main:app --reload --port 8000

or:

    python run.py
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from analytical_engine import AnalysisOutcome, decode_claim_type
from analytical_engine.inference import ModelRegistry
from analytical_engine.state import SignalRecord

from . import config, services
from . import llm as llm_boundary
from . import review as review_store
from . import uploads
from .data_access import store
from .services import _clean_float as _opt_float, _clean_str as _opt_str
from .schemas import (
    BatchClaimResponse,
    BatchProviderResponse,
    BatchRequest,
    ClaimExplainRequest,
    ClaimScoreRequest,
    ClaimScoreResult,
    EvidenceItem,
    ExplanationResult,
    FeatureDeviation,
    ProviderExplainRequest,
    ProviderScoreRequest,
    ProviderScoreResult,
    ReviewRecord,
    ReviewRequest,
    RoutingDecision,
    Signal,
    TrustFlags,
)

registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load both hybrid artifacts once at startup so first requests are fast.
    registry.claim
    registry.provider
    yield


app = FastAPI(
    title="FWA Risk Investigator API",
    description="Hybrid XGBoost + LightGBM Fraud/Waste/Abuse scoring backend with the Analytical Engine.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": ["claim_fwa_hybrid_model", "provider_fwa_hybrid_model"]}


@app.get("/api/models")
def models() -> dict:
    return registry.metadata()


# --------------------------------------------------------------- API mapping
def _signals(records: list[SignalRecord]) -> list[Signal]:
    return [Signal(**r.__dict__) for r in records]


def _evidence(records: list[SignalRecord]) -> list[EvidenceItem]:
    return [EvidenceItem(**r.__dict__) for r in records]


def _deviations(items: list[dict[str, Any]]) -> list[FeatureDeviation]:
    return [FeatureDeviation(**d) for d in items]


def _trust_flags(context: dict[str, Any]) -> TrustFlags | None:
    flags = (context or {}).get("trust_flags")
    return TrustFlags(**flags) if flags else None


def _routing(context: dict[str, Any]) -> RoutingDecision | None:
    """Phase 4 routing decision, produced by app/routing.py AFTER the
    Analytical Engine and echoed unchanged here."""
    decision = (context or {}).get("routing")
    return RoutingDecision(**decision) if decision else None


def _upstream(context: dict[str, Any], ml) -> dict[str, Any]:
    """Phase 3 response fields from the READ-ONLY upstream context.

    raw/calibrated scores and TrustFlags are produced BEFORE the Analytical
    Engine and echoed unchanged; prediction/risk_rating mirror the frozen ML
    output for the Phase 3 contract.
    """
    context = context or {}
    return {
        "raw_ml_risk_score": context.get("raw_ml_risk_score"),
        "calibrated_risk_score": context.get("calibrated_risk_score"),
        "prediction": ml.predicted_class,
        "prediction_label": ml.predicted_class,
        "risk_rating": ml.rating,
        "trust_flags": _trust_flags(context),
    }


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if value is not None and value != "":
            merged[key] = value
    return merged


def _claim_result(outcome: AnalysisOutcome, claim_id: str | None, provider_id: str | None) -> ClaimScoreResult:
    ml = outcome.ml
    row = outcome.prepared_row
    return ClaimScoreResult(
        claim_id=claim_id,
        provider_id=provider_id,
        beneficiary_id=_opt_str(row.get("BENE_ID")),
        claim_type=decode_claim_type(registry.claim, row.get("Claim_Type")),
        predicted_class=ml.predicted_class,
        predicted_class_id=ml.predicted_class_id,
        class_probabilities=ml.class_probabilities,
        risk_score=ml.risk_score,
        rating=ml.rating,
        suspicious=ml.suspicious,
        **_upstream(outcome.upstream_context, ml),
        routing=_routing(outcome.upstream_context),
        submitted_amount=_opt_float(row.get("Claim_Submitted_Amount")),
        allowed_amount=_opt_float(row.get("Claim_Allowed_Amount")),
        payment_amount=_opt_float(row.get("Claim_Payment_Amount")),
        deductible_amount=_opt_float(row.get("Deductible_Amount")),
        service_count=_opt_float(row.get("Service_Count")),
        duration_days=_opt_float(row.get("Claim_Duration_Days")),
        diagnosis_count=_opt_float(row.get("Diagnosis_Count")),
        procedure_count=_opt_float(row.get("Procedure_Count")),
        previous_claim_count=_opt_float(row.get("Previous_Claim_Count")),
        signals=_signals(outcome.signals),
        evidence=_evidence(outcome.evidence),
        top_features=_deviations(outcome.top_feature_deviations),
    )


def _provider_result(outcome: AnalysisOutcome, provider_npi: str | None) -> ProviderScoreResult:
    ml = outcome.ml
    row = outcome.prepared_row
    legit_prob = ml.class_probabilities.get("Legitimate", 0.0)
    return ProviderScoreResult(
        provider_npi=provider_npi,
        provider_state=_opt_str(row.get("provider_state")),
        provider_type=_opt_str(row.get("provider_type")),
        predicted_class=ml.predicted_class,
        predicted_class_id=ml.predicted_class_id,
        class_probabilities=ml.class_probabilities,
        suspicious_probability=round(1.0 - legit_prob, 6),
        risk_score=ml.risk_score,
        rating=ml.rating,
        suspicious=ml.suspicious,
        **_upstream(outcome.upstream_context, ml),
        routing=_routing(outcome.upstream_context),
        total_beneficiaries=_opt_float(row.get("cms_total_beneficiaries")),
        total_services=_opt_float(row.get("cms_total_services")),
        claim_count=_opt_float(row.get("claim_count")),
        weighted_avg_payment=_opt_float(row.get("cms_weighted_avg_payment")),
        services_per_beneficiary=_opt_float(row.get("services_per_beneficiary")),
        peer_deviation_score=_opt_float(row.get("peer_deviation_score")),
        cms_weighted_avg_submitted_charge=_opt_float(row.get("cms_weighted_avg_submitted_charge")),
        cms_total_beneficiary_days=_opt_float(row.get("cms_total_beneficiary_days")),
        treatment_service_percentile=_opt_float(row.get("treatment_service_percentile")),
        signals=_signals(outcome.signals),
        evidence=_evidence(outcome.evidence),
        top_features=_deviations(outcome.top_feature_deviations),
    )


# ---------------------------------------------------------------------- claims
@app.post("/api/claims/score", response_model=ClaimScoreResult)
def score_claim(payload: ClaimScoreRequest) -> ClaimScoreResult:
    try:
        base: dict[str, Any] = {}
        dataset_row = store.get_claim(payload.claim_id) if (payload.enrich_from_dataset and payload.claim_id) else None
        if dataset_row is not None:
            base = dataset_row.to_dict()
        record = _merge(base, payload.features)
        if payload.claim_id is not None:
            record.setdefault("Claim_ID", payload.claim_id)

        # Direct flow: ML inference -> frozen MLResult -> Claim Analytical
        # Engine. Only the CLAIM model is loaded/executed on this path and
        # there is no queue for a single claim.
        outcome = services.analyze_claim(registry.claim, record)
        return _claim_result(
            outcome,
            claim_id=_opt_str(record.get("Claim_ID")) or payload.claim_id,
            provider_id=_opt_str(record.get("Provider_ID")),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Claim scoring failed: {exc}") from exc


# ------------------------------------------------------------------ batches
# Batch input contract:
#   * multipart/form-data with the ACTUAL uploaded CSV file -> the uploaded
#     rows (and nothing else) go through vectorised ML inference.
#   * application/json BatchRequest -> legacy path (explicit rows or a sample
#     drawn from the backing dataset), kept unchanged for tooling/tests.
async def _batch_records(request: Request, parser, model) -> list[dict[str, Any]]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = next((v for v in form.values() if hasattr(v, "file")), None)
        if upload is None:
            raise uploads.UploadValidationError(
                "No file provided. Send the CSV as a multipart form field named 'file'."
            )
        raw = await upload.read()
        try:
            return parser(raw, upload.filename, upload.content_type, model)
        finally:
            await form.close()
    payload = BatchRequest.model_validate(await request.json())
    if payload.rows:
        return payload.rows[: payload.limit]
    if payload.use_dataset:
        sample = (
            store.claim_sample(payload.limit)
            if parser is uploads.parse_claim_upload
            else store.provider_sample(payload.limit)
        )
        return sample.to_dict(orient="records")
    return []


@app.post("/api/claims/batch", response_model=BatchClaimResponse)
async def batch_claims(request: Request) -> BatchClaimResponse:
    try:
        # The records are EXACTLY the uploaded rows (or the legacy JSON input);
        # one vectorized inference pass for the whole batch (claim model
        # only), then the Analytical Engine per row, then the queue sorted by
        # the ORIGINAL ML risk score.
        records = await _batch_records(request, uploads.parse_claim_upload, registry.claim)
        batch = services.analyze_claims_batch(registry.claim, records)
        results = [
            _claim_result(outcome, claim_id, provider_id)
            for outcome, claim_id, provider_id in zip(
                batch["outcomes"],
                batch["identifiers"]["claim_ids"],
                batch["identifiers"]["provider_ids"],
            )
        ]
        ordered = sorted(results, key=lambda r: r.risk_score, reverse=True)
        return BatchClaimResponse(total=batch["total"], results=ordered, queue=batch["queue"], summary=batch["summary"])
    except uploads.UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Batch claim scoring failed: {exc}") from exc


# -------------------------------------------------------------------- providers
@app.post("/api/providers/score", response_model=ProviderScoreResult)
def score_provider(payload: ProviderScoreRequest) -> ProviderScoreResult:
    try:
        base: dict[str, Any] = {}
        dataset_row = store.get_provider(payload.provider_npi) if (payload.enrich_from_dataset and payload.provider_npi) else None
        if dataset_row is not None:
            base = dataset_row.to_dict()
        record = _merge(base, payload.features)
        if payload.provider_npi is not None:
            record.setdefault("provider_npi", payload.provider_npi)

        # Direct flow: ML inference -> frozen MLResult -> Provider Analytical
        # Engine. Only the PROVIDER model is loaded/executed on this path and
        # there is no queue for a single provider.
        outcome = services.analyze_provider(registry.provider, record)
        return _provider_result(outcome, provider_npi=_opt_str(record.get("provider_npi")) or payload.provider_npi)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Provider scoring failed: {exc}") from exc


@app.post("/api/providers/batch", response_model=BatchProviderResponse)
async def batch_providers(request: Request) -> BatchProviderResponse:
    try:
        # The records are EXACTLY the uploaded rows (or the legacy JSON input);
        # one vectorized inference pass for the whole batch (provider model
        # only), then the Analytical Engine per row, then the queue sorted by
        # the ORIGINAL ML risk score.
        records = await _batch_records(request, uploads.parse_provider_upload, registry.provider)
        batch = services.analyze_providers_batch(registry.provider, records)
        results = [
            _provider_result(outcome, npi)
            for outcome, npi in zip(batch["outcomes"], batch["identifiers"]["provider_npis"])
        ]
        ordered = sorted(results, key=lambda r: r.risk_score, reverse=True)
        return BatchProviderResponse(total=batch["total"], results=ordered, queue=batch["queue"], summary=batch["summary"])
    except uploads.UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Batch provider scoring failed: {exc}") from exc


# ------------------------------------------------- upload schema preflight
# The upload card calls these BEFORE scoring so the frontend shows the REAL
# record count and the backend's schema verdict (never an invented number).
# They run the upload parser only — no ML, no calibration, no engine.
@app.post("/api/claims/validate")
async def validate_claims(request: Request) -> dict[str, Any]:
    try:
        records = await _batch_records(request, uploads.parse_claim_upload, registry.claim)
    except uploads.UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"valid": True, "record_count": len(records)}


@app.post("/api/providers/validate")
async def validate_providers(request: Request) -> dict[str, Any]:
    try:
        records = await _batch_records(request, uploads.parse_provider_upload, registry.provider)
    except uploads.UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"valid": True, "record_count": len(records)}


# ---------------------------------------------------------- LLM explanation
# Selected case -> LLM reasoning. The request carries the ALREADY-GENERATED
# analysis result: ML, calibration, Trust Gate and the Analytical Engine are
# NEVER rerun here, and the response echoes the ML prediction / risk scores /
# rating / deterministic route unchanged. The LLM explains the route; it
# never decides it (route integrity is enforced below).
@app.get("/api/llm/status")
def llm_status() -> dict:
    return llm_boundary.llm_status()


def _trust_factor_lines(flags) -> list[str]:
    """Human-readable echo of the TrustFlags from the input result."""
    if flags is None:
        return []
    dump = flags.model_dump() if hasattr(flags, "model_dump") else dict(flags)
    return [f"{name}: {value}" for name, value in dump.items()]


def _explain_response(
    entity_type: str, case_id: str | None, analysis, context: dict
) -> ExplanationResult:
    routing = getattr(analysis, "routing", None)
    # The deterministic route is taken from the INPUT result — it is the only
    # source of truth for routing, before and after the LLM call.
    deterministic_route = routing.route if routing is not None else None
    try:
        reasoning = llm_boundary.request_reasoning(context)
    except llm_boundary.LLMConfigurationError as exc:
        # Clear, safe failure — no fabricated reasoning; the case keeps its
        # deterministic route and remains reviewable.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except llm_boundary.LLMReasoningError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    llm_route = reasoning.pop("llm_route", None)
    caveats = list(reasoning.get("caveats", []))
    # Route integrity: if the LLM echoed a DIFFERENT route, ignore it, keep
    # the deterministic route, and record the mismatch as an audit caveat.
    if (
        llm_route is not None
        and deterministic_route is not None
        and llm_route != deterministic_route
    ):
        caveats.append(
            f"LLM returned route '{llm_route}' which differs from the deterministic "
            f"route '{deterministic_route}'; the deterministic route was retained "
            "and the LLM route ignored."
        )
    reasoning["caveats"] = caveats
    return ExplanationResult(
        case_id=case_id,
        entity_type=entity_type,  # type: ignore[arg-type]
        route=deterministic_route,
        prediction=analysis.predicted_class,
        risk_score=analysis.risk_score,
        risk_rating=analysis.rating,
        raw_ml_risk_score=getattr(analysis, "raw_ml_risk_score", None),
        calibrated_risk_score=getattr(analysis, "calibrated_risk_score", None),
        routing_reasons=list(routing.reasons) if routing is not None else [],
        trust_factors=_trust_factor_lines(getattr(analysis, "trust_flags", None)),
        model_provider=llm_boundary.model_provider_label(),
        **reasoning,
    )


@app.post("/api/claims/explain", response_model=ExplanationResult)
def explain_claim(payload: ClaimExplainRequest) -> ExplanationResult:
    analysis = payload.analysis
    context = llm_boundary.build_claim_context(analysis)
    return _explain_response("claim", analysis.claim_id, analysis, context)


@app.post("/api/providers/explain", response_model=ExplanationResult)
def explain_provider(payload: ProviderExplainRequest) -> ExplanationResult:
    analysis = payload.analysis
    context = llm_boundary.build_provider_context(analysis)
    return _explain_response("provider", analysis.provider_npi, analysis, context)


# -------------------------------------------------------------- human review
# Accept / Reject stores the human decision next to the analysis result. It
# never overwrites the ML prediction or the ML risk score.
@app.post("/api/review/accept", response_model=ReviewRecord)
def review_accept(payload: ReviewRequest) -> ReviewRecord:
    return ReviewRecord(**review_store.record_review(payload.case_id, payload.entity_type, "accept"))


@app.post("/api/review/reject", response_model=ReviewRecord)
def review_reject(payload: ReviewRequest) -> ReviewRecord:
    return ReviewRecord(**review_store.record_review(payload.case_id, payload.entity_type, "reject"))


@app.get("/api/review/{entity_type}/{case_id}", response_model=ReviewRecord)
def review_lookup(entity_type: str, case_id: str) -> ReviewRecord:
    record = review_store.get_review(entity_type, case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No review recorded for this case")
    return ReviewRecord(**record)
