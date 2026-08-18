"""Direct analysis flows — the locked architecture, with NO orchestrator.

    Request
        -> validation                        (app/schemas.py, FastAPI)
        -> ML inference                      (inference.predict_claims /
                                              predict_providers — vectorized)
        -> MLResult                          (frozen — immutable from here on)
        -> raw_ml_risk_score                 (unchanged ML risk formula)
        -> calibration                       (app/calibration.py — stored
                                              versioned artifact, strictly
                                              post-processing)
        -> calibrated_risk_score
        -> Trust Gate                        (app/trust_gate.py — metadata
                                              only, exactly once per case)
        -> TrustFlags
        -> Analytical Engine                 (engine.run_claim_engine /
                                              run_provider_engine) — receives
                                              the upstream context read-only
        -> signals -> evidence
        -> deterministic routing             (app/routing.py — rule-based,
                                              exactly once per case, AFTER the
                                              Analytical Engine; never reruns
                                              any upstream stage, never the LLM)
        -> routing decision
        -> response                          (app/main.py maps these to JSON)
        -> dashboard

The Analytical Engine is called DIRECTLY after the Trust Gate. It receives the
original ML prediction, both risk scores and the TrustFlags and preserves them
unchanged. There is no orchestrator, no agent loop, no tool re-calling and no
evidence-sufficiency decision between the ML stage and the dashboard. The
deterministic routing layer runs only AFTER the Analytical Engine has produced
its signals and evidence; it adds a routing decision and changes nothing else.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytical_engine.engine import run_claim_engine, run_provider_engine
from analytical_engine.inference import HybridModel, predict_claims, predict_providers
from analytical_engine.state import AnalysisOutcome, MLResult

from . import config
from . import routing as rt
from trust_engine import pipeline as trust_pipeline

DEFAULT_STATUS = "Pending Review"

# Lazily loaded, cached per process: one shared routing configuration.
_ROUTING_CONFIG: dict[str, Any] | None = None


def _routing_config() -> dict[str, Any]:
    global _ROUTING_CONFIG
    if _ROUTING_CONFIG is None:
        _ROUTING_CONFIG = rt.load_config()
    return _ROUTING_CONFIG


def _legit_index(model: HybridModel) -> int:
    """Column index of the 'Legitimate' class in the probability matrix."""
    mapping = model.class_mapping
    return next(
        (i for i, cid in enumerate(model.class_ids) if str(mapping[cid]).lower() == "legitimate"),
        0,
    )


# ------------------------------------------------------------------ helpers
def _clean_str(value: Any) -> str | None:
    """NaN-safe string coercion for identifier columns."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value)
    return None if text.lower() == "nan" else text


def _clean_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _merged_rows(records: list[dict[str, Any]], prepared) -> list[dict[str, Any]]:
    """Submitted record + prepared (imputed/encoded) model features.

    Identifiers and non-feature attributes stay exactly as submitted; the
    prepared feature values (training-median imputation, ordinal encoding)
    override the raw ones so signals and ML see identical inputs.
    """
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        row = dict(record)
        row.update(prepared.iloc[i].to_dict())
        rows.append(row)
    return rows


def _imputed_numerics(
    record: dict[str, Any],
    numerical_features: list[str],
) -> list[str]:
    """Infer which numeric inputs required imputation before preprocessing."""
    frame = pd.DataFrame([record])
    imputed: list[str] = []
    for feature in numerical_features:
        raw = frame[feature] if feature in frame.columns else pd.Series(np.nan, index=frame.index)
        coerced = pd.to_numeric(raw, errors="coerce")
        if bool(coerced.isna().iloc[0]):
            imputed.append(feature)
    return imputed


# ------------------------------------------------------ upstream stage
# ML -> raw risk -> calibration -> calibrated risk -> Trust Gate, executed as
# ONE vectorized stage per batch. Returns per-row upstream contexts with the
# exact Phase 2 Trust Gate structure; neither score is ever modified again
# downstream.


def _claim_stage(model: HybridModel, records: list[dict[str, Any]]):
    prepared, results, xgb_prob, lgb_prob = predict_claims(model, records, return_blends=True)
    blend = model.xgboost_ratio * xgb_prob + model.lightgbm_ratio * lgb_prob
    legit = _legit_index(model)
    raw_scores = (1.0 - blend[:, legit]) * 100.0  # raw claim risk, unchanged formula

    contexts: list[dict[str, Any]] = []
    for i in range(len(records)):
        trust_output = trust_pipeline.evaluate(
            {
                "entity_type": "claims",
                "prediction": results[i].predicted_class,
                "raw_risk_score": float(raw_scores[i]),
                "raw_xgb_probabilities": xgb_prob[i].tolist(),
                "raw_lgb_probabilities": lgb_prob[i].tolist(),
                "blended_probabilities": blend[i].tolist(),
            },
            {
                **prepared.iloc[i].to_dict(),
                "imputed_features": _imputed_numerics(records[i], model.numerical_features),
            },
        )
        contexts.append({
            "raw_ml_risk_score": round(float(raw_scores[i]), 6),
            "calibrated_risk_score": round(float(trust_output["calibrated_score"]), 6),
            "prediction": results[i].predicted_class,
            "trust_flags": trust_output["trust_flags"],
            "claim_submitted_amount": _clean_float(prepared.iloc[i].to_dict().get("Claim_Submitted_Amount")),
        })
    return prepared, results, contexts


def _provider_stage(model: HybridModel, records: list[dict[str, Any]]):
    prepared, results, xgb_prob, lgb_prob = predict_providers(model, records, return_blends=True)
    blend = model.xgboost_ratio * xgb_prob + model.lightgbm_ratio * lgb_prob
    legit = _legit_index(model)
    raw_scores = blend[:, 1 - legit] * 100.0  # raw provider risk, unchanged formula

    contexts: list[dict[str, Any]] = []
    for i in range(len(records)):
        trust_output = trust_pipeline.evaluate(
            {
                "entity_type": "providers",
                "prediction": results[i].predicted_class,
                "raw_risk_score": float(raw_scores[i]),
                "raw_xgb_probabilities": xgb_prob[i].tolist(),
                "raw_lgb_probabilities": lgb_prob[i].tolist(),
                "blended_probabilities": blend[i].tolist(),
            },
            {
                **prepared.iloc[i].to_dict(),
                "imputed_features": _imputed_numerics(records[i], model.numerical_features),
            },
        )
        contexts.append({
            "raw_ml_risk_score": round(float(raw_scores[i]), 6),
            "calibrated_risk_score": round(float(trust_output["calibrated_score"]), 6),
            "prediction": results[i].predicted_class,
            "trust_flags": trust_output["trust_flags"],
        })
    return prepared, results, contexts


# -------------------------------------------------------------- single flows
def analyze_claim(model: HybridModel, record: dict[str, Any]) -> AnalysisOutcome:
    """Single claim: ML -> calibration -> Trust Gate -> Claim Analytical
    Engine -> deterministic routing.

    No queue for a single claim — the outcome is the one-case dashboard payload.
    """
    prepared, results, contexts = _claim_stage(model, [record])
    row = _merged_rows([record], prepared)[0]
    outcome = run_claim_engine(model, results[0], row, upstream_context=contexts[0])
    _attach_routing(outcome, rt.route_claim)
    return outcome


def analyze_provider(model: HybridModel, record: dict[str, Any]) -> AnalysisOutcome:
    """Single provider: ML -> calibration -> Trust Gate -> Provider Analytical
    Engine -> deterministic routing.

    No queue for a single provider — the outcome is the one-case dashboard payload.
    """
    prepared, results, contexts = _provider_stage(model, [record])
    row = _merged_rows([record], prepared)[0]
    outcome = run_provider_engine(model, results[0], row, upstream_context=contexts[0])
    _attach_routing(outcome, rt.route_provider)
    return outcome


def _attach_routing(outcome: AnalysisOutcome, router) -> None:
    """Run the deterministic router EXACTLY once on the already-generated
    result and store the decision under upstream_context['routing'].

    The router only READS the upstream context and the Analytical Engine
    signals — it never reruns ML, calibration, the Trust Gate, the Analytical
    Engine or the LLM, and it never mutates any upstream value.
    """
    outcome.upstream_context["routing"] = router(
        outcome.upstream_context, outcome.signals, config=_routing_config()
    )


# -------------------------------------------------------------- batch flows
def analyze_claims_batch(model: HybridModel, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch claims: ONE vectorized ML pass -> calibration + Trust Gate (once
    per case) -> engine per row -> queue + summary."""
    if not records:
        return {
            "total": 0,
            "outcomes": [],
            "identifiers": {"claim_ids": [], "provider_ids": [], "beneficiary_ids": []},
            "queue": [],
            "summary": _claim_summary([], records),
        }

    # One transform + one XGBoost + one LightGBM call for the whole batch,
    # followed by ONE vectorized calibration + Trust Gate pass.
    prepared, results, contexts = _claim_stage(model, records)
    rows = _merged_rows(records, prepared)

    # Analytical Engine runs per row AFTER the frozen ML results and the
    # read-only upstream context exist.
    outcomes = [
        run_claim_engine(model, ml, row, upstream_context=ctx)
        for ml, row, ctx in zip(results, rows, contexts)
    ]
    # Deterministic routing per row, AFTER each Analytical Engine outcome
    # exists — one route per case, never one route for the whole batch.
    for outcome in outcomes:
        _attach_routing(outcome, rt.route_claim)

    claim_ids = [_clean_str(r.get("Claim_ID")) for r in records]
    provider_ids = [_clean_str(r.get("Provider_ID")) for r in records]
    beneficiary_ids = [_clean_str(r.get("BENE_ID")) for r in records]

    return {
        "total": len(records),
        "outcomes": outcomes,
        "identifiers": {
            "claim_ids": claim_ids,
            "provider_ids": provider_ids,
            "beneficiary_ids": beneficiary_ids,
        },
        "queue": _claim_queue(results, claim_ids, provider_ids),
        "summary": _claim_summary(results, records, outcomes),
    }


def analyze_providers_batch(model: HybridModel, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch providers: ONE vectorized ML pass -> calibration + Trust Gate
    (once per case) -> engine per row -> queue + summary."""
    if not records:
        return {
            "total": 0,
            "outcomes": [],
            "identifiers": {"provider_npis": []},
            "queue": [],
            "summary": _provider_summary([], records),
        }

    prepared, results, contexts = _provider_stage(model, records)
    rows = _merged_rows(records, prepared)
    outcomes = [
        run_provider_engine(model, ml, row, upstream_context=ctx)
        for ml, row, ctx in zip(results, rows, contexts)
    ]
    # Deterministic routing per row, AFTER each Analytical Engine outcome
    # exists — one route per case, never one route for the whole batch.
    for outcome in outcomes:
        _attach_routing(outcome, rt.route_provider)

    npis = [_clean_str(r.get("provider_npi")) for r in records]

    return {
        "total": len(records),
        "outcomes": outcomes,
        "identifiers": {"provider_npis": npis},
        "queue": _provider_queue(results, npis, records),
        "summary": _provider_summary(results, records, outcomes),
    }


# ------------------------------------------------------------------- queues
# Queue entries carry the ORIGINAL ML risk score (never a second queue score).
# ``sorted`` is stable, so equal risk scores keep their submission order —
# deterministic tie handling.


def _claim_queue(
    results: list[MLResult], claim_ids: list[str | None], provider_ids: list[str | None]
) -> list[dict[str, Any]]:
    items = [
        {
            "claim_id": claim_ids[i],
            "provider_id": provider_ids[i],
            "risk_score": results[i].risk_score,
            "rating": results[i].rating,
            "predicted_class": results[i].predicted_class,
            "status": DEFAULT_STATUS,
        }
        for i in range(len(results))
    ]
    items = sorted(items, key=lambda r: r["risk_score"], reverse=True)
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
    return items


def _provider_queue(
    results: list[MLResult], npis: list[str | None], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    items = [
        {
            "provider_npi": npis[i],
            "provider_type": _clean_str(records[i].get("provider_type")),
            "risk_score": results[i].risk_score,
            "rating": results[i].rating,
            "predicted_class": results[i].predicted_class,
            "status": DEFAULT_STATUS,
        }
        for i in range(len(results))
    ]
    items = sorted(items, key=lambda r: r["risk_score"], reverse=True)
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
    return items


# ----------------------------------------------------------------- summaries
# Only values computable from the submitted records — nothing is invented.


def _rating_counts(results: list[MLResult]) -> dict[str, int]:
    return {
        "high_risk_count": sum(1 for r in results if r.rating == "High"),
        "medium_risk_count": sum(1 for r in results if r.rating == "Medium"),
        "low_risk_count": sum(1 for r in results if r.rating == "Low"),
    }


def _average_risk(results: list[MLResult]) -> float:
    if not results:
        return 0.0
    return round(sum(r.risk_score for r in results) / len(results), 2)


def _routing_counts(outcomes: list[AnalysisOutcome]) -> dict[str, int]:
    """Per-route case counts for the batch summary (additive; the existing
    summary fields and the queue are untouched by routing)."""
    counts = {route: 0 for route in rt.ROUTE_PRECEDENCE}
    for outcome in outcomes:
        route = (outcome.upstream_context.get("routing") or {}).get("route")
        if route in counts:
            counts[route] += 1
    return counts


def _claim_summary(
    results: list[MLResult], records: list[dict[str, Any]], outcomes: list[AnalysisOutcome] | None = None
) -> dict[str, Any]:
    claim_ids = {_clean_str(r.get("Claim_ID")) for r in records} - {None}
    provider_ids = {_clean_str(r.get("Provider_ID")) for r in records} - {None}
    beneficiary_ids = {_clean_str(r.get("BENE_ID")) for r in records} - {None}
    return {
        "total_records": len(records),
        "suspicious_count": sum(1 for r in results if r.suspicious),
        "average_risk": _average_risk(results),
        **_rating_counts(results),
        "unique_claims": len(claim_ids),
        "unique_providers": len(provider_ids),
        "unique_beneficiaries": len(beneficiary_ids),
        "routing_counts": _routing_counts(outcomes or []),
    }


def _provider_summary(
    results: list[MLResult], records: list[dict[str, Any]], outcomes: list[AnalysisOutcome] | None = None
) -> dict[str, Any]:
    npis = {_clean_str(r.get("provider_npi")) for r in records} - {None}
    suspicious_npis = {
        _clean_str(records[i].get("provider_npi"))
        for i, r in enumerate(results)
        if r.suspicious
    } - {None}
    # The provider data carries per-provider beneficiary counts
    # (cms_total_beneficiaries); a cross-provider UNIQUE beneficiary count is
    # not available in the submitted data, so only the served total is exposed.
    served = sum(_clean_float(r.get("cms_total_beneficiaries")) or 0.0 for r in records)
    return {
        "total_records": len(records),
        "total_providers": len(npis),
        "suspicious_count": sum(1 for r in results if r.suspicious),
        "suspicious_providers": len(suspicious_npis),
        "average_risk": _average_risk(results),
        **_rating_counts(results),
        "total_beneficiaries_served": int(served),
        "routing_counts": _routing_counts(outcomes or []),
    }
