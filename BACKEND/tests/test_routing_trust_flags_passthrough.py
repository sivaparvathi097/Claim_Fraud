from __future__ import annotations

from app.routing import route_claim, ROUTE_AUTO_APPROVE


def _base_context() -> dict:
    return {
        "raw_ml_risk_score": 12.0,
        "calibrated_risk_score": 12.0,
        "prediction": "Legitimate",
        "claim_submitted_amount": 1000.0,
        "trust_flags": {
            "model_agreement": 0.01,
            "low_agreement": False,
            "confidence_entropy": 0.2,
            "confidence": 0.2,
            "low_confidence": False,
            "in_distribution": True,
            "out_of_distribution": False,
            "feature_completeness": True,
        },
    }


def test_route_claim_auto_approve_when_all_predicates_pass() -> None:
    decision = route_claim(_base_context(), [])
    assert decision["route"] == ROUTE_AUTO_APPROVE


def test_route_claim_denies_auto_approve_when_single_flag_fails() -> None:
    ctx = _base_context()
    ctx["trust_flags"] = {**ctx["trust_flags"], "low_agreement": True}
    decision = route_claim(ctx, [])
    assert decision["route"] != ROUTE_AUTO_APPROVE
