"""Phase 4 — Deterministic routing (rule-based, NEVER the LLM).

Position in the locked architecture:

    INPUT -> ML -> raw risk -> calibration -> calibrated risk -> Trust Gate
    -> TrustFlags -> Analytical Engine -> signals + evidence
    -> DETERMINISTIC ROUTING -> dashboard

The routing layer consumes ONLY the already-generated result of a case:

    * raw_ml_risk_score        (from the upstream context)
    * calibrated_risk_score    (from the upstream context)
    * prediction               (from the upstream context)
    * trust_flags              (from the upstream context)
    * signals                  (from the Analytical Engine)

It never reruns ML, calibration, the Trust Gate or the Analytical Engine, it
never calls the LLM, and it never modifies any upstream value. It adds ONE
routing decision per case:

    {
      "route": "auto_approve" | "fast_track" | "full_investigation",
      "reasons": [...human-readable, auditable...],
      "requires_human_review": bool,
      "predicate_results": {<predicate name>: bool, ...}
    }

ROUTE PRECEDENCE (explicit, never dictionary order):

    1. full_investigation   — any failing predicate escalates here
    2. fast_track           — moderate calibrated risk or a moderate count of
                              supporting signals, with every predicate passing
    3. auto_approve         — every predicate passes and neither escalation
                              condition applies

Auto-approve only means the case satisfies the configured automated policy.
It is NOT a claim that the case is proven legitimate, clinically or legally.

ARCHITECTURAL RULE FOR THE FUTURE LLM PHASE:

    The routing layer decides the route. The LLM explains the route. The LLM
    must eventually be called for EVERY route — including auto_approve — and
    the LLM never determines the route. That LLM wiring is NOT part of this
    module and is intentionally not implemented here.

Fail-safe: if the upstream context lacks calibrated risk or trust metadata,
the case routes to full_investigation. Unknown trust never de-escalates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROUTE_AUTO_APPROVE = "auto_approve"
ROUTE_FAST_TRACK = "fast_track"
ROUTE_FULL_INVESTIGATION = "full_investigation"

# Explicit, documented precedence — safest first. Route selection below uses
# this ordered tuple directly; it never depends on dict ordering.
ROUTE_PRECEDENCE: tuple[str, ...] = (
    ROUTE_FULL_INVESTIGATION,
    ROUTE_FAST_TRACK,
    ROUTE_AUTO_APPROVE,
)

ROUTES: tuple[str, ...] = ROUTE_PRECEDENCE

# Fixed audit order for predicate evaluation/reporting (independent of dict
# ordering in any Python version).
ROUTE_PRECEDENCE_PREDICATE_ORDER: tuple[str, ...] = (
    "calibration_band_check",
    "risk_check",
    "agreement_check",
    "confidence_check",
    "distribution_check",
    "completeness_check",
    "no_analytical_anomaly_check",
    "provider_clean_check",
    "claim_value_under_threshold_check",
    "not_rare_class_driven_check",
    "prediction_check",
    "signal_check",
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "routing_config.json"
_CONFIG: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Routing thresholds, separate sections for claims and providers."""
    with open(path or _CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    global _CONFIG
    if config is not None:
        return config
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


# ------------------------------------------------------------- predicate parts
def supporting_signal_count(signals: list[Any], severities: tuple[str, ...]) -> int:
    """Count Analytical Engine signals whose severity supports escalation.

    Read-only: signals are inspected, never modified or re-generated.
    """
    return sum(1 for s in signals if getattr(s, "severity", None) in severities)


def _evaluate_predicates(
    kind: str,
    entity_config: dict[str, Any],
    upstream_context: dict[str, Any] | None,
    signals: list[Any],
) -> tuple[dict[str, bool], list[str], dict[str, Any]]:
    """Evaluate every configured predicate individually.

    Returns (predicate_results, escalation_reasons, observed_values).
    Each predicate is a PASS boolean (True = the case satisfies the safe
    condition); any failing predicate triggers full_investigation.
    """
    context = upstream_context or {}
    flags = context.get("trust_flags") or {}
    calibrated = context.get("calibrated_risk_score")
    prediction = context.get("prediction")
    raw_ml_risk = context.get("raw_ml_risk_score")
    claim_submitted_amount = context.get("claim_submitted_amount")
    severities = tuple(entity_config.get("supporting_signal_severities", ["High", "Medium"]))
    support = supporting_signal_count(signals, severities)
    rare_classes = {str(v) for v in entity_config.get("rare_classes", [])}
    claim_value_max = entity_config.get("claim_value_auto_approve_max")

    predicates: dict[str, bool] = {}
    reasons: list[str] = []

    # calibration_band_check — calibrated score must be numeric and bounded.
    predicates["calibration_band_check"] = (
        calibrated is not None and 0.0 <= float(calibrated) <= 100.0
    )
    if not predicates["calibration_band_check"]:
        reasons.append("calibrated risk score is outside the valid [0,100] band")

    # risk_check — calibrated risk below the full-investigation threshold
    full_threshold = float(entity_config["full_investigation_risk_threshold"])
    if calibrated is None:
        predicates["risk_check"] = False
        reasons.append("calibrated risk score is unavailable — fail-safe escalation")
    else:
        predicates["risk_check"] = float(calibrated) < full_threshold
        if not predicates["risk_check"]:
            reasons.append(
                f"calibrated risk {float(calibrated):.2f} reaches the "
                f"full-investigation threshold {full_threshold:.2f}"
            )

    # Trust Gate predicates — unknown trust never de-escalates
    if not flags:
        for name in (
            "agreement_check",
            "confidence_check",
            "distribution_check",
            "completeness_check",
            "no_analytical_anomaly_check",
            "provider_clean_check",
            "claim_value_under_threshold_check",
            "not_rare_class_driven_check",
        ):
            predicates[name] = False
        reasons.append("trust metadata unavailable — fail-safe escalation")
    else:
        predicates["agreement_check"] = not bool(flags.get("low_agreement", True))
        if not predicates["agreement_check"]:
            reasons.append(
                f"model agreement gap {float(flags.get('model_agreement', 0.0)):.4f} "
                "exceeds the Trust Gate tolerance (low_agreement)"
            )
        predicates["confidence_check"] = not bool(flags.get("low_confidence", True))
        if not predicates["confidence_check"]:
            reasons.append(
                f"prediction confidence {float(flags.get('confidence_entropy', flags.get('confidence', 0.0))):.4f} "
                "is below the Trust Gate tolerance (low_confidence)"
            )
        predicates["distribution_check"] = bool(flags.get("in_distribution", False))
        if not predicates["distribution_check"]:
            reasons.append("features are out of the documented training ranges (out_of_distribution)")
        predicates["completeness_check"] = bool(flags.get("feature_completeness", False))
        if not predicates["completeness_check"]:
            reasons.append("required features needed imputation before preprocessing (incomplete features)")

        # no_analytical_anomaly_check — high/medium support must stay below the
        # full-investigation signal count to remain auto-approve eligible.
        full_signals = int(entity_config["min_supporting_signals_full_investigation"])
        predicates["no_analytical_anomaly_check"] = support < full_signals
        if not predicates["no_analytical_anomaly_check"]:
            reasons.append(
                f"{support} supporting Analytical Engine signals indicate an analytical anomaly "
                f"(threshold {full_signals})"
            )

        # provider_clean_check — non-legitimate provider predictions are never
        # auto-approve eligible.
        if kind == "providers":
            legitimate = set(entity_config.get("legitimate_classes", ["Legitimate"]))
            predicates["provider_clean_check"] = prediction in legitimate
            if not predicates["provider_clean_check"]:
                reasons.append(f"provider prediction '{prediction}' is not in legitimate classes")
        else:
            predicates["provider_clean_check"] = True

        # claim_value_under_threshold_check — optional claim auto-approve guard.
        if kind == "claims" and claim_value_max is not None:
            if claim_submitted_amount is None:
                predicates["claim_value_under_threshold_check"] = False
                reasons.append("claim submitted amount unavailable for auto-approve threshold check")
            else:
                predicates["claim_value_under_threshold_check"] = float(claim_submitted_amount) <= float(claim_value_max)
                if not predicates["claim_value_under_threshold_check"]:
                    reasons.append(
                        f"claim submitted amount {float(claim_submitted_amount):.2f} exceeds auto-approve threshold {float(claim_value_max):.2f}"
                    )
        else:
            predicates["claim_value_under_threshold_check"] = True

        # not_rare_class_driven_check — prediction class must not be configured as rare.
        predicates["not_rare_class_driven_check"] = prediction not in rare_classes
        if not predicates["not_rare_class_driven_check"]:
            reasons.append(f"predicted class '{prediction}' is configured as rare-class-driven")

    # prediction_check — case type: a non-legitimate prediction escalates
    legitimate = set(entity_config.get("legitimate_classes", ["Legitimate"]))
    predicates["prediction_check"] = prediction in legitimate
    if not predicates["prediction_check"]:
        reasons.append(f"predicted class '{prediction}' is not a legitimate class")

    # signal_check — Analytical Engine support below the full-investigation count
    full_signals = int(entity_config["min_supporting_signals_full_investigation"])
    predicates["signal_check"] = support < full_signals
    if not predicates["signal_check"]:
        reasons.append(
            f"{support} supporting Analytical Engine signals reach the "
            f"full-investigation count {full_signals}"
        )

    observed = {
        "calibrated_risk_score": calibrated,
        "raw_ml_risk_score": raw_ml_risk,
        "prediction": prediction,
        "supporting_signal_count": support,
    }
    return predicates, reasons, observed


# ------------------------------------------------------------------- decision
def route_case(
    kind: str,
    upstream_context: dict[str, Any] | None,
    signals: list[Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic route for ONE already-analyzed case.

    ``kind`` selects the claim- or provider-specific configuration section —
    the two never share thresholds or artifacts.
    """
    entity_config = _config(config)[kind]
    predicates, reasons, observed = _evaluate_predicates(kind, entity_config, upstream_context, signals)

    calibrated = observed["calibrated_risk_score"]
    support = observed["supporting_signal_count"]
    fast_threshold = float(entity_config["fast_track_risk_threshold"])
    fast_signals = int(entity_config["min_supporting_signals_fast_track"])

    # Precedence step 1: full_investigation — ANY failing predicate.
    failed = [name for name in ROUTE_PRECEDENCE_PREDICATE_ORDER if not predicates[name]]
    if failed:
        return _decision(ROUTE_FULL_INVESTIGATION, reasons, predicates)

    # Precedence step 2: fast_track — every predicate passes, but moderate
    # calibrated risk or a moderate count of supporting signals applies.
    fast_reasons: list[str] = []
    if calibrated is not None and float(calibrated) >= fast_threshold:
        fast_reasons.append(
            f"calibrated risk {float(calibrated):.2f} reaches the fast-track threshold {fast_threshold:.2f}"
        )
    if support >= fast_signals:
        fast_reasons.append(
            f"{support} supporting Analytical Engine signals reach the fast-track count {fast_signals}"
        )
    if fast_reasons:
        return _decision(ROUTE_FAST_TRACK, fast_reasons, predicates)

    # Precedence step 3: auto_approve — every configured predicate passed and
    # no escalation condition applied. This means the case satisfies the
    # configured automated policy; it does NOT prove legitimacy.
    auto_reasons = [
        f"all {len(predicates)} routing predicates passed",
        f"calibrated risk {float(calibrated):.2f} is below the fast-track threshold {fast_threshold:.2f}",
        f"supporting signal count {support} is below the fast-track count {fast_signals}",
    ]
    return _decision(ROUTE_AUTO_APPROVE, auto_reasons, predicates)


def _decision(route: str, reasons: list[str], predicates: dict[str, bool]) -> dict[str, Any]:
    ordered = {name: predicates[name] for name in ROUTE_PRECEDENCE_PREDICATE_ORDER}
    return {
        "route": route,
        "reasons": reasons,
        "requires_human_review": route != ROUTE_AUTO_APPROVE,
        "predicate_results": ordered,
    }


# ---------------------------------------------------------- per-entity entry
def route_claim(
    upstream_context: dict[str, Any] | None,
    signals: list[Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Claim routing — claim configuration and claim signals only."""
    return route_case("claims", upstream_context, signals, config=config)


def route_provider(
    upstream_context: dict[str, Any] | None,
    signals: list[Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provider routing — provider configuration and provider signals only."""
    return route_case("providers", upstream_context, signals, config=config)
