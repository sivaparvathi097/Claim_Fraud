"""Evidence adapter — assembles the structured evidence package.

Input : ML prediction + ML risk score + Analytical Engine signals
Output: evidence package (structured signal items with interpretation,
        supporting attributes and severity)

The adapter only READS the ML output — it never modifies the prediction or
the risk score.
"""
from __future__ import annotations

from .state import AnalysisOutcome, MLResult, SignalRecord

# Evidence items shown per record (highest-value signals first). Zero-valued
# signals carry no evidence and are omitted.
MAX_EVIDENCE_ITEMS = 8
MIN_EVIDENCE_VALUE = 1.0


def build_evidence_package(
    ml: MLResult, signals: list[SignalRecord], max_items: int = MAX_EVIDENCE_ITEMS
) -> list[SignalRecord]:
    """Structured evidence supporting the ML result.

    Each item keeps: signal name, value, interpretation, supporting attributes
    (with their actual values from the submitted record) and severity.
    """
    ranked = sorted(
        (s for s in signals if s.value >= MIN_EVIDENCE_VALUE),
        key=lambda s: s.value,
        reverse=True,
    )
    return ranked[:max_items]


def outcome(
    ml: MLResult,
    prepared_row: dict,
    signals: list[SignalRecord],
    deviations: list[dict],
    upstream_context: dict | None = None,
) -> AnalysisOutcome:
    """Assemble the final immutable analysis outcome for one record.

    ``upstream_context`` (Phase 3) is the READ-ONLY context produced before
    the engine (raw/calibrated risk scores, prediction, TrustFlags). It is
    attached as-is — the engine never recalculates or consumes it.
    """
    return AnalysisOutcome(
        ml=ml,
        prepared_row=prepared_row,
        signals=signals,
        top_feature_deviations=deviations,
        evidence=build_evidence_package(ml, signals),
        upstream_context=dict(upstream_context) if upstream_context else {},
    )
