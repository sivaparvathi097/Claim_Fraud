"""Analysis state objects.

``MLResult`` is frozen: once the hybrid model produces a prediction and a risk
score, neither the Analytical Engine nor the evidence adapter can modify them.
Score immutability contract:

    score before Analytical Engine
    = score after Analytical Engine
    = score returned in the API
    = score used by the queue
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MLResult:
    """Immutable output of the trained hybrid model for one record."""

    predicted_class_id: int
    predicted_class: str
    class_probabilities: dict[str, float]
    risk_score: float
    rating: str
    suspicious: bool


@dataclass
class SignalRecord:
    """One deterministic Analytical-Engine signal (NOT a model score)."""

    signal: str
    value: float
    interpretation: str
    supporting_attributes: list[str]
    severity: str
    attribute_values: dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisOutcome:
    """Full pipeline output for one record.

    ``ml`` is frozen; the engine/evidence sections only ADD supporting
    information next to the ML output — they never touch ``ml.risk_score``.

    ``upstream_context`` (Phase 3) carries READ-ONLY values produced by the
    stages BEFORE the engine — raw_ml_risk_score, calibrated_risk_score,
    prediction and TrustFlags. The engine never recalculates them and signal
    calculations never consume them.
    """

    ml: MLResult
    prepared_row: dict[str, Any]
    signals: list[SignalRecord] = field(default_factory=list)
    top_feature_deviations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[SignalRecord] = field(default_factory=list)
    upstream_context: dict[str, Any] = field(default_factory=dict)
