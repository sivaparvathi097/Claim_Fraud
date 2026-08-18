"""Analytical Engine — the final analytical evidence layer before the dashboard.

Locked architecture (no orchestrator, no agent loop, no second evidence stage):

    ML result + submitted data
        -> deterministic signal calculators
        -> structured evidence
        -> final analysis result

The engine RECEIVES the original ML prediction and risk score and PRESERVES
them unchanged. It only ADDS signals, evidence and feature deviations next to
the ML output. It never modifies the ML prediction or risk score, never
produces a second fraud score, and never decides whether evidence is
"sufficient" — there is no evidence-sufficiency concept anywhere.
"""
from __future__ import annotations

from typing import Any

from . import evidence_adapter
from .calculators.claim_calculators import calculate_claim_signals
from .calculators.deviations import top_feature_deviations
from .calculators.provider_calculators import calculate_provider_signals
from .inference import HybridModel
from .state import AnalysisOutcome, MLResult


def run_claim_engine(
    model: HybridModel,
    ml: MLResult,
    prepared_row: dict[str, Any],
    upstream_context: dict[str, Any] | None = None,
) -> AnalysisOutcome:
    """Claim Analytical Engine entry point.

    Input : frozen ML result + submitted (prepared) claim data (+ optional
            READ-ONLY upstream context: raw_ml_risk_score, calibrated_risk_score,
            prediction, TrustFlags — attached, never recalculated).
    Output: final claim analysis result = unchanged ML output + claim signals
            + claim evidence + top feature deviations.
    """
    medians = model.training_medians
    signals = calculate_claim_signals(prepared_row, medians)
    deviations = top_feature_deviations(prepared_row, medians)
    return evidence_adapter.outcome(ml, prepared_row, signals, deviations, upstream_context)


def run_provider_engine(
    model: HybridModel,
    ml: MLResult,
    prepared_row: dict[str, Any],
    upstream_context: dict[str, Any] | None = None,
) -> AnalysisOutcome:
    """Provider Analytical Engine entry point.

    Input : frozen ML result + submitted (prepared) provider data (+ optional
            READ-ONLY upstream context: raw_ml_risk_score, calibrated_risk_score,
            prediction, TrustFlags — attached, never recalculated).
    Output: final provider analysis result = unchanged ML output + provider
            signals + provider evidence + top feature deviations.
    """
    medians = model.training_medians
    signals = calculate_provider_signals(prepared_row, medians)
    deviations = top_feature_deviations(prepared_row, medians)
    return evidence_adapter.outcome(ml, prepared_row, signals, deviations, upstream_context)
