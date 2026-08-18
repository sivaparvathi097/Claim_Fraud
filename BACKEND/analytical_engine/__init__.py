"""Analytical Engine for the FWA Risk Investigator.

Locked architecture (no orchestrator, no agent loop, no second evidence stage):

    USER INPUT
        -> CLAIM / PROVIDER ML MODEL            (inference.py: predict_claims /
                                                 predict_providers — vectorized)
        -> ML PREDICTION + RISK SCORE           (state.MLResult — frozen)
        -> ANALYTICAL ENGINE                    (engine.py: deterministic
                                                 signal calculators)
        -> SUPPORTING SIGNALS + EVIDENCE        (evidence_adapter.py)
        -> DASHBOARD

The Analytical Engine is NOT a model. It never replaces, rescales or modifies
the ML prediction or the ML risk score; it only adds supporting signals and
evidence next to the ML output. There is no orchestrator between the ML stage
and the dashboard.
"""
from .engine import run_claim_engine, run_provider_engine
from .inference import (
    HybridModel,
    ModelRegistry,
    decode_claim_type,
    predict_claims,
    predict_providers,
    rating_from_score,
)
from .state import AnalysisOutcome, MLResult, SignalRecord

__all__ = [
    "HybridModel",
    "ModelRegistry",
    "MLResult",
    "AnalysisOutcome",
    "SignalRecord",
    "run_claim_engine",
    "run_provider_engine",
    "predict_claims",
    "predict_providers",
    "decode_claim_type",
    "rating_from_score",
]
