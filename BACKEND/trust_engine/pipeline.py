"""Trust engine orchestration pipeline.

Public contract:
    evaluate(raw_model_output, input_features) -> {
        calibrated_score,
        prediction,
        trust_flags
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .calibration import apply_calibration, load_calibration_artifact
from .trust_gate import compute_trust_flags

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_MODELS_DIR = _BACKEND_DIR / "models"

_CLAIM_CAL = _MODELS_DIR / "claim_calibration_v1.pkl"
_PROVIDER_CAL = _MODELS_DIR / "provider_calibration_v1.pkl"

_CAL_CACHE: dict[str, Any] = {}


def _calibrator_for(entity_type: str):
    if entity_type not in _CAL_CACHE:
        if entity_type == "claims":
            _CAL_CACHE[entity_type] = load_calibration_artifact(_CLAIM_CAL)
        elif entity_type == "providers":
            _CAL_CACHE[entity_type] = load_calibration_artifact(_PROVIDER_CAL)
        else:
            raise ValueError(f"Unknown entity_type '{entity_type}'")
    return _CAL_CACHE[entity_type]


def evaluate(raw_model_output: dict[str, Any], input_features: dict[str, Any]) -> dict[str, Any]:
    """Evaluate calibration and trust flags as one combined trust-engine output."""
    entity_type = str(raw_model_output["entity_type"])
    raw_risk_score = float(raw_model_output["raw_risk_score"])
    prediction = str(raw_model_output["prediction"])

    raw_probability = np.array([max(0.0, min(1.0, raw_risk_score / 100.0))], dtype=np.float64)
    calibrated_probability = apply_calibration(_calibrator_for(entity_type), raw_probability)
    calibrated_score = float(calibrated_probability[0] * 100.0)

    trust_flags = compute_trust_flags(entity_type, raw_model_output, input_features)

    return {
        "calibrated_score": calibrated_score,
        "prediction": prediction,
        "trust_flags": trust_flags,
    }
