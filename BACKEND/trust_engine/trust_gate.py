"""Trust flag computation for trust_engine.

Computes trust metadata from raw model output and input features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parents[1] / "trust_gate_config.json"
TRUST_FLAG_FIELDS = (
    "model_agreement",
    "low_agreement",
    "confidence",
    "confidence_entropy",
    "low_confidence",
    "in_distribution",
    "out_of_distribution",
    "feature_completeness",
)


class TrustGateError(ValueError):
    """Invalid trust gate input."""


@dataclass(frozen=True)
class TrustThresholds:
    low_agreement_threshold: float = 0.10
    claim_entropy_threshold: float = 0.80
    provider_midpoint_distance_threshold: float = 0.20

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrustThresholds":
        return cls(
            low_agreement_threshold=float(config["agreement"]["low_agreement_threshold"]),
            claim_entropy_threshold=float(config["confidence"]["claim_entropy_threshold"]),
            provider_midpoint_distance_threshold=float(
                config["confidence"]["provider_midpoint_distance_threshold"]
            ),
        )


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg = Path(path) if path is not None else CONFIG_PATH
    return json.loads(cfg.read_text(encoding="utf-8"))


def _predicted_index(blended_probabilities: np.ndarray) -> int:
    return int(np.argmax(blended_probabilities))


def _predicted_positions(proba: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(proba, dtype=np.float64), axis=1)


def _entropy(probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1.0)
    return float(-np.sum(p * np.log(p)))


def model_agreement(xgb_prob: np.ndarray, lgb_prob: np.ndarray, proba: np.ndarray) -> np.ndarray:
    xgb = np.asarray(xgb_prob, dtype=np.float64)
    lgb = np.asarray(lgb_prob, dtype=np.float64)
    positions = _predicted_positions(np.asarray(proba, dtype=np.float64))
    rows = np.arange(positions.shape[0])
    return np.abs(xgb[rows, positions] - lgb[rows, positions])


def claim_entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-15, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def provider_confidence(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    positions = _predicted_positions(p)
    rows = np.arange(positions.shape[0])
    return np.abs(p[rows, positions] - 0.5)


def _provider_midpoint_distance(probabilities: np.ndarray) -> float:
    idx = _predicted_index(probabilities)
    return float(abs(float(probabilities[idx]) - 0.5))


def _in_distribution(entity: str, features: dict[str, Any], config: dict[str, Any]) -> bool:
    ranges = config["in_distribution"][entity]["ranges"]
    for feature, bounds in ranges.items():
        if feature not in features:
            return False
        try:
            value = float(features[feature])
        except (TypeError, ValueError):
            return False
        if np.isnan(value):
            return False
        if value < float(bounds["min"]) or value > float(bounds["max"]):
            return False
    return True


def missing_numerics(
    records,
    feature_columns,
    numerical_features,
) -> list[list[str]]:
    frame = pd.DataFrame(list(records))
    missing: list[list[str]] = [[] for _ in range(len(records))]
    for col in numerical_features:
        if col not in feature_columns:
            continue
        raw = frame[col] if col in frame.columns else pd.Series(np.nan, index=frame.index)
        coerced = pd.to_numeric(raw, errors="coerce")
        for i in np.flatnonzero(coerced.isna().to_numpy()):
            missing[int(i)].append(col)
    return missing


def in_distribution_flags(prepared: pd.DataFrame, ranges: dict[str, dict[str, float]]) -> list[bool]:
    flags = np.ones(len(prepared), dtype=bool)
    for feature, bounds in ranges.items():
        if feature not in prepared.columns:
            raise TrustGateError(f"Critical feature '{feature}' not present in prepared frame")
        values = pd.to_numeric(prepared[feature], errors="coerce").to_numpy(dtype=np.float64)
        lo, hi = float(bounds["min"]), float(bounds["max"])
        flags &= ~np.isnan(values) & (values >= lo) & (values <= hi)
    return [bool(f) for f in flags]


def _trust_flags(
    agreement: float,
    low_agreement: bool,
    confidence: float,
    low_confidence: bool,
    in_dist: bool,
    completeness: bool,
) -> dict[str, Any]:
    return {
        "model_agreement": float(agreement),
        "low_agreement": bool(low_agreement),
        "confidence": float(confidence),
        "confidence_entropy": float(confidence),
        "low_confidence": bool(low_confidence),
        "in_distribution": bool(in_dist),
        "out_of_distribution": bool(not in_dist),
        "feature_completeness": bool(completeness),
    }


def compute_trust_flags(
    entity_type: str,
    raw_model_output: dict[str, Any],
    input_features: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute trust flags for one prediction.

    Expected raw_model_output keys:
      raw_xgb_probabilities, raw_lgb_probabilities, blended_probabilities
    """
    cfg = load_config() if config is None else config
    xgb = np.asarray(raw_model_output["raw_xgb_probabilities"], dtype=np.float64)
    lgb = np.asarray(raw_model_output["raw_lgb_probabilities"], dtype=np.float64)
    blended = np.asarray(raw_model_output["blended_probabilities"], dtype=np.float64)

    if xgb.ndim != 1 or lgb.ndim != 1 or blended.ndim != 1:
        raise TrustGateError("Trust gate expects 1-D probability vectors")

    pred_idx = _predicted_index(blended)
    agreement = float(abs(xgb[pred_idx] - lgb[pred_idx]))
    low_agreement = agreement > float(cfg["agreement"]["low_agreement_threshold"])

    if entity_type == "claims":
        confidence_entropy = _entropy(blended)
        low_confidence = confidence_entropy > float(cfg["confidence"]["claim_entropy_threshold"])
    elif entity_type == "providers":
        confidence_entropy = _provider_midpoint_distance(blended)
        low_confidence = confidence_entropy < float(cfg["confidence"]["provider_midpoint_distance_threshold"])
    else:
        raise TrustGateError(f"Unknown entity_type '{entity_type}'")

    in_distribution = _in_distribution(entity_type, input_features, cfg)
    imputed_features = input_features.get("imputed_features", [])
    feature_completeness = len(imputed_features) == 0

    return {
        "model_agreement": agreement,
        "low_agreement": bool(low_agreement),
        "confidence_entropy": float(confidence_entropy),
        "confidence": float(confidence_entropy),
        "low_confidence": bool(low_confidence),
        "in_distribution": bool(in_distribution),
        "out_of_distribution": bool(not in_distribution),
        "feature_completeness": bool(feature_completeness),
    }


def trust_metadata(
    raw_ml_risk_score: float,
    calibrated_risk_score: float,
    prediction: str,
    trust_flags: dict[str, Any],
) -> dict[str, Any]:
    return {
        "raw_ml_risk_score": float(raw_ml_risk_score),
        "calibrated_risk_score": float(calibrated_risk_score),
        "prediction": str(prediction),
        "trust_flags": dict(trust_flags),
    }


def claim_trust_flags(
    proba: np.ndarray,
    xgb_prob: np.ndarray,
    lgb_prob: np.ndarray,
    raw_records,
    prepared: pd.DataFrame,
    feature_columns,
    numerical_features,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config() if config is None else config
    thresholds = TrustThresholds.from_config(cfg)
    ranges = cfg["in_distribution"]["claims"]["ranges"]
    n = np.asarray(proba).shape[0]
    agreement = model_agreement(xgb_prob, lgb_prob, proba)
    entropy = claim_entropy(proba)
    in_dist = in_distribution_flags(prepared, ranges)
    imputed = missing_numerics(raw_records, feature_columns, numerical_features)

    out: list[dict[str, Any]] = []
    for i in range(n):
        out.append(
            _trust_flags(
                agreement=float(agreement[i]),
                low_agreement=bool(agreement[i] > thresholds.low_agreement_threshold),
                confidence=float(entropy[i]),
                low_confidence=bool(entropy[i] > thresholds.claim_entropy_threshold),
                in_dist=in_dist[i],
                completeness=len(imputed[i]) == 0,
            )
        )
    return out


def provider_trust_flags(
    proba: np.ndarray,
    xgb_prob: np.ndarray,
    lgb_prob: np.ndarray,
    raw_records,
    prepared: pd.DataFrame,
    feature_columns,
    numerical_features,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config() if config is None else config
    thresholds = TrustThresholds.from_config(cfg)
    ranges = cfg["in_distribution"]["providers"]["ranges"]
    n = np.asarray(proba).shape[0]
    agreement = model_agreement(xgb_prob, lgb_prob, proba)
    distance = provider_confidence(proba)
    in_dist = in_distribution_flags(prepared, ranges)
    imputed = missing_numerics(raw_records, feature_columns, numerical_features)

    out: list[dict[str, Any]] = []
    for i in range(n):
        out.append(
            _trust_flags(
                agreement=float(agreement[i]),
                low_agreement=bool(agreement[i] > thresholds.low_agreement_threshold),
                confidence=float(distance[i]),
                low_confidence=bool(distance[i] < thresholds.provider_midpoint_distance_threshold),
                in_dist=in_dist[i],
                completeness=len(imputed[i]) == 0,
            )
        )
    return out
