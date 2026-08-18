"""Trust Gate — deterministic trust metadata, strictly BETWEEN calibration and
the Analytical Engine.

Pipeline position (Phase 2 contract):

    ML -> raw_ml_risk_score -> calibration -> calibrated_risk_score
       -> TRUST GATE -> TrustFlags -> Analytical Engine -> signals + evidence

The Trust Gate ONLY attaches metadata. It never:
  * changes the ML prediction, raw_ml_risk_score or calibrated_risk_score
  * generates another risk score
  * calls the LLM or the Analytical Engine
  * makes routing decisions or approves/rejects cases

TrustFlags per prediction
-------------------------
    model_agreement       |P_xgb(predicted) - P_lgb(predicted)|  (pre-blend)
    low_agreement         agreement > threshold (default 0.10, strict >)
    confidence            claims  : entropy H = -sum(p*log(p)) over the 4
                                    hybrid class probabilities
                          providers: |P(predicted class) - 0.5| midpoint
                                    distance of the hybrid probability
    low_confidence        claims: entropy > threshold (default 0.80)
                          providers: distance < threshold (default 0.20)
    in_distribution       all critical numerical features within the
                          documented training/reference ranges
    out_of_distribution   a critical feature outside its reference range
    feature_completeness  no required feature needed imputation before
                          preprocessing (missing/non-numeric numerics are the
                          imputed values per the existing median policy)

Reference ranges are NOT invented: they were recovered from the project's
training data (claims: Claim_Year 2015-2021 training years; providers: the
reproduced 70% training split, rs=42) and are stored OUTSIDE the model
artifact in ``BACKEND/trust_gate_config.json`` together with the configurable
thresholds. Claim and provider gate logic are fully separate; running one
never loads the other model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

TRUST_GATE_CONFIG_FILE = Path(__file__).resolve().parents[1] / "trust_gate_config.json"

TRUST_FLAG_FIELDS = (
    "model_agreement",
    "low_agreement",
    "confidence",
    "low_confidence",
    "in_distribution",
    "out_of_distribution",
    "feature_completeness",
)


class TrustGateError(ValueError):
    """Invalid Trust Gate input or configuration."""


# ------------------------------------------------------------ configuration
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the Trust Gate configuration (thresholds + reference ranges).

    Kept separate from the model artifacts on purpose: ranges are documented
    training-data facts, thresholds are configurable defaults.
    """
    cfg_path = Path(path) if path is not None else TRUST_GATE_CONFIG_FILE
    if not cfg_path.exists():
        raise TrustGateError(f"Trust Gate config not found: {cfg_path}")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    for section in ("agreement", "confidence", "in_distribution"):
        if section not in config:
            raise TrustGateError(f"Trust Gate config is missing section '{section}'")
    return config


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


# ------------------------------------------------- flag building blocks
def _predicted_positions(proba: np.ndarray) -> np.ndarray:
    """Argmax over the blended probabilities — the SAME prediction rule as ML."""
    return np.argmax(proba, axis=1)


def model_agreement(xgb_prob: np.ndarray, lgb_prob: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """|P_xgb(predicted class) - P_lgb(predicted class)| per row.

    Uses the PRE-BLEND probabilities already produced by inference and the
    predicted class of the blended matrix. No additional model call, no new
    score — the gap is a similarity measure between the two ensemble members.
    """
    xgb_prob = np.asarray(xgb_prob, dtype=np.float64)
    lgb_prob = np.asarray(lgb_prob, dtype=np.float64)
    positions = _predicted_positions(np.asarray(proba, dtype=np.float64))
    rows = np.arange(positions.shape[0])
    return np.abs(xgb_prob[rows, positions] - lgb_prob[rows, positions])


def claim_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of the hybrid 4-class probabilities (natural log).

    Zero probabilities contribute 0 (limit of p*log(p)); entropy is kept as-is
    and is never converted into a risk score.
    """
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-15, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def provider_confidence(proba: np.ndarray) -> np.ndarray:
    """Distance of the predicted-class hybrid probability from the 0.5 midpoint."""
    p = np.asarray(proba, dtype=np.float64)
    positions = _predicted_positions(p)
    rows = np.arange(positions.shape[0])
    return np.abs(p[rows, positions] - 0.5)


def missing_numerics(
    records: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    numerical_features: Sequence[str],
) -> list[list[str]]:
    """Per row: the required numeric features imputed by the EXISTING policy.

    Mirrors ``HybridModel.prepare_frame`` exactly: a feature is imputed when it
    is absent from the frame or its value coerces to NaN via ``to_numeric``.
    Imputation behavior itself is NOT touched here — this only observes it.
    """
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


def in_distribution_flags(
    prepared: pd.DataFrame,
    ranges: dict[str, dict[str, float]],
) -> list[bool]:
    """Deterministic feature-range check on the values the model actually saw.

    A row is in-distribution only when EVERY critical feature is within its
    documented [min, max] reference range. The prepared (post-imputation)
    values are used so the check reflects the existing preprocessing exactly
    (e.g. an absent sparse feature evaluates at its artifact training median).
    """
    flags = np.ones(len(prepared), dtype=bool)
    for feature, bounds in ranges.items():
        if feature not in prepared.columns:
            raise TrustGateError(f"Critical feature '{feature}' not present in prepared frame")
        values = pd.to_numeric(prepared[feature], errors="coerce").to_numpy(dtype=np.float64)
        lo, hi = float(bounds["min"]), float(bounds["max"])
        # NaN cannot occur post-imputation for numeric features; guard anyway.
        flags &= ~np.isnan(values) & (values >= lo) & (values <= hi)
    return [bool(f) for f in flags]


# ------------------------------------------------------------ gate outputs
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
        "low_confidence": bool(low_confidence),
        "in_distribution": bool(in_dist),
        "out_of_distribution": bool(not in_dist),
        "feature_completeness": bool(completeness),
    }


def trust_metadata(
    raw_ml_risk_score: float,
    calibrated_risk_score: float,
    prediction: str,
    trust_flags: dict[str, Any],
) -> dict[str, Any]:
    """Exact Phase 2 output structure. Both scores are passed through UNCHANGED."""
    return {
        "raw_ml_risk_score": float(raw_ml_risk_score),
        "calibrated_risk_score": float(calibrated_risk_score),
        "prediction": str(prediction),
        "trust_flags": dict(trust_flags),
    }


# --------------------------------------------------- claim gate (separate)
def claim_trust_flags(
    proba: np.ndarray,
    xgb_prob: np.ndarray,
    lgb_prob: np.ndarray,
    raw_records: Sequence[dict[str, Any]],
    prepared: pd.DataFrame,
    feature_columns: Sequence[str],
    numerical_features: Sequence[str],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """TrustFlags for claim predictions (4-class hybrid).

    confidence = entropy of the 4 hybrid class probabilities (kept as-is).
    """
    config = config if config is not None else load_config()
    thresholds = TrustThresholds.from_config(config)
    ranges = config["in_distribution"]["claims"]["ranges"]

    n = proba.shape[0]
    if not (len(raw_records) == len(prepared) == xgb_prob.shape[0] == lgb_prob.shape[0] == n):
        raise TrustGateError("Claim Trust Gate inputs have inconsistent lengths")

    agreement = model_agreement(xgb_prob, lgb_prob, proba)
    entropy = claim_entropy(proba)
    in_dist = in_distribution_flags(prepared, ranges)
    imputed = missing_numerics(raw_records, feature_columns, numerical_features)

    flags: list[dict[str, Any]] = []
    for i in range(n):
        flags.append(_trust_flags(
            agreement=agreement[i],
            low_agreement=bool(agreement[i] > thresholds.low_agreement_threshold),
            confidence=entropy[i],
            low_confidence=bool(entropy[i] > thresholds.claim_entropy_threshold),
            in_dist=in_dist[i],
            completeness=len(imputed[i]) == 0,
        ))
    return flags


# ------------------------------------------------ provider gate (separate)
def provider_trust_flags(
    proba: np.ndarray,
    xgb_prob: np.ndarray,
    lgb_prob: np.ndarray,
    raw_records: Sequence[dict[str, Any]],
    prepared: pd.DataFrame,
    feature_columns: Sequence[str],
    numerical_features: Sequence[str],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """TrustFlags for provider predictions (binary hybrid).

    confidence = distance of the predicted-class probability from the 0.5
    uncertain midpoint.
    """
    config = config if config is not None else load_config()
    thresholds = TrustThresholds.from_config(config)
    ranges = config["in_distribution"]["providers"]["ranges"]

    n = proba.shape[0]
    if not (len(raw_records) == len(prepared) == xgb_prob.shape[0] == lgb_prob.shape[0] == n):
        raise TrustGateError("Provider Trust Gate inputs have inconsistent lengths")

    agreement = model_agreement(xgb_prob, lgb_prob, proba)
    distance = provider_confidence(proba)
    in_dist = in_distribution_flags(prepared, ranges)
    imputed = missing_numerics(raw_records, feature_columns, numerical_features)

    flags: list[dict[str, Any]] = []
    for i in range(n):
        flags.append(_trust_flags(
            agreement=agreement[i],
            low_agreement=bool(agreement[i] > thresholds.low_agreement_threshold),
            confidence=distance[i],
            low_confidence=bool(distance[i] < thresholds.provider_midpoint_distance_threshold),
            in_dist=in_dist[i],
            completeness=len(imputed[i]) == 0,
        ))
    return flags
