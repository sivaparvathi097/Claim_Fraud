"""Calibration layer for trust_engine.

This module post-processes model raw risk/probability outputs. It does not
load or retrain XGBoost/LightGBM model artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

SUPPORTED_METHODS = ("isotonic", "sigmoid")
CLAIM_CALIBRATION_FILE = "claim_calibration_v1.pkl"
PROVIDER_CALIBRATION_FILE = "provider_calibration_v1.pkl"


class CalibrationError(ValueError):
    """Invalid calibration input or artifact."""


@dataclass(frozen=True)
class CalibratedScore:
    raw_ml_risk_score: float
    calibrated_probability: float
    calibrated_risk_score: float


@dataclass(frozen=True)
class CalibrationModel:
    """Fitted calibrator plus metadata."""

    calibrator: Any
    method: str
    model_name: str
    calibration_version: str
    metadata: dict[str, Any]


class ProbabilityCalibrator:
    """Compatibility wrapper for bundle-style calibration artifacts."""

    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.method = str(bundle["method"])
        self.model_name = str(bundle["model_name"])
        self.calibration_version = str(bundle["calibration_version"])

    @property
    def source_split(self) -> dict[str, Any]:
        return dict(self.bundle.get("source_split", {}))

    def calibrate_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        model = CalibrationModel(
            calibrator=self.bundle["calibrator"],
            method=self.method,
            model_name=self.model_name,
            calibration_version=self.calibration_version,
            metadata={k: v for k, v in self.bundle.items() if k != "calibrator"},
        )
        return apply_calibration(model, raw_probabilities)

    def calibrate_risk_scores(self, raw_ml_risk_scores: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_ml_risk_scores, dtype=np.float64)
        if np.isnan(raw).any():
            raise CalibrationError("Calibration input contains NaN")
        probs = np.clip(raw, 0.0, 100.0) / 100.0
        return self.calibrate_probabilities(probs) * 100.0

    def paired_score(self, raw_ml_risk_score: float) -> CalibratedScore:
        raw = float(raw_ml_risk_score)
        calibrated = float(self.calibrate_risk_scores(np.array([raw]))[0])
        return CalibratedScore(
            raw_ml_risk_score=raw,
            calibrated_probability=calibrated / 100.0,
            calibrated_risk_score=calibrated,
        )


def _validate_probabilities(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1:
        raise CalibrationError("Expected a 1-D probability array")
    if np.isnan(p).any():
        raise CalibrationError("Calibration input contains NaN")
    return np.clip(p, 0.0, 1.0)


def _build_calibrator(method: str):
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    if method == "sigmoid":
        return LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    raise CalibrationError(f"Unsupported calibration method: {method}")


def fit_calibration(
    raw_probabilities: np.ndarray,
    y_binary: np.ndarray,
    method: str = "isotonic",
    *,
    model_name: str = "",
    calibration_version: str = "",
    metadata: dict[str, Any] | None = None,
) -> CalibrationModel:
    """Fit a calibration model from held-out probabilities and binary labels."""
    if method not in SUPPORTED_METHODS:
        raise CalibrationError(f"Unsupported method '{method}'")

    p = _validate_probabilities(raw_probabilities)
    y = np.asarray(y_binary)
    if p.shape[0] != y.shape[0]:
        raise CalibrationError("raw_probabilities and labels length mismatch")
    if len(np.unique(y)) < 2:
        raise CalibrationError("Calibration fit requires both classes")

    calibrator = _build_calibrator(method)
    X = p.reshape(-1, 1) if method == "sigmoid" else p
    calibrator.fit(X, y)

    return CalibrationModel(
        calibrator=calibrator,
        method=method,
        model_name=model_name,
        calibration_version=calibration_version,
        metadata=dict(metadata or {}),
    )


def apply_calibration(model: CalibrationModel, raw_probabilities: np.ndarray) -> np.ndarray:
    """Apply calibration to raw probabilities in [0,1]."""
    p = _validate_probabilities(raw_probabilities)
    X = p.reshape(-1, 1) if model.method == "sigmoid" else p
    out = model.calibrator.predict(X)
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def _apply(calibrator, method: str, raw_probabilities: np.ndarray) -> np.ndarray:
    """Compatibility function used by existing calibration-fit script."""
    p = _validate_probabilities(raw_probabilities)
    X = p.reshape(-1, 1) if method == "sigmoid" else p
    out = calibrator.predict(X)
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def fit_calibrator(method: str, raw_probabilities: np.ndarray, y_binary: np.ndarray):
    """Compatibility fit function returning only the fitted sklearn calibrator."""
    return fit_calibration(raw_probabilities, y_binary, method=method).calibrator


def generate_calibration_report(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> dict[str, float]:
    """Return compact calibration-quality metrics."""
    y = np.asarray(y_true)
    raw = _validate_probabilities(raw_probabilities)
    cal = _validate_probabilities(calibrated_probabilities)
    if not (len(y) == len(raw) == len(cal)):
        raise CalibrationError("Metric inputs length mismatch")
    return {
        "raw_brier": float(brier_score_loss(y, raw)),
        "calibrated_brier": float(brier_score_loss(y, cal)),
        "n_samples": float(len(y)),
    }


def load_calibration_artifact(path) -> CalibrationModel:
    """Load versioned calibration artifact produced previously."""
    bundle = joblib.load(path)
    required = ("calibrator", "method", "model_name", "calibration_version")
    if not isinstance(bundle, dict) or any(k not in bundle for k in required):
        raise CalibrationError(f"Invalid calibration artifact: {path}")
    method = str(bundle["method"])
    if method not in SUPPORTED_METHODS:
        raise CalibrationError(f"Unsupported method in artifact: {method}")

    meta = {k: v for k, v in bundle.items() if k != "calibrator"}
    return CalibrationModel(
        calibrator=bundle["calibrator"],
        method=method,
        model_name=str(bundle["model_name"]),
        calibration_version=str(bundle["calibration_version"]),
        metadata=meta,
    )


def load_calibrator(path) -> ProbabilityCalibrator:
    bundle = joblib.load(path)
    required = ("calibrator", "method", "model_name", "calibration_version")
    if not isinstance(bundle, dict) or any(k not in bundle for k in required):
        raise CalibrationError(f"Invalid calibration artifact: {path}")
    method = str(bundle["method"])
    if method not in SUPPORTED_METHODS:
        raise CalibrationError(f"Unsupported method in artifact: {method}")
    return ProbabilityCalibrator(bundle)


def save_calibrator(bundle: dict[str, Any], path) -> None:
    joblib.dump(bundle, path, compress=3)
