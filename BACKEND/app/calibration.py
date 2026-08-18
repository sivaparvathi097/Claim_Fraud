"""Probability calibration — strictly POST-PROCESSING on top of the locked ML stage.

    raw hybrid probabilities  ->  calibrator  ->  calibrated probabilities
    raw_ml_risk_score         ->  NEVER MODIFIED
    calibrated_risk_score     ->  calibrated_probability * 100   (bounded [0, 100])

Contracts
---------
* The trained XGBoost/LightGBM artifacts, preprocessing, hybrid weights,
  prediction logic and the raw ML risk score are NOT touched. Calibration is
  an independent post-processing layer with its OWN versioned artifacts:

      models/claim_calibration_v1.pkl
      models/provider_calibration_v1.pkl

* Claim and provider calibration are completely independent: separate
  artifacts, separate held-out calibration splits, separate metadata.

* Claims calibrate the binary non-legitimate probability
      P(Fraud or Waste or Abuse) = 1 - P(Legitimate)
  (the same quantity the raw claim risk score is built from).
  Providers calibrate P(Suspicious).

* Methods are configuration-driven: "isotonic" (isotonic regression) or
  "sigmoid" (Platt scaling via logistic regression on the raw probability).

* Calibrated outputs are always bounded to [0, 1]. Out-of-range inputs are
  clipped into [0, 1] before calibration; NaN inputs are rejected with a
  clear error instead of silently producing a value.

This module must not be imported by the Analytical Engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

SUPPORTED_METHODS = ("isotonic", "sigmoid")

CLAIM_CALIBRATION_FILE = "claim_calibration_v1.pkl"
PROVIDER_CALIBRATION_FILE = "provider_calibration_v1.pkl"


class CalibrationError(ValueError):
    """Invalid calibration input or artifact."""


# ----------------------------------------------------------------- fitting
def build_calibrator(method: str):
    """Create an UNFITTED calibrator for the configured method.

    isotonic : IsotonicRegression with clipping (bounded, monotone)
    sigmoid  : Platt scaling — logistic regression on the raw probability
    """
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    if method == "sigmoid":
        # Effectively unregularized logistic regression on the single
        # probability feature (classical Platt scaling).
        return LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    raise CalibrationError(
        f"Unsupported calibration method '{method}'. "
        f"Supported methods: {', '.join(SUPPORTED_METHODS)}"
    )


def fit_calibrator(method: str, raw_probabilities: np.ndarray, y_binary: np.ndarray):
    """Fit a calibrator on a held-out (raw probability, binary label) pair."""
    p = _validate_probabilities(np.asarray(raw_probabilities, dtype=np.float64))
    y = np.asarray(y_binary)
    if p.shape[0] != y.shape[0]:
        raise CalibrationError("raw_probabilities and labels have different lengths")
    if len(np.unique(y)) < 2:
        raise CalibrationError("Calibration fit requires both classes in the held-out split")
    calibrator = build_calibrator(method)
    X = p.reshape(-1, 1) if method == "sigmoid" else p
    calibrator.fit(X, y)
    return calibrator


# ------------------------------------------------------------- transforms
def _validate_probabilities(p: np.ndarray) -> np.ndarray:
    if p.ndim != 1:
        raise CalibrationError("Calibration expects a 1-D array of probabilities")
    if np.isnan(p).any():
        raise CalibrationError("Calibration input contains NaN probabilities")
    return p


def _apply(calibrator, method: str, p: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator; inputs clipped to [0,1], outputs bounded."""
    p = np.clip(p, 0.0, 1.0)
    X = p.reshape(-1, 1) if method == "sigmoid" else p
    out = calibrator.predict(X)
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


@dataclass(frozen=True)
class CalibratedScore:
    """Raw score preserved + calibrated companion. The raw value is NEVER replaced."""

    raw_ml_risk_score: float
    calibrated_probability: float
    calibrated_risk_score: float


class ProbabilityCalibrator:
    """A fitted calibrator plus the versioned metadata of its artifact."""

    def __init__(self, bundle: dict[str, Any]):
        for key in ("calibrator", "method", "model_name", "calibration_version"):
            if key not in bundle:
                raise CalibrationError(f"Calibration artifact is missing '{key}'")
        if bundle["method"] not in SUPPORTED_METHODS:
            raise CalibrationError(f"Unsupported method in artifact: {bundle['method']}")
        self.bundle = bundle
        self.method: str = bundle["method"]
        self.model_name: str = bundle["model_name"]
        self.calibration_version: str = bundle["calibration_version"]

    # ------------------------------------------------------------- metadata
    @property
    def source_split(self) -> dict[str, Any]:
        return dict(self.bundle.get("source_split", {}))

    @property
    def metadata(self) -> dict[str, Any]:
        meta = {k: v for k, v in self.bundle.items() if k != "calibrator"}
        return meta

    # ------------------------------------------------------------- scoring
    def calibrate_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Calibrated probabilities for raw hybrid probabilities, bounded [0,1]."""
        p = _validate_probabilities(np.asarray(raw_probabilities, dtype=np.float64))
        return _apply(self.bundle["calibrator"], self.method, p)

    def calibrate_risk_scores(self, raw_ml_risk_scores: np.ndarray) -> np.ndarray:
        """Calibrated risk scores (0..100). Raw scores are NOT modified here."""
        raw = np.asarray(raw_ml_risk_scores, dtype=np.float64)
        if np.isnan(raw).any():
            raise CalibrationError("Risk score input contains NaN")
        probs = np.clip(raw, 0.0, 100.0) / 100.0
        return self.calibrate_probabilities(probs) * 100.0

    def paired_score(self, raw_ml_risk_score: float) -> CalibratedScore:
        """Keep BOTH values: raw_ml_risk_score and calibrated_risk_score."""
        raw = float(raw_ml_risk_score)
        calibrated = float(self.calibrate_risk_scores(np.array([raw]))[0])
        return CalibratedScore(
            raw_ml_risk_score=raw,
            calibrated_probability=calibrated / 100.0,
            calibrated_risk_score=calibrated,
        )


# --------------------------------------------------------------- artifacts
def load_calibrator(path) -> ProbabilityCalibrator:
    """Load one versioned calibration artifact (claim XOR provider)."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise CalibrationError(f"{path} is not a calibration bundle")
    return ProbabilityCalibrator(bundle)


def save_calibrator(bundle: dict[str, Any], path) -> None:
    joblib.dump(bundle, path, compress=3)
