"""Hybrid model inference — exact reproduction of the trained artifacts.

Source of truth
---------------
The two ``.pkl`` artifacts are joblib-compressed dictionaries that fully describe
each trained model. Everything needed for inference (features, preprocessor,
encoders, blend ratios, thresholds, class maps, imputation medians) is read
directly from the artifact. Nothing is hard-coded, recreated or retrained here.

Runtime contracts reproduced from the artifacts
-----------------------------------------------
Claim artifact (``claim_fwa_hybrid_model.pkl``):

    raw Claim_Type
        -> artifact["ordinal_encoder"]          (OrdinalEncoder, stored in artifact)
        -> artifact["preprocessor"]             (OneHotEncoder on the encoded ints
                                                 + passthrough for the 30 numerics)
        -> XGBoost.predict_proba                (multi:softprob, 4 classes)
        -> LightGBM.predict_proba               (multiclass, 4 classes)
        -> 0.40 * XGB + 0.60 * LGBM             (probability-level blend)
        -> prediction = argmax(blended probability)
        -> risk score = (1 - P(Legitimate)) * 100          [ML stage, below]

Provider artifact (``provider_fwa_hybrid_model.pkl``):

    raw provider_state / provider_type strings
        -> artifact["preprocessor"]             (OneHotEncoder on the raw strings
                                                 + passthrough for the 27 numerics)
        -> XGBoost.predict_proba                (binary:logistic)
        -> LightGBM.predict_proba               (binary)
        -> 0.50 * XGB + 0.50 * LGBM             (probability-level blend)
        -> prediction = argmax(blended probability)
        -> risk score = P(Suspicious) * 100                [ML stage, below]

No claim ordinal-encoding logic is applied to provider data (the provider
artifact stores no ordinal encoder).

All batch paths are vectorized: one transform, one XGBoost ``predict_proba``
and one LightGBM ``predict_proba`` per batch, regardless of batch size. The
``xgb_calls`` / ``lgb_calls`` counters make that verifiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .state import MLResult

# Sentinel used for missing categorical values before ordinal encoding. The
# stored OrdinalEncoder was configured with handle_unknown="use_encoded_value"
# (unknown_value=-1), so unknown/missing categories are safely encoded as -1
# and the downstream OneHotEncoder (handle_unknown="ignore") maps them to zeros.
_MISSING_CAT = "__missing__"


@dataclass
class HybridModel:
    """A loaded hybrid artifact plus the metadata needed to run it."""

    artifact: dict[str, Any]
    name: str
    xgb_calls: int = 0
    lgb_calls: int = 0

    # ------------------------------------------------------------- metadata
    @property
    def feature_columns(self) -> list[str]:
        return list(self.artifact["feature_columns"])

    @property
    def numerical_features(self) -> list[str]:
        return list(self.artifact["numerical_features"])

    @property
    def categorical_features(self) -> list[str]:
        return list(self.artifact["categorical_features"])

    @property
    def training_medians(self) -> dict[str, float]:
        return dict(self.artifact["training_medians"])

    @property
    def preprocessor(self):
        return self.artifact["preprocessor"]

    @property
    def ordinal_encoder(self):
        """Stored ordinal encoder, present only in the claim artifact."""
        return self.artifact.get("ordinal_encoder")

    @property
    def xgboost_model(self):
        return self.artifact["xgboost_model"]

    @property
    def lightgbm_model(self):
        return self.artifact["lightgbm_model"]

    @property
    def xgboost_ratio(self) -> float:
        return float(self.artifact["xgboost_ratio"])

    @property
    def lightgbm_ratio(self) -> float:
        return float(self.artifact["lightgbm_ratio"])

    @property
    def class_mapping(self) -> dict[int, str]:
        return {int(k): v for k, v in self.artifact["class_mapping"].items()}

    @property
    def class_ids(self) -> list[int]:
        """Class ids in the column order of ``predict_proba`` output."""
        return sorted(self.class_mapping.keys())

    # ------------------------------------------------------------- prep
    def prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Align a raw frame to the model's feature schema (vectorized).

        - Selects exactly ``feature_columns`` (missing columns become NaN).
        - Imputes numeric NaNs with the training medians stored in the artifact.
        - If the artifact stores an ordinal encoder (claim model), applies it to
          the categorical columns BEFORE the stored preprocessor, exactly as done
          at training time. Provider data never goes through this step because
          the provider artifact stores no ordinal encoder.
        """
        out = pd.DataFrame(index=frame.index)
        for col in self.feature_columns:
            out[col] = frame[col] if col in frame.columns else np.nan

        # Numeric features: coerce + impute with stored training medians.
        medians = self.training_medians
        for col in self.numerical_features:
            if col in medians:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(medians[col])

        # Categorical features: keep as object dtype.
        for col in self.categorical_features:
            out[col] = out[col].astype(object)

        # Claim contract: stored ordinal encoder runs before the preprocessor.
        encoder = self.ordinal_encoder
        if encoder is not None and self.categorical_features:
            cat_frame = out[self.categorical_features].fillna(_MISSING_CAT).astype(str)
            encoded = encoder.transform(cat_frame)
            for i, col in enumerate(self.categorical_features):
                out[col] = encoded[:, i]
        return out

    # ------------------------------------------------------------- inference
    def transform(self, prepared: pd.DataFrame):
        """Apply the stored preprocessor to an already-prepared frame."""
        return self.preprocessor.transform(prepared)

    def predict_proba(self, prepared: pd.DataFrame, return_blends: bool = False):
        """Blended class-probability matrix, shape (n_rows, n_classes).

        Exactly one XGBoost and one LightGBM call per invocation, regardless of
        the number of rows (vectorized batch inference contract).

        With ``return_blends=True`` the PRE-BLEND XGBoost and LightGBM
        probability matrices are returned as well — read-only metadata for the
        Trust Gate (model agreement). The blend itself and the call-counting
        contract are unchanged; the default call signature is backward
        compatible and returns only the blended matrix.
        """
        transformed = self.transform(prepared)
        self.xgb_calls += 1
        xgb_prob = self.xgboost_model.predict_proba(transformed)
        self.lgb_calls += 1
        lgb_prob = self.lightgbm_model.predict_proba(transformed)
        blended = self.xgboost_ratio * xgb_prob + self.lightgbm_ratio * lgb_prob
        if return_blends:
            return blended, xgb_prob, lgb_prob
        return blended

    def reset_call_counters(self) -> None:
        self.xgb_calls = 0
        self.lgb_calls = 0


class ModelRegistry:
    """Loads the trained artifacts once and exposes them for inference.

    Claim and provider models are fully isolated: each analysis path touches
    only its own registry property, so claim analysis never loads or executes
    the provider model and vice versa.
    """

    def __init__(self, claim_path, provider_path):
        self._claim_path = claim_path
        self._provider_path = provider_path
        self._claim: HybridModel | None = None
        self._provider: HybridModel | None = None

    def _load(self, path) -> HybridModel:
        artifact = joblib.load(path)
        return HybridModel(artifact=artifact, name=path.stem)

    @property
    def claim(self) -> HybridModel:
        if self._claim is None:
            self._claim = self._load(self._claim_path)
        return self._claim

    @property
    def provider(self) -> HybridModel:
        if self._provider is None:
            self._provider = self._load(self._provider_path)
        return self._provider

    @property
    def claim_loaded(self) -> bool:
        return self._claim is not None

    @property
    def provider_loaded(self) -> bool:
        return self._provider is not None

    def metadata(self) -> dict[str, Any]:
        """Human-readable model metadata (for GET /api/models)."""

        def describe(m: HybridModel) -> dict[str, Any]:
            a = m.artifact
            return {
                "model_type": a.get("model_type"),
                "formula": a.get("formula"),
                "target_column": a.get("target_column"),
                "class_mapping": {str(k): v for k, v in m.class_mapping.items()},
                "n_features": len(m.feature_columns),
                "categorical_features": m.categorical_features,
                "numerical_features": m.numerical_features,
                "xgboost_ratio": m.xgboost_ratio,
                "lightgbm_ratio": m.lightgbm_ratio,
                "prediction_threshold": a.get("prediction_threshold"),
                "prediction_rule": a.get("prediction_rule"),
                "risk_score_definition": a.get("formula"),
            }

        return {"claim": describe(self.claim), "provider": describe(self.provider)}


# ================================================================ ML stage
# Prediction + risk score production. This is the ML stage of the locked
# pipeline (Request -> validation -> ML inference -> MLResult); the Analytical
# Engine runs afterwards and can only ADD signals/evidence next to these
# frozen results.
#
# Risk score definitions (Phase 1 contract, unchanged):
#     claim    : risk_score = (1 - P(Legitimate)) * 100
#     provider : risk_score = P(Suspicious) * 100


def rating_from_score(score: float) -> str:
    """Centralized risk-rating bands (High >= 75, Medium >= 45, else Low).

    Not stored in the artifacts; kept in exactly one place and mirrored by the
    frontend.
    """
    if score >= 75:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def _legit_index(model: HybridModel) -> int:
    """Column index of the 'Legitimate' class in the blended probability matrix."""
    mapping = model.class_mapping
    return next(
        (i for i, cid in enumerate(model.class_ids) if str(mapping[cid]).lower() == "legitimate"),
        0,
    )


def _ml_results(model: HybridModel, proba: np.ndarray, risk_scores: np.ndarray) -> list[MLResult]:
    """Frozen per-row ML results, created BEFORE the Analytical Engine runs."""
    class_ids = model.class_ids
    mapping = model.class_mapping
    pred_positions = np.argmax(proba, axis=1)
    results: list[MLResult] = []
    for i in range(proba.shape[0]):
        pred_id = class_ids[int(pred_positions[i])]
        results.append(MLResult(
            predicted_class_id=pred_id,
            predicted_class=str(mapping[pred_id]),
            class_probabilities={
                str(mapping[cid]): round(float(proba[i, j]), 6) for j, cid in enumerate(class_ids)
            },
            risk_score=float(round(float(risk_scores[i]), 1)),
            rating=rating_from_score(float(risk_scores[i])),
            suspicious=str(mapping[pred_id]).lower() != "legitimate",
        ))
    return results


def predict_claims(
    model: HybridModel,
    records: list[dict[str, Any]],
    return_blends: bool = False,
) -> tuple[pd.DataFrame, list[MLResult]] | tuple[pd.DataFrame, list[MLResult], np.ndarray, np.ndarray]:
    """Vectorized claim ML inference.

    One transform + one XGBoost predict_proba + one LightGBM predict_proba for
    the whole batch, then the stored 0.40/0.60 blend, argmax prediction and
    risk score = (1 - P(Legitimate)) * 100.

    Returns (prepared_frame, frozen MLResult per row). With
    ``return_blends=True`` the pre-blend (xgb_prob, lgb_prob) matrices are
    appended read-only for the Trust Gate; the ML results are identical.
    """
    frame = pd.DataFrame(records)
    prepared = model.prepare_frame(frame)
    if return_blends:
        proba, xgb_prob, lgb_prob = model.predict_proba(prepared, return_blends=True)
    else:
        proba = model.predict_proba(prepared)
    legit = _legit_index(model)
    risk_scores = (1.0 - proba[:, legit]) * 100.0
    results = _ml_results(model, proba, risk_scores)
    if return_blends:
        return prepared, results, xgb_prob, lgb_prob
    return prepared, results


def predict_providers(
    model: HybridModel,
    records: list[dict[str, Any]],
    return_blends: bool = False,
) -> tuple[pd.DataFrame, list[MLResult]] | tuple[pd.DataFrame, list[MLResult], np.ndarray, np.ndarray]:
    """Vectorized provider ML inference.

    One transform + one XGBoost predict_proba + one LightGBM predict_proba for
    the whole batch, then the stored 0.50/0.50 blend, argmax prediction and
    risk score = P(Suspicious) * 100.

    Returns (prepared_frame, frozen MLResult per row). With
    ``return_blends=True`` the pre-blend (xgb_prob, lgb_prob) matrices are
    appended read-only for the Trust Gate; the ML results are identical.
    """
    frame = pd.DataFrame(records)
    prepared = model.prepare_frame(frame)
    if return_blends:
        proba, xgb_prob, lgb_prob = model.predict_proba(prepared, return_blends=True)
    else:
        proba = model.predict_proba(prepared)
    legit = _legit_index(model)
    risk_scores = proba[:, 1 - legit] * 100.0
    results = _ml_results(model, proba, risk_scores)
    if return_blends:
        return prepared, results, xgb_prob, lgb_prob
    return prepared, results


def decode_claim_type(model: HybridModel, value: Any) -> str | None:
    """Map an ordinal-encoded Claim_Type back to its original category string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    encoder = model.ordinal_encoder
    if encoder is None:
        return str(value)
    try:
        categories = encoder.categories_[0]
        idx = int(float(value))
        if 0 <= idx < len(categories):
            return str(categories[idx])
    except (TypeError, ValueError):
        pass
    return str(value)
