"""Real uploaded-file handling for the batch endpoints.

    USER UPLOADS CSV
        -> READ ACTUAL UPLOADED FILE
        -> VALIDATE SCHEMA (against the feature contract stored in the .pkl)
        -> ACTUAL UPLOADED RECORDS
        -> vectorised ML inference -> Analytical Engine -> dashboard -> queue

Data-leakage protection: training labels (``target`` / ``Target_Label`` for
claims, ``Potential Fraud`` for providers) are REMOVED before inference — an
uploaded export may carry them, but they never enter production inference and
are never required in an uploaded production CSV.

Nothing here invents missing feature values: missing/invalid numerics become
NaN and go through the artifact's own policy (training-median imputation
inside ``HybridModel.prepare_frame``); unknown claim categories go through the
stored ordinal encoder's ``handle_unknown`` contract. Missing REQUIRED columns
or identifiers are hard validation errors (HTTP 400).
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd

from analytical_engine.inference import HybridModel

MAX_UPLOAD_ROWS = 5000

# Training-target columns that must never enter production inference.
_CLAIM_TARGETS = {"target", "target_label"}
_PROVIDER_TARGETS = {"potential fraud"}

# Optional identifier columns preserved when present (never modified).
_CLAIM_OPTIONAL_IDS = ("Provider_ID", "BENE_ID")


class UploadValidationError(ValueError):
    """Clear, user-facing validation failure for an uploaded file."""


# ------------------------------------------------------------------ reading
def _require_csv(filename: str | None, content_type: str | None) -> None:
    name = (filename or "").lower()
    if not name.endswith(".csv"):
        ctype = (content_type or "").lower()
        if "csv" not in ctype and "text/plain" not in ctype:
            raise UploadValidationError(
                f"Unsupported file type: '{filename or 'unknown'}'. Only .csv uploads are accepted."
            )


def _read_frame(raw: bytes, filename: str | None) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise UploadValidationError("The uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # parser errors -> clear 400, never a crash
        raise UploadValidationError(f"Malformed CSV: {exc}") from exc
    if frame.empty:
        raise UploadValidationError("The uploaded CSV contains no data rows (header only).")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise UploadValidationError(
            f"Upload too large: {len(frame)} rows (maximum {MAX_UPLOAD_ROWS} per batch)."
        )
    return frame


def _drop_targets(frame: pd.DataFrame, targets: set[str]) -> pd.DataFrame:
    drop = [c for c in frame.columns if str(c).strip().lower() in targets]
    return frame.drop(columns=drop)


def _validate_columns(frame: pd.DataFrame, model: HybridModel, label: str) -> None:
    required = list(model.feature_columns)
    present = set(frame.columns)
    missing = [c for c in required if c not in present]
    if missing:
        raise UploadValidationError(
            f"The uploaded {label} CSV is missing required model columns: "
            f"{', '.join(missing)}. Required columns: {', '.join(required)}."
        )


def _validate_identifiers(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame.columns:
        raise UploadValidationError(
            f"The uploaded {label} CSV is missing the identifier column '{column}'."
        )
    ids = frame[column]
    if ids.isna().all():
        raise UploadValidationError(f"All '{column}' values in the uploaded file are missing.")
    if ids.isna().any():
        raise UploadValidationError(
            f"{int(ids.isna().sum())} row(s) have a missing '{column}' value."
        )
    duplicated = ids[ids.duplicated(keep=False)]
    if not duplicated.empty:
        examples = ", ".join(str(v) for v in duplicated.unique()[:5])
        raise UploadValidationError(
            f"Duplicate '{column}' values in the uploaded file ({len(duplicated)} rows): {examples}."
        )


def _coerce_numerics(frame: pd.DataFrame, model: HybridModel, label: str) -> pd.DataFrame:
    """Coerce numeric feature columns; invalid values -> NaN -> artifact policy.

    A column that cannot be interpreted as numeric AT ALL is a schema error
    (it would otherwise be silently replaced by medians row after row).
    Individual bad cells follow the artifact's missing-value policy.
    """
    out = frame.copy()
    for col in model.numerical_features:
        coerced = pd.to_numeric(out[col], errors="coerce")
        had_values = out[col].notna().any()
        if had_values and coerced.isna().all():
            raise UploadValidationError(
                f"Column '{col}' in the uploaded {label} CSV contains no valid numeric values."
            )
        out[col] = coerced
    return out


# ------------------------------------------------------------------- public
def parse_claim_upload(
    raw: bytes, filename: str | None, content_type: str | None, model: HybridModel
) -> list[dict[str, Any]]:
    """Read + validate an uploaded claim CSV -> the ACTUAL uploaded records."""
    _require_csv(filename, content_type)
    frame = _read_frame(raw, filename)
    frame = _drop_targets(frame, _CLAIM_TARGETS)  # labels never enter inference
    _validate_columns(frame, model, "claim")
    _validate_identifiers(frame, "Claim_ID", "claim")
    for col in _CLAIM_OPTIONAL_IDS:  # preserved when present, never modified
        if col not in frame.columns:
            frame[col] = None
    frame = _coerce_numerics(frame, model, "claim")
    return frame.to_dict(orient="records")


def parse_provider_upload(
    raw: bytes, filename: str | None, content_type: str | None, model: HybridModel
) -> list[dict[str, Any]]:
    """Read + validate an uploaded provider CSV -> the ACTUAL uploaded records."""
    _require_csv(filename, content_type)
    frame = _read_frame(raw, filename)
    frame = _drop_targets(frame, _PROVIDER_TARGETS)  # labels never enter inference
    _validate_columns(frame, model, "provider")
    _validate_identifiers(frame, "provider_npi", "provider")
    frame = _coerce_numerics(frame, model, "provider")
    return frame.to_dict(orient="records")
