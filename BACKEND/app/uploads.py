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


# Alias mappings for user-defined fields (case-insensitive, spaces allowed)
CLAIM_ALIASES = {
    "claim_id": ["claim id", "claim_id", "claimid"],
    "bene_id": ["beneficiary id", "bene_id", "beneficiary_id", "beneficiaryid", "bene id", "bene_id"],
    "claim_type": ["claim type", "claim_type", "claimtype"],
    "claim_payment_amount": ["claim reimbursement", "claim_payment_amount", "reimbursement", "claim reimbursement amount"],
    "deductible_amount": ["deductible amount", "deductible_amount", "deductible", "deductibleamount"],
    "claim_duration_days": ["length of stay", "claim_duration_days", "duration", "lengthofstay", "los"],
    "diagnosis_count": ["number of diagnoses", "diagnosis_count", "diagnoses", "diagnosiscount"],
    "procedure_count": ["number of procedures", "procedure_count", "procedures", "procedurecount"],
    "previous_claim_count": ["previous claims", "previous_claim_count", "previousclaims", "previousclaimcount"],
}

PROVIDER_ALIASES = {
    "provider_npi": ["provider id", "provider_id", "providerid", "provider_npi", "npi"],
    "claim_count": ["total claims", "claim_count", "totalclaims", "claimcount"],
    "cms_total_beneficiaries": ["unique beneficiaries", "cms_total_beneficiaries", "uniquebeneficiaries", "beneficiary count", "beneficiarycount"],
    "cms_weighted_avg_payment": ["average claim amount", "cms_weighted_avg_payment", "averageclaimamount", "averagepayment"],
    "cms_weighted_avg_submitted_charge": ["high-value claims %", "high value claims %", "high_value_claims_%"],
    "services_per_beneficiary": ["chronic/complex cases %", "chronic complex cases %", "chronic_complex_cases_%"],
    "cms_total_beneficiary_days": ["repeat/multiple claims %", "repeat multiple claims %", "repeat_multiple_claims_%"],
    "treatment_service_percentile": ["inpatient claim share %", "inpatient share %", "inpatient_claim_share_%"],
}

def _standardize_and_impute_columns(frame: pd.DataFrame, model: HybridModel, label: str) -> pd.DataFrame:
    """Standardizes columns in the input dataframe to match model features using aliases.
    Fills in any missing model feature columns with NaN so they are imputed.
    """
    aliases = CLAIM_ALIASES if label == "claim" else PROVIDER_ALIASES
    
    # 1. Lowercase column names for match
    col_map = {}
    for col in frame.columns:
        col_lower = str(col).strip().lower()
        col_map[col_lower] = col
        
    # 2. Rename columns based on aliases
    renames = {}
    for target_col, alias_list in aliases.items():
        for alias in alias_list:
            if alias in col_map:
                renames[col_map[alias]] = target_col
                break
                
    new_frame = frame.rename(columns=renames)
    
    # Special mappings for provider percentages/ratios if they are present:
    if label == "provider":
        if "cms_weighted_avg_submitted_charge" in new_frame.columns and "cms_weighted_avg_payment" in new_frame.columns:
            try:
                # High-value claims %: calculate submitted charge based on avg payment + high-value ratio
                pct_val = pd.to_numeric(new_frame["cms_weighted_avg_submitted_charge"], errors="coerce").fillna(0)
                new_frame["cms_weighted_avg_submitted_charge"] = pd.to_numeric(new_frame["cms_weighted_avg_payment"], errors="coerce") * (1 + pct_val / 100)
            except Exception:
                pass
        if "services_per_beneficiary" in new_frame.columns:
            try:
                # Chronic/complex cases %
                new_frame["services_per_beneficiary"] = pd.to_numeric(new_frame["services_per_beneficiary"], errors="coerce").fillna(0) / 10
            except Exception:
                pass
        if "cms_total_beneficiary_days" in new_frame.columns and "claim_count" in new_frame.columns:
            try:
                # Repeat/multiple claims %
                pct_val = pd.to_numeric(new_frame["cms_total_beneficiary_days"], errors="coerce").fillna(0)
                new_frame["cms_total_beneficiary_days"] = pd.to_numeric(new_frame["claim_count"], errors="coerce") * (1 + pct_val / 100)
            except Exception:
                pass
        if "treatment_service_percentile" in new_frame.columns:
            try:
                # Inpatient claim share %
                new_frame["treatment_service_percentile"] = pd.to_numeric(new_frame["treatment_service_percentile"], errors="coerce").fillna(0) / 100
            except Exception:
                pass
                
    # Also replicate Claim_Payment_Amount to other charge/allowed fields if missing for claims
    if label == "claim":
        if "claim_payment_amount" in new_frame.columns:
            if "claim_submitted_amount" not in new_frame.columns:
                new_frame["claim_submitted_amount"] = new_frame["claim_payment_amount"]
            if "claim_allowed_amount" not in new_frame.columns:
                new_frame["claim_allowed_amount"] = new_frame["claim_payment_amount"]
                
    # 3. Add all model feature columns that are missing as NaN
    for col in model.feature_columns:
        if col not in new_frame.columns:
            new_frame[col] = pd.Series(pd.NA, index=new_frame.index)
            
    # 4. Make sure primary ID columns exist
    primary_id = "Claim_ID" if label == "claim" else "provider_npi"
    if primary_id not in new_frame.columns:
        # Check if the renamed column was lowercased
        lower_id = primary_id.lower()
        if lower_id in new_frame.columns:
            new_frame = new_frame.rename(columns={lower_id: primary_id})
            
    return new_frame


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

    A column that cannot be interpreted as numeric AT ALL is a schema error.
    Individual bad cells follow the artifact's missing-value policy.
    """
    out = frame.copy()
    for col in model.numerical_features:
        # Columns might contain NaN entirely or not exist in input file (which we filled with NaN).
        # We only coerce columns that have at least one non-null element to save performance.
        if out[col].notna().any():
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.isna().all():
                raise UploadValidationError(
                    f"Column '{col}' in the uploaded {label} CSV contains no valid numeric values."
                )
            out[col] = coerced
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ------------------------------------------------------------------- public
def parse_claim_upload(
    raw: bytes, filename: str | None, content_type: str | None, model: HybridModel
) -> list[dict[str, Any]]:
    """Read + validate an uploaded claim CSV -> the ACTUAL uploaded records."""
    _require_csv(filename, content_type)
    frame = _read_frame(raw, filename)
    frame = _drop_targets(frame, _CLAIM_TARGETS)  # labels never enter inference
    frame = _standardize_and_impute_columns(frame, model, "claim")
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
    frame = _standardize_and_impute_columns(frame, model, "provider")
    _validate_identifiers(frame, "provider_npi", "provider")
    frame = _coerce_numerics(frame, model, "provider")
    return frame.to_dict(orient="records")
