"""Central configuration and path resolution for the FWA backend.

Everything is resolved relative to the repository layout:

    BACKEND/
        data/
            claimsfinal_with_target.csv
            Provider_with_potential_fraud.csv
        models/
            claim_fwa_hybrid_model.pkl
            provider_fwa_hybrid_model.pkl
"""
from __future__ import annotations

import os
from pathlib import Path

# BACKEND/ directory (this file lives in BACKEND/app/config.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"

CLAIM_MODEL_PATH = MODELS_DIR / "claim_fwa_hybrid_model.pkl"
PROVIDER_MODEL_PATH = MODELS_DIR / "provider_fwa_hybrid_model.pkl"

# Versioned calibration artifacts (Phase 1) — separate per model, strictly
# post-processing, never a replacement for the raw ML risk score.
CLAIM_CALIBRATION_PATH = MODELS_DIR / "claim_calibration_v1.pkl"
PROVIDER_CALIBRATION_PATH = MODELS_DIR / "provider_calibration_v1.pkl"

CLAIM_DATA_PATH = DATA_DIR / "claimsfinal_with_target.csv"
PROVIDER_DATA_PATH = DATA_DIR / "Provider_with_potential_fraud.csv"

# Identifier columns used to look up full feature rows from the datasets.
CLAIM_ID_COLUMN = "Claim_ID"
PROVIDER_ID_COLUMN = "provider_npi"

# CORS — allow the local Vite dev server. Can be overridden via environment.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FWA_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8080,http://127.0.0.1:8080,http://localhost:8081,http://127.0.0.1:8081",
    ).split(",")
    if origin.strip()
]
