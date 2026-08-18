"""Calibration verification suite.

Tests (per phase brief):
  * calibration artifact separation (claim vs provider, versioned metadata)
  * calibrated output range bounded to [0, 1]
  * monotonicity of both calibrators
  * out-of-range input handling (clipping + NaN rejection)
  * raw ML score preservation (ML stage untouched, raw_ml_risk_score kept)
  * claim/provider isolation
  * calibration fitted ONLY on the approved held-out test splits

Run from BACKEND/:  python verify_calibration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from trust_engine import calibration as calib  # noqa: E402
from trust_engine.calibration import CalibrationError  # noqa: E402

MODELS_DIR = BACKEND_DIR / "models"
CLAIM_PATH = MODELS_DIR / calib.CLAIM_CALIBRATION_FILE
PROVIDER_PATH = MODELS_DIR / calib.PROVIDER_CALIBRATION_FILE

PASS = FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def main() -> int:
    # ============================================= 1. ARTIFACT SEPARATION
    print("\n[1] Calibration artifact separation")
    check("claim calibration artifact exists", CLAIM_PATH.exists())
    check("provider calibration artifact exists", PROVIDER_PATH.exists())
    claim_cal = calib.load_calibrator(CLAIM_PATH)
    provider_cal = calib.load_calibrator(PROVIDER_PATH)
    check("claim artifact targets the claim model",
          claim_cal.model_name == "claim_fwa_hybrid_model", claim_cal.model_name)
    check("provider artifact targets the provider model",
          provider_cal.model_name == "provider_fwa_hybrid_model", provider_cal.model_name)
    check("independent versioning (claim-calibration-v1)",
          claim_cal.calibration_version == "claim-calibration-v1")
    check("independent versioning (provider-calibration-v1)",
          provider_cal.calibration_version == "provider-calibration-v1")
    check("claim artifact calibrates P(non-legitimate)",
          claim_cal.bundle["calibrated_quantity"].startswith("P(Fraud or Waste or Abuse)"))
    check("provider artifact calibrates P(Suspicious)",
          provider_cal.bundle["calibrated_quantity"] == "P(Suspicious)")
    check("claim raw risk definition preserved in metadata",
          "(1 - P(Legitimate)) * 100" in claim_cal.bundle["raw_risk_definition"])
    check("provider raw risk definition preserved in metadata",
          "P(Suspicious) * 100" in provider_cal.bundle["raw_risk_definition"])
    check("both artifacts record a supported method",
          claim_cal.method in calib.SUPPORTED_METHODS and provider_cal.method in calib.SUPPORTED_METHODS)
    check("fitting metadata present (fitted_at, source_split, method_selection)",
          all(k in claim_cal.bundle for k in ("fitted_at", "source_split", "method_selection")))

    # ============================================= 2. OUTPUT RANGE
    print("\n[2] Calibrated output range bounded to [0, 1]")
    rng = np.random.default_rng(7)
    grid = np.concatenate([np.linspace(0.0, 1.0, 401), rng.random(2000)])
    c_claim = claim_cal.calibrate_probabilities(grid)
    c_prov = provider_cal.calibrate_probabilities(grid)
    check("claim calibrated probabilities within [0,1]",
          bool((c_claim >= 0.0).all() and (c_claim <= 1.0).all()),
          f"min={c_claim.min():.6f} max={c_claim.max():.6f}")
    check("provider calibrated probabilities within [0,1]",
          bool((c_prov >= 0.0).all() and (c_prov <= 1.0).all()),
          f"min={c_prov.min():.6f} max={c_prov.max():.6f}")
    risk = provider_cal.calibrate_risk_scores(np.array([0.0, 50.0, 100.0]))
    check("calibrated risk scores within [0,100]",
          bool((risk >= 0.0).all() and (risk <= 100.0).all()), str(risk))

    # ============================================= 3. MONOTONICITY
    print("\n[3] Monotonicity (non-decreasing calibration curves)")
    fine = np.linspace(0.0, 1.0, 5001)
    cc = claim_cal.calibrate_probabilities(fine)
    cp = provider_cal.calibrate_probabilities(fine)
    check("claim calibrator monotone non-decreasing", bool(np.diff(cc).min() >= -1e-12))
    check("provider calibrator monotone non-decreasing", bool(np.diff(cp).min() >= -1e-12))

    # ============================================= 4. OUT-OF-RANGE INPUTS
    print("\n[4] Out-of-range input handling")
    check("probability below 0 clipped to transform(0)",
          float(claim_cal.calibrate_probabilities(np.array([-0.5]))[0])
          == float(claim_cal.calibrate_probabilities(np.array([0.0]))[0]))
    check("probability above 1 clipped to transform(1)",
          float(provider_cal.calibrate_probabilities(np.array([1.5]))[0])
          == float(provider_cal.calibrate_probabilities(np.array([1.0]))[0]))
    check("risk score below 0 clipped", float(claim_cal.calibrate_risk_scores(np.array([-20.0]))[0])
          == float(claim_cal.calibrate_risk_scores(np.array([0.0]))[0]))
    check("risk score above 100 clipped", float(provider_cal.calibrate_risk_scores(np.array([120.0]))[0])
          == float(provider_cal.calibrate_risk_scores(np.array([100.0]))[0]))
    try:
        claim_cal.calibrate_probabilities(np.array([np.nan]))
        check("NaN probability rejected", False, "no error raised")
    except CalibrationError:
        check("NaN probability rejected", True)
    try:
        provider_cal.calibrate_risk_scores(np.array([np.nan]))
        check("NaN risk score rejected", False, "no error raised")
    except CalibrationError:
        check("NaN risk score rejected", True)

    # ============================================= 5. RAW SCORE PRESERVATION
    print("\n[5] Raw ML score preservation (ML stage untouched)")
    from analytical_engine.inference import ModelRegistry, predict_claims, predict_providers
    registry = ModelRegistry(MODELS_DIR / "claim_fwa_hybrid_model.pkl",
                             MODELS_DIR / "provider_fwa_hybrid_model.pkl")

    fixtures = BACKEND_DIR.parent / "test_fixtures"
    claim_records = pd.read_csv(fixtures / "test_uploaded_claims.csv").to_dict(orient="records")
    provider_records = pd.read_csv(fixtures / "test_uploaded_providers.csv").to_dict(orient="records")

    _, claim_results = predict_claims(registry.claim, claim_records)
    _, provider_results = predict_providers(registry.provider, provider_records)

    # risk_score must equal the locked formula computed from blended probabilities.
    claim_formula_ok = all(
        abs(r.risk_score - round((1.0 - r.class_probabilities["Legitimate"]) * 100.0, 1)) < 0.06
        for r in claim_results
    )
    check("claim raw risk = (1 - P(Legitimate)) x 100 (unchanged)", claim_formula_ok)
    prov_formula_ok = all(
        abs(r.risk_score - round(r.class_probabilities["Suspicious"] * 100.0, 1)) < 0.06
        for r in provider_results
    )
    check("provider raw risk = P(Suspicious) x 100 (unchanged)", prov_formula_ok)

    paired = claim_cal.paired_score(claim_results[0].risk_score)
    check("paired score keeps raw_ml_risk_score identical",
          paired.raw_ml_risk_score == claim_results[0].risk_score)
    check("paired score adds calibrated_risk_score separately",
          paired.calibrated_risk_score is not None and 0.0 <= paired.calibrated_risk_score <= 100.0)
    check("MLResult frozen object not modified by calibration",
          claim_results[0].risk_score == paired.raw_ml_risk_score)

    # Calibration must not be reachable from the Analytical Engine.
    engine_src = (BACKEND_DIR / "analytical_engine").rglob("*.py")
    imports_cal = any("calibration" in p.read_text(encoding="utf-8") for p in engine_src)
    check("Analytical Engine does not import calibration", not imports_cal)

    # ============================================= 6. CLAIM/PROVIDER ISOLATION
    print("\n[6] Claim/provider calibration isolation")
    sample = np.array([0.2, 0.5, 0.8])
    a = claim_cal.calibrate_probabilities(sample)
    b = provider_cal.calibrate_probabilities(sample)
    check("same raw probabilities map differently per entity (separate fits)",
          bool(np.abs(a - b).max() > 1e-3), f"claim={a} provider={b}")
    check("artifacts are distinct objects with distinct versions",
          claim_cal.calibration_version != provider_cal.calibration_version)
    check("claim metadata carries no provider split info",
          "provider" not in str(claim_cal.source_split.get("dataset", "")).lower())
    check("provider metadata carries no claim split info",
          "claim" not in str(provider_cal.source_split.get("dataset", "")).lower())

    # ================================= 7. APPROVED HELD-OUT SPLIT ONLY
    print("\n[7] Calibration uses ONLY the approved held-out test splits")
    cs = claim_cal.source_split
    ps = provider_cal.source_split
    check("claim split role is the TEST split", cs.get("role") == "test")
    check("claim split is Claim_Year == 2023 (temporal test)",
          "Claim_Year == 2023" in cs.get("split_definition", ""))
    check("claim split flagged held-out from training", cs.get("held_out_from_training") is True)
    check("SMOTE never applied to the claim calibration split", cs.get("smote_applied_to_split") is False)
    check("claim split never used for ratio selection", cs.get("used_for_ensemble_ratio_selection") is False)

    check("provider split role is the TEST split", ps.get("role") == "test")
    check("provider split reproduces the training-script 70/15/15 test split",
          "70/15/15" in ps.get("split_definition", "") and "random_state=42" in ps.get("split_definition", ""))
    check("provider split flagged held-out from training", ps.get("held_out_from_training") is True)
    check("provider split never used for ratio selection", ps.get("used_for_ensemble_ratio_selection") is False)

    # Independently recompute the held-out split sizes from the datasets and
    # compare against the recorded fit sizes (proof the fit used those rows).
    from sklearn.model_selection import train_test_split
    years = pd.read_csv(BACKEND_DIR / "data" / "claimsfinal_with_target.csv", usecols=["Claim_Year"])
    n_2023 = int((years["Claim_Year"] == 2023).sum())
    check("claim calibration size matches dataset 2023 rows", cs.get("n_samples") == n_2023,
          f"{cs.get('n_samples')} vs {n_2023}")

    y_all = pd.read_csv(BACKEND_DIR / "data" / "Provider_with_potential_fraud.csv",
                        usecols=["Potential Fraud"])["Potential Fraud"].astype(np.int8).to_numpy()
    idx = np.arange(len(y_all))
    _, idx_temp = train_test_split(idx, test_size=0.30, stratify=y_all, random_state=42)
    _, idx_test = train_test_split(idx_temp, test_size=0.50, stratify=y_all[idx_temp], random_state=42)
    check("provider calibration size matches reproduced test split", ps.get("n_samples") == len(idx_test),
          f"{ps.get('n_samples')} vs {len(idx_test)}")

    check("method selection kept 40% of the held-out split untouched",
          claim_cal.bundle["method_selection"]["selection_fraction"] == 0.40)

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
